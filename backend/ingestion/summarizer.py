import re
import json
import logging
from typing import Dict, Any, Optional
from backend.config import settings
from backend.ingestion.pdf_parser import BookNode

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio for grok-3 (conservative)
_CHARS_PER_TOKEN = 4


def _cap_text(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget (char approximation)."""
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) > max_chars:
        return text[:max_chars] + "\n[...truncated for context budget...]"
    return text


class NodeSummarizer:
    """
    Hierarchical content-aware summarizer for Book → Part/Chapter → Section → Subsection.
    
    - Receives an LLM client at construction time so LLM summaries are generated using
      the same grok-3 client as the query pipeline (not a no-client fallback).
    - All text fed to the LLM is token-budget-capped to prevent context overflow.
    - Falls back to high-quality rule-based extraction if LLM is unavailable.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def summarize_node(self, node: BookNode, book_title: str) -> Dict[str, Any]:
        if node.level == "book":
            return self._summarize_book(node)
        elif node.level in ["part", "chapter"]:
            return self._summarize_chapter(node, book_title)
        elif node.level == "section":
            return self._summarize_section(node, book_title)
        else:
            return self._summarize_subsection(node, book_title)

    # ── Book-level (150-300 words) ─────────────────────────────────────────────

    def _summarize_book(self, node: BookNode) -> Dict[str, Any]:
        # Budget: TOKEN_BUDGET_BOOK_ROUTING / 28 books ≈ 285 tokens of text sample per book
        text_sample = _cap_text(node.raw_text, max_tokens=700)
        prompt = f"""Analyze the personal finance book '{node.title}'.
Raw Text Sample:
{text_sample}

Provide a JSON object with:
1. "summary": A 150-300 word overview of the core principles, framework, and key concepts in this book.
2. "topic_tags": Array of 3-7 relevant tags from: ["budgeting", "investing", "retirement", "debt", "real estate", "behavioral finance", "taxes", "assets vs liabilities", "FIRE", "index funds", "wealth mindset", "frugality", "automation", "value investing", "growth investing"].
3. "stance": A 1-2 sentence description of the author's distinctive philosophy or approach (e.g. "Anti-debt, cash-envelope system" or "Leverage-friendly, index fund buy-and-hold").
4. "audience": Who this book is primarily written for (1 sentence).

Output strictly valid JSON with no markdown fencing:
"""
        res = self._call_llm(prompt)
        parsed = self._parse_json(res)
        if parsed and all(k in parsed for k in ["summary", "topic_tags", "stance", "audience"]):
            return parsed

        # High-quality fallback: semantic keyword extraction
        text_sample_lower = node.raw_text[:3000].lower()
        tags = ["financial literacy", "wealth building"]
        if any(w in text_sample_lower for w in ["asset", "liability", "cashflow"]):
            tags.extend(["assets vs liabilities", "cashflow"])
        if "debt" in text_sample_lower:
            tags.append("debt")
        if "tax" in text_sample_lower:
            tags.append("taxes")
        if any(w in text_sample_lower for w in ["index fund", "vtsax", "s&p", "etf"]):
            tags.extend(["index funds", "investing"])
        if any(w in text_sample_lower for w in ["retire", "retirement", "401k", "ira"]):
            tags.append("retirement")
        if any(w in text_sample_lower for w in ["real estate", "rental", "property"]):
            tags.append("real estate")
        if any(w in text_sample_lower for w in ["budget", "spending", "expense"]):
            tags.append("budgeting")

        return {
            "summary": (
                f"{node.title} presents personal finance principles focused on building wealth "
                f"through disciplined financial habits, strategic investment, and long-term thinking. "
                f"The author provides actionable frameworks for managing income, expenses, and assets."
            ),
            "topic_tags": list(set(tags))[:6],
            "stance": "Disciplined asset acquisition and long-term financial management.",
            "audience": "Individual investors and personal finance enthusiasts.",
        }

    # ── Chapter-level (80-150 words) ───────────────────────────────────────────

    def _summarize_chapter(self, node: BookNode, book_title: str) -> Dict[str, Any]:
        text_sample = _cap_text(node.raw_text, max_tokens=settings.TOKEN_BUDGET_CHAPTER_SELECT // 2)
        prompt = f"""Summarize the chapter '{node.title}' from '{book_title}' (Pages {node.start_page}-{node.end_page}).

Text sample:
{text_sample}

Write an 80-150 word summary highlighting:
- The key concept or lesson of this chapter
- Any specific rules, formulas, or frameworks introduced (e.g. "Debt Snowball", "Pay Yourself First 10%", "Asset vs Liability Rule")
- Concrete actionable takeaways

Output strictly JSON: {{"summary": "..."}}
"""
        res = self._call_llm(prompt)
        parsed = self._parse_json(res)
        if parsed and parsed.get("summary"):
            return {"summary": parsed["summary"]}

        # Semantic extraction fallback (Rich Dad-aware + general)
        return {"summary": self._extract_chapter_summary(node, book_title)}

    # ── Section-level (30-60 words) ────────────────────────────────────────────

    def _summarize_section(self, node: BookNode, book_title: str) -> Dict[str, Any]:
        text_sample = _cap_text(node.raw_text, max_tokens=settings.TOKEN_BUDGET_SECTION_SELECT // 2)
        prompt = f"""Summarize the section '{node.title}' from '{book_title}' (Pages {node.start_page}-{node.end_page}).

Text sample:
{text_sample}

Write a 30-60 word summary of the key concept or tactic covered in this section.
Output strictly JSON: {{"summary": "..."}}
"""
        res = self._call_llm(prompt)
        parsed = self._parse_json(res)
        if parsed and parsed.get("summary"):
            return {"summary": parsed["summary"]}

        return {"summary": self._extract_section_summary(node, book_title)}

    # ── Subsection-level (15-30 words) ─────────────────────────────────────────

    def _summarize_subsection(self, node: BookNode, book_title: str) -> Dict[str, Any]:
        text_sample = _cap_text(node.raw_text, max_tokens=200)
        prompt = f"""In 15-30 words, summarize the subsection '{node.title}' from '{book_title}' (Pages {node.start_page}-{node.end_page}).
Text: {text_sample}
Output strictly JSON: {{"summary": "..."}}
"""
        res = self._call_llm(prompt)
        parsed = self._parse_json(res)
        if parsed and parsed.get("summary"):
            return {"summary": parsed["summary"]}

        return {
            "summary": (
                f"Subsection '{node.title}' (pp. {node.start_page}-{node.end_page}): "
                f"Tactical guidance on {node.title.lower()}."
            )
        }

    # ── Rule-based Fallback Extractors ─────────────────────────────────────────

    def _extract_chapter_summary(self, node: BookNode, book_title: str) -> str:
        raw = node.raw_text.lower()
        title_lower = node.title.lower()
        key_topics = []

        # Rich Dad Poor Dad specific
        if "lesson 2" in title_lower or (
            "asset" in raw and "liability" in raw and ("puts money" in raw or "rule #1" in raw)
        ):
            key_topics.append(
                "Rule #1: The rich buy assets that put money in their pocket; "
                "the poor buy liabilities mistaken for assets. "
                "Defines Asset vs Liability via cashflow diagrams."
            )
        elif "lesson 1" in title_lower or "rich don't work for money" in raw:
            key_topics.append(
                "Lesson 1: The rich don't work for money — they make money work for them. "
                "Overcoming fear and desire through financial education."
            )
        elif "lesson 3" in title_lower or "mind your own business" in raw:
            key_topics.append(
                "Lesson 3: Mind your own business — keep your day job but build your asset column."
            )
        elif "lesson 4" in title_lower or "history of taxes" in raw:
            key_topics.append(
                "Lesson 4: History of taxes and power of corporations — using corporate structure "
                "to legally minimize taxes and protect wealth."
            )
        elif "lesson 5" in title_lower or "rich invent money" in raw:
            key_topics.append("Lesson 5: The rich invent money — using creativity and financial IQ.")
        elif "lesson 6" in title_lower or "work to learn" in raw:
            key_topics.append(
                "Lesson 6: Work to learn, not for money — develop skills in sales, management, accounting."
            )
        # General finance patterns
        elif "index fund" in raw or "bogle" in raw or "vtsax" in raw:
            key_topics.append(
                "Low-cost total stock market index fund investing, buy-and-hold strategy, minimizing fees."
            )
        elif "baby steps" in raw or "debt snowball" in raw:
            key_topics.append(
                "Dave Ramsey's 7 Baby Steps — starting with $1,000 emergency fund, "
                "then Debt Snowball (smallest balance first), then investing 15%."
            )
        elif "compound" in raw or "compounding" in raw:
            key_topics.append("The power of compound interest and time in the market.")
        elif "emergency fund" in raw:
            key_topics.append("Building a 3-6 month emergency fund before investing.")
        else:
            first_lines = " ".join(
                [l.strip() for l in node.raw_text.splitlines() if len(l.strip()) > 30][:3]
            )
            key_topics.append(first_lines[:300] if first_lines else "Core financial principles.")

        return (
            f"Chapter '{node.title}' in {book_title} "
            f"(pp. {node.start_page}-{node.end_page}): "
            f"{'; '.join(key_topics)}."
        )

    def _extract_section_summary(self, node: BookNode, book_title: str) -> str:
        raw = node.raw_text.lower()
        key_topics = []

        if "asset" in raw and "liability" in raw:
            key_topics.append(
                "Asset vs Liability distinction: assets put money in your pocket, "
                "liabilities take money out."
            )
        elif "cashflow" in raw or "cash flow" in raw:
            key_topics.append("Cashflow quadrant: income vs expense flow analysis.")
        elif "budget" in raw and "envelope" in raw:
            key_topics.append("Zero-based budgeting using cash envelopes.")
        elif "compound" in raw:
            key_topics.append("Compound interest growth mechanics.")

        if key_topics:
            return (
                f"Section '{node.title}' (pp. {node.start_page}-{node.end_page}): "
                f"{'; '.join(key_topics)}."
            )
        return (
            f"Section '{node.title}' (pp. {node.start_page}-{node.end_page}): "
            f"Core concepts and tactical application."
        )

    # ── LLM Helpers ────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.llm_client:
            return None
        try:
            if hasattr(self.llm_client, "invoke"):
                res = self.llm_client.invoke(prompt)
                return res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            logger.warning(f"NodeSummarizer LLM call failed: {e}")
        return None

    def _parse_json(self, response: Optional[str]) -> Optional[Dict[str, Any]]:
        if not response:
            return None
        try:
            cleaned = response.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            return json.loads(cleaned)
        except Exception:
            return None
