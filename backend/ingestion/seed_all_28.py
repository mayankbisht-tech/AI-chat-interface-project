import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import settings
from backend.ingestion.pdf_parser import PDFBookParser
from backend.ingestion.summarizer import NodeSummarizer
from backend.ingestion.storage import CorpusStorage
from backend.ingestion.ingest_all import BOOK_METADATA_PRESETS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def seed_all(llm_client=None) -> Dict[str, Any]:
    """
    Idempotent ingestion pipeline.
    - Skips books already fully ingested with non-empty LLM summaries at all levels.
    - Passes the LLM client into NodeSummarizer so summaries are LLM-generated (not fallback).
    - Returns a summary dict with counts for reporting.
    """
    pdf_dir = settings.PDF_DIR
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found at {pdf_dir}")

    storage = CorpusStorage()
    summarizer = NodeSummarizer(llm_client=llm_client)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF books. LLM client: {'YES' if llm_client else 'NO (fallback mode)'}")

    skipped = 0
    processed = 0
    failed = 0

    for idx, pdf_path in enumerate(pdf_files, 1):
        filename_stem = pdf_path.stem

        # Match preset metadata
        preset = None
        for key, val in BOOK_METADATA_PRESETS.items():
            if key.lower() in filename_stem.lower() or filename_stem.lower() in key.lower():
                preset = val
                break

        # ── Idempotency Check ──────────────────────────────────────────────────
        # Determine the expected book_id for this file (matches PDFBookParser logic)
        import re
        clean_title = re.sub(r'\(.*?\)|\[.*?\]', '', filename_stem)
        clean_title = clean_title.replace('_', ' ').replace('-', ' ').strip()
        clean_title = re.sub(r'\s+', ' ', clean_title)
        book_id = f"book_{re.sub(r'[^a-zA-Z0-9]', '_', clean_title).lower()}"

        existing_node = storage.get_node(book_id)
        if existing_node and existing_node.get("summary") and len(existing_node["summary"]) > 80:
            # Also check children have summaries
            children = storage.get_children(book_id)
            children_ok = all(
                c.get("summary") and len(c["summary"]) > 20 for c in children
            ) if children else True

            if children_ok:
                logger.info(f"[{idx}/{len(pdf_files)}] SKIP (already ingested): {pdf_path.name}")
                skipped += 1
                continue

        logger.info(f"[{idx}/{len(pdf_files)}] Ingesting: {pdf_path.name}")

        try:
            parser = PDFBookParser(pdf_path)
            root_node = parser.parse()

            # ── Book-level metadata ────────────────────────────────────────────
            if preset:
                root_node.title = preset["title"]
                stance = preset["stance"]
                audience = preset["audience"]
                topic_tags = preset["topic_tags"]
                author = preset.get("author", "")
                full_summary = (
                    f"{preset['title']} by {author}. "
                    f"Core Stance: {stance}. "
                    f"Target Audience: {audience}."
                )
            else:
                book_summary_info = summarizer.summarize_node(root_node, root_node.title)
                stance = book_summary_info.get("stance", "Disciplined financial management")
                audience = book_summary_info.get("audience", "Individual investors")
                topic_tags = book_summary_info.get("topic_tags", ["investing", "wealth management"])
                full_summary = book_summary_info.get(
                    "summary", f"{root_node.title} covers essential personal finance principles."
                )

            root_node.summary = full_summary
            root_node.summary_dict = {
                "summary": full_summary,
                "topic_tags": topic_tags,
                "stance": stance,
                "audience": audience,
            }

            # ── Recursively summarize chapters & sections with LLM ─────────────
            def summarize_tree(node):
                for child in node.children:
                    # Idempotency: skip if child already has a good summary in DB
                    existing_child = storage.get_node(child.node_id)
                    if (
                        existing_child
                        and existing_child.get("summary")
                        and len(existing_child["summary"]) > 20
                    ):
                        child.summary = existing_child["summary"]
                        child.summary_dict = {
                            "summary": child.summary,
                            "topic_tags": topic_tags,
                            "stance": stance,
                        }
                    else:
                        c_sum_info = summarizer.summarize_node(child, root_node.title)
                        child.summary = c_sum_info.get("summary", "")
                        child.summary_dict = {
                            "summary": child.summary,
                            "topic_tags": topic_tags,
                            "stance": stance,
                        }
                    summarize_tree(child)

            summarize_tree(root_node)
            storage.save_tree(root_node)

            logger.info(
                f"[{idx}/{len(pdf_files)}] Stored '{root_node.title}' — "
                f"{len(root_node.children)} chapters."
            )
            processed += 1

        except Exception as e:
            logger.error(f"[{idx}/{len(pdf_files)}] FAILED: {pdf_path.name} — {e}", exc_info=True)
            failed += 1

    # Export updated Corpus Index
    index_path = storage.export_corpus_index()
    result = {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_books": len(pdf_files),
        "index_path": str(index_path),
        "llm_used": llm_client is not None,
    }
    logger.info(f"Ingestion complete: {result}")
    return result


if __name__ == "__main__":
    # When run directly, attempt to load the LLM client from environment
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from backend.main import get_llm_client
    client = get_llm_client()
    seed_all(llm_client=client)
