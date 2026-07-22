"""
Guardrails for the Personal Finance Assistant.

Two layers:

1. HARD-BLOCKED topics — returns an immediate polite refusal with no LLM call.
   These cover 15 dangerous / clearly off-topic categories.

2. OFF-TOPIC soft check — uses fast regex + keyword matching to decide whether
   a query is about personal finance.  Queries that are pure social chit-chat
   (except simple greetings) get a polite redirect.

The module is intentionally dependency-free (no LLM call) so it adds near-zero
latency to the critical path.
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger("Guardrails")

# ── 1. HARD-BLOCKED patterns ────────────────────────────────────────────────
# Each entry: (label, [regex patterns])
# If ANY pattern matches the query (case-insensitive), the request is blocked.

HARD_BLOCKED: list[tuple[str, list[str]]] = [
    (
        "violent_crime",
        [
            r"\b(rob|robbing|robbery)\b.*\b(bank|store|house|someone)\b",
            r"\b(steal|theft|thief|mug|mugging|carjack)\b",
            r"\b(kill|murder|assassin|shoot|stab|bomb|explode)\b",
            r"\bhow\s+to\s+(hurt|harm|attack|assault)\b",
        ],
    ),
    (
        "financial_crime",
        [
            r"\b(launder|money.{0,10}launder|launder.{0,10}money)\b",
            r"\b(ponzi|pyramid.{0,6}scheme|fraud|scam.{0,6}(run|start|create|build))\b",
            r"\b(counterfeit|fake.{0,6}(currency|bills|notes|money))\b",
            r"\b(tax.{0,6}evasion|evade.{0,6}tax|hide.{0,10}(income|assets|money).{0,10}(irs|government|authorities))\b",
            r"\b(insider.{0,6}trad(e|ing)|front.{0,6}run(ning)?)\b",
            r"\b(embezzle|embezzlement|misappropriat)\b",
        ],
    ),
    (
        "hacking_cybercrime",
        [
            r"\b(hack|hacking|cracking|exploit)\b.{0,20}\b(bank|account|password|system|atm)\b",
            r"\b(phish(ing)?|social.{0,6}engineer|keylogger|malware|ransomware)\b",
            r"\b(dark.?web|darknet).{0,30}\b(buy|sell|purchase|get)\b",
        ],
    ),
    (
        "drug_manufacturing_dealing",
        [
            r"\b(synthesize|manufacture|make|cook|produce).{0,20}\b(drug|meth|heroin|cocaine|fentanyl|lsd|mdma)\b",
            r"\b(buy|sell|deal|traffic).{0,20}\b(drug|narcotics|controlled.{0,6}substance)\b",
        ],
    ),
    (
        "weapons",
        [
            r"\b(make|build|create|buy|acquire|3d.{0,6}print).{0,20}\b(gun|firearm|weapon|explosive|bomb|grenade)\b",
            r"\b(illegal.{0,6}weapon|untraceable.{0,6}gun|ghost.{0,6}gun)\b",
        ],
    ),
    (
        "self_harm",
        [
            r"\b(suicide|kill\s+myself|end\s+my\s+life|self.{0,6}harm|cut\s+myself)\b",
            r"\b(want\s+to\s+die|no\s+reason\s+to\s+live|better\s+off\s+dead)\b",
        ],
    ),
    (
        "hate_speech",
        [
            r"\b(n[i1]gg[ae]r|ch[i1]nk|sp[i1]c|k[i1]ke|f[a4]gg[o0]t)\b",
            r"\b(white.{0,10}superior|racial.{0,6}purity|ethnic.{0,6}cleans)\b",
        ],
    ),
    (
        "sexual_exploitation",
        [
            r"\b(child.{0,10}(porn|abuse|exploit|nude|sexual))\b",
            r"\b(minor.{0,10}(sexual|nude|porn|exploit))\b",
            r"\b(cp\b|csam)\b",
        ],
    ),
    (
        "identity_theft",
        [
            r"\b(steal.{0,15}(identity|ssn|social.{0,6}security|credit.{0,6}card.{0,6}number))\b",
            r"\b(fake.{0,10}(id|passport|drivers.{0,6}license|social.{0,6}security))\b",
            r"\b(clone.{0,10}(credit.{0,6}card|sim|phone))\b",
        ],
    ),
    (
        "market_manipulation",
        [
            r"\b(pump.{0,6}and.{0,6}dump|wash.{0,6}trad(e|ing)|manipulat.{0,10}(stock|market|price))\b",
            r"\b(short.{0,6}sell.{0,10}naked|naked.{0,6}short)\b",
        ],
    ),
    (
        "illegal_gambling",
        [
            r"\b(run.{0,10}illegal.{0,10}(casino|gambling|betting))\b",
            r"\b(fix.{0,10}(match|game|race|fight|sport).{0,10}(bet|gambl))\b",
        ],
    ),
    (
        "extortion_blackmail",
        [
            r"\b(blackmail|extort|ransom(ware)?).{0,20}\b(someone|person|company|business)\b",
        ],
    ),
    (
        "unregulated_investment_advice",
        [
            r"\b(guarantee(d)?.{0,15}return|risk.{0,6}free.{0,10}(profit|return|investment|gain))\b",
            r"\b(get.{0,6}rich.{0,6}quick.{0,6}(scheme|trick|secret|system))\b",
        ],
    ),
    (
        "terrorism_extremism",
        [
            r"\b(terror(ism|ist)|jihadist|extremist).{0,20}\b(fund|financ|support|join)\b",
            r"\b(fund.{0,10}terrori|financ.{0,10}terrori)\b",
        ],
    ),
    (
        "human_trafficking",
        [
            r"\b(traffick.{0,10}(human|people|women|children)|human.{0,10}traffick)\b",
            r"\b(smuggl.{0,10}(human|people|migrant))\b",
        ],
    ),
]

# Compile all patterns once at import time
_COMPILED_BLOCKS: list[tuple[str, list[re.Pattern]]] = [
    (label, [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns])
    for label, patterns in HARD_BLOCKED
]

# ── 2. FINANCE RELEVANCE check ──────────────────────────────────────────────

# Keywords that strongly indicate a personal-finance question
FINANCE_KEYWORDS = {
    # core concepts
    "money", "finance", "financial", "budget", "budgeting", "invest", "investing",
    "investment", "investor", "saving", "savings", "debt", "loan", "mortgage",
    "interest", "rate", "compound", "compound interest", "inflation", "tax", "taxes",
    "income", "expense", "expenses", "salary", "wage", "earn", "earning", "revenue",
    "profit", "loss", "asset", "assets", "liability", "liabilities", "net worth",
    "cashflow", "cash flow", "portfolio", "stock", "stocks", "bond", "bonds",
    "etf", "index fund", "mutual fund", "dividend", "equity", "real estate",
    "retirement", "401k", "ira", "roth", "pension", "social security",
    "credit", "credit card", "credit score", "debit", "bank", "banking",
    "insurance", "premium", "deductible", "wealth", "rich", "poor", "afford",
    "spend", "spending", "frugal", "frugality", "cheap", "cost", "price",
    "pay", "payment", "paycheck", "payoff", "debt free", "fire", "financial independence",
    "early retirement", "side hustle", "passive income", "emergency fund",
    "dollar cost averaging", "rebalance", "diversify", "diversification",
    "hedge", "hedge fund", "short sell", "option", "futures", "crypto",
    "bitcoin", "ethereum", "nft", "annuity", "robo advisor", "fiduciary",
    "brokerage", "trading", "trader", "market", "bull", "bear", "recession",
    "gdp", "fed", "federal reserve", "central bank", "quantitative easing",
    "vtsax", "bogle", "graham", "buffett", "ramsey", "kiyosaki", "sethi",
    "snowball", "avalanche", "baby steps", "latte factor",
}

# Simple greetings AND short follow-up questions — always allowed
GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|howdy|good\s+(morning|afternoon|evening|day)|what'?s\s+up|greetings"
    r"|thanks?|thank\s+you|ok|okay|got\s+it|sounds\s+good|great|sure|yep|nope|yes|no|bye|goodbye"
    r"|nice|cool|awesome|wow|hm+|ah+|who\s+am\s+i|what\s+do\s+you\s+know\s+about\s+me"
    r"|what\s+did\s+i\s+say|remind\s+me|what\s+was\s+my|tell\s+me\s+more|why|how|explain"
    r"|can\s+you\s+elaborate|go\s+on|continue|and\s+then|what\s+about|what\s+next)\W*$",
    re.IGNORECASE,
)

# Patterns that flag clear off-topic intent
OFF_TOPIC_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(recipe|cook|food|restaurant|menu|dish|cuisine|ingredient)\b",
        r"\b(movie|film|series|netflix|show|actor|actress|celebrity|pop\s+star)\b",
        r"\b(sport|football|cricket|basketball|soccer|match|score|player|team)\b",
        r"\b(weather|forecast|rain|sunny|temperature|climate)\b",
        r"\b(relationship|breakup|divorce|love|dating|girlfriend|boyfriend|marriage|wedding)\b",
        r"\b(homework|assignment|essay|history|geography|physics|chemistry|biology)\b",
        r"\b(game|gaming|video\s+game|playstation|xbox|nintendo|minecraft|fortnite)\b",
        r"\b(travel|vacation|holiday|tour|flight|hotel|booking)\b",
        r"\b(health|doctor|medicine|symptom|disease|illness|diagnos)\b",
        r"\b(legal\s+advice|lawsuit|sue|attorney|lawyer|court)\b",
    ]
]


def check(query: str) -> Tuple[bool, str]:
    """
    Returns (is_blocked: bool, response_message: str).

    - (True, message)  → block the query and stream this message to the user
    - (False, "")      → query is fine, proceed normally
    """
    query_stripped = query.strip()

    # ── Hard block check ───────────────────────────────────────────────────
    for label, patterns in _COMPILED_BLOCKS:
        for pattern in patterns:
            if pattern.search(query_stripped):
                logger.warning(f"[Guardrails] HARD BLOCK triggered — category: {label!r}")
                return True, _hard_block_message(label)

    # ── Greeting pass-through ──────────────────────────────────────────────
    if GREETING_PATTERNS.match(query_stripped):
        return False, ""

    # ── Finance relevance check ────────────────────────────────────────────
    lower = query_stripped.lower()
    words = set(re.findall(r"\b\w+\b", lower))

    has_finance_keyword = bool(words & FINANCE_KEYWORDS)
    if has_finance_keyword:
        return False, ""

    # Check for off-topic signals only when no finance keyword is present
    for pat in OFF_TOPIC_PATTERNS:
        if pat.search(lower):
            logger.info("[Guardrails] Off-topic query redirected.")
            return True, _off_topic_message()

    # Short queries with no finance keywords and no obvious off-topic signal
    # are let through — they're almost always follow-up questions
    if len(query_stripped.split()) <= 10:
        return False, ""

    # Longer queries with no finance keywords — soft off-topic redirect
    logger.info("[Guardrails] Soft off-topic redirect for longer non-finance query.")
    return True, _off_topic_message()


# ── Response message builders ────────────────────────────────────────────────

_HARD_BLOCK_MESSAGES = {
    "violent_crime": (
        "I'm here to help you build wealth — not to assist with anything that could cause harm. "
        "Let's keep our conversation focused on your financial goals. What would you like to know about personal finance?"
    ),
    "financial_crime": (
        "That falls outside what I can help with. I'm designed to give honest, legal financial guidance. "
        "If you're looking for legitimate tax optimisation or investment strategies, I'm happy to help with those!"
    ),
    "hacking_cybercrime": (
        "I'm not able to assist with anything related to hacking or cybercrime. "
        "Is there a cybersecurity concern about protecting your financial accounts? I can help with that."
    ),
    "drug_manufacturing_dealing": (
        "That's outside my scope entirely. I specialise in personal finance — "
        "budgeting, investing, debt payoff, and building long-term wealth."
    ),
    "weapons": (
        "I can't help with that. My expertise is in personal finance and wealth building. "
        "Feel free to ask me anything about investing, budgeting, or financial planning."
    ),
    "self_harm": (
        "I'm concerned about you. Please reach out to a crisis helpline — "
        "in the US you can call or text **988** (Suicide & Crisis Lifeline) any time. "
        "When you're ready, I'm here to talk about your financial goals."
    ),
    "hate_speech": (
        "I won't engage with that kind of language. Everyone deserves respectful, helpful financial guidance. "
        "What financial question can I help you with?"
    ),
    "sexual_exploitation": (
        "I cannot and will not assist with that under any circumstances."
    ),
    "identity_theft": (
        "I'm not able to help with identity theft or fraud. "
        "If you're worried about protecting your own identity or financial accounts, I can absolutely help with that."
    ),
    "market_manipulation": (
        "Market manipulation is illegal and I won't advise on it. "
        "If you're interested in legitimate trading strategies or understanding market mechanics, I'm here to help."
    ),
    "illegal_gambling": (
        "I can't assist with illegal gambling operations. "
        "If you're curious about the financial risks of gambling or how to manage a gambling problem, I can help."
    ),
    "extortion_blackmail": (
        "That's not something I can help with. I'm built for personal finance guidance — "
        "ask me anything about building legitimate wealth."
    ),
    "unregulated_investment_advice": (
        "\"Guaranteed returns\" and \"risk-free profits\" are hallmarks of financial scams. "
        "Real wealth is built through disciplined saving and diversified investing over time. "
        "Want to learn how to invest safely?"
    ),
    "terrorism_extremism": (
        "I'm not able to assist with that. If you have a personal finance question, I'm here to help."
    ),
    "human_trafficking": (
        "I cannot assist with anything related to human trafficking. "
        "This is completely outside my scope as a personal finance assistant."
    ),
}

_DEFAULT_HARD_BLOCK = (
    "I'm sorry, I'm not able to help with that request. "
    "I'm a personal finance assistant — ask me about investing, budgeting, debt payoff, or wealth building!"
)


def _hard_block_message(label: str) -> str:
    return _HARD_BLOCK_MESSAGES.get(label, _DEFAULT_HARD_BLOCK)


def _off_topic_message() -> str:
    return (
        "I'm a personal finance assistant, so I'm best suited to help with topics like "
        "investing, budgeting, debt payoff, retirement planning, and building long-term wealth. "
        "I'm not really the right tool for that question — but feel free to ask me anything about money! 💰"
    )
