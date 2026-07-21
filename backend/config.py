import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Explicitly load .env file from project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path, override=True)


class Settings(BaseSettings):
    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    PDF_DIR: Path = BASE_DIR / "data" / "pdf_files"
    DB_PATH: Path = BASE_DIR / "data" / "finance_corpus.db"
    INDEX_PATH: Path = BASE_DIR / "data" / "corpus_index.json"

    # ── Default provider ──────────────────────────────────────────────────────
    # Options: gemini | groq | xai | openai
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "gemini")

    # ── Google Gemini (FREE — https://aistudio.google.com/apikey) ─────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

    # ── Groq Cloud (FREE — https://console.groq.com/keys) ─────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", os.getenv("GROK_API_KEY", ""))
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ── xAI Grok-3 (PAID) ─────────────────────────────────────────────────────
    XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
    GROK_MODEL: str = os.getenv("GROK_MODEL", "grok-3")
    XAI_BASE_URL: str = "https://api.x.ai/v1"

    # ── OpenAI (PAID) ─────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ── Web Search ────────────────────────────────────────────────────────────
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    ENABLE_WEB_SEARCH_FALLBACK: bool = True

    # ── RAG Pipeline Config ───────────────────────────────────────────────────
    NUM_SUBQUERIES: int = int(os.getenv("NUM_SUBQUERIES", "2"))
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "6"))
    TREE_EXPANSION_BREADTH: int = int(os.getenv("TREE_EXPANSION_BREADTH", "2"))
    MAX_BOOKS_TO_ROUTE: int = int(os.getenv("MAX_BOOKS_TO_ROUTE", "2"))
    MAX_CHAPTERS_PER_BOOK: int = int(os.getenv("MAX_CHAPTERS_PER_BOOK", "2"))
    MAX_SECTIONS_PER_CHAPTER: int = int(os.getenv("MAX_SECTIONS_PER_CHAPTER", "2"))

    # ── Token Budgets ─────────────────────────────────────────────────────────
    # Reduced budgets for faster LLM calls
    TOKEN_BUDGET_BOOK_ROUTING: int = int(os.getenv("TOKEN_BUDGET_BOOK_ROUTING", "3000"))
    TOKEN_BUDGET_CHAPTER_SELECT: int = int(os.getenv("TOKEN_BUDGET_CHAPTER_SELECT", "2000"))
    TOKEN_BUDGET_SECTION_SELECT: int = int(os.getenv("TOKEN_BUDGET_SECTION_SELECT", "1000"))
    TOKEN_BUDGET_ANSWER_SYNTHESIS: int = int(os.getenv("TOKEN_BUDGET_ANSWER_SYNTHESIS", "8000"))
    CONTEXT_TOKEN_BUDGET_PER_STEP: int = int(os.getenv("CONTEXT_TOKEN_BUDGET_PER_STEP", "2000"))

    # ── Reranking ─────────────────────────────────────────────────────────────
    CROSS_ENCODER_ENABLED: bool = os.getenv("CROSS_ENCODER_ENABLED", "false").lower() == "true"
    DEEP_RESEARCH_CROSS_ENCODER: bool = os.getenv("DEEP_RESEARCH_CROSS_ENCODER", "true").lower() == "true"
    CROSS_ENCODER_MODEL: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "6.0"))

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
