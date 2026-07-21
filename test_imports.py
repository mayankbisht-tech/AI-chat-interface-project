import sys
print("Testing all backend module imports...")

try:
    from backend.config import settings
    print(f"  [OK] config — model={settings.GROK_MODEL}, provider={settings.DEFAULT_PROVIDER}")
except Exception as e:
    print(f"  [FAIL] config: {e}")

try:
    from backend.ingestion.storage import CorpusStorage
    s = CorpusStorage()
    books = s.get_all_books()
    print(f"  [OK] storage — {len(books)} books in DB")
except Exception as e:
    print(f"  [FAIL] storage: {e}")

try:
    from backend.ingestion.summarizer import NodeSummarizer
    print("  [OK] summarizer")
except Exception as e:
    print(f"  [FAIL] summarizer: {e}")

try:
    from backend.retrieval.vagueness_detector import VaguenessDetector
    print("  [OK] vagueness_detector")
except Exception as e:
    print(f"  [FAIL] vagueness_detector: {e}")

try:
    from backend.retrieval.multi_query import MultiQueryGenerator
    print("  [OK] multi_query")
except Exception as e:
    print(f"  [FAIL] multi_query: {e}")

try:
    from backend.retrieval.tree_traverser import VectorlessTreeTraverser
    print("  [OK] tree_traverser")
except Exception as e:
    print(f"  [FAIL] tree_traverser: {e}")

try:
    from backend.retrieval.reranker import VectorlessReranker, CrossEncoderReranker, get_reranker
    r_llm = get_reranker(use_cross_encoder=False)
    r_ce = get_reranker(use_cross_encoder=True)
    print(f"  [OK] reranker — LLM={type(r_llm).__name__}, CE={type(r_ce).__name__}")
except Exception as e:
    print(f"  [FAIL] reranker: {e}")

try:
    from backend.retrieval.relevance_gate import HardRelevanceGate
    print("  [OK] relevance_gate")
except Exception as e:
    print(f"  [FAIL] relevance_gate: {e}")

try:
    from backend.tools.tavily_search import TavilySearchTool
    t = TavilySearchTool()
    print(f"  [OK] tavily_search — client_ready={t.client is not None}")
except Exception as e:
    print(f"  [FAIL] tavily_search: {e}")

try:
    from backend.agents.normal_agent import NormalAgent
    print("  [OK] normal_agent")
except Exception as e:
    print(f"  [FAIL] normal_agent: {e}")

try:
    from backend.agents.deep_research_agent import DeepResearchAgent
    print("  [OK] deep_research_agent")
except Exception as e:
    print(f"  [FAIL] deep_research_agent: {e}")

try:
    from backend.main import app, get_llm_client
    llm = get_llm_client()
    print(f"  [OK] main — FastAPI app loaded, LLM client={type(llm).__name__ if llm else 'None (API key not set)'}")
except Exception as e:
    print(f"  [FAIL] main: {e}")

print("\nAll import tests complete.")
