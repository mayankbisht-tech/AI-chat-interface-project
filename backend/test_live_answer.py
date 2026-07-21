import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import get_llm_client
from backend.ingestion.storage import CorpusStorage
from backend.retrieval.multi_query import MultiQueryGenerator
from backend.retrieval.tree_traverser import VectorlessTreeTraverser
from backend.retrieval.reranker import VectorlessReranker
from backend.agents.normal_agent import NormalAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_live_answer():
    llm = get_llm_client()
    storage = CorpusStorage()
    agent = NormalAgent(storage, llm_client=llm)

    query = "What is the difference between an asset and a liability according to Rich Dad Poor Dad?"
    print(f"Executing Query with live LLM: '{query}'")
    res = agent.run(query, skip_vagueness=True)

    print("\n==================================================")
    print("LIVE LLM ANSWER:")
    print("==================================================")
    print(res.get("answer"))

if __name__ == "__main__":
    test_live_answer()
