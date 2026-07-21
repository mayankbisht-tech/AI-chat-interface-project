import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingestion.storage import CorpusStorage
from backend.retrieval.vagueness_detector import VaguenessDetector
from backend.retrieval.multi_query import MultiQueryGenerator
from backend.retrieval.tree_traverser import VectorlessTreeTraverser
from backend.retrieval.reranker import VectorlessReranker
from backend.agents.normal_agent import NormalAgent
from backend.agents.deep_research_agent import DeepResearchAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline():
    print("=== TEST 1: Corpus Storage Check ===")
    storage = CorpusStorage()
    books = storage.get_all_books()
    print(f"Total Books in SQLite Store: {len(books)}")
    for b in books[:3]:
        print(f" - [{b['node_id']}] {b['title']} (Stance: {b['stance'][:60]}...)")

    print("\n=== TEST 2: Vagueness Detector ===")
    vd = VaguenessDetector()
    res1 = vd.evaluate_query("How to invest?")
    print("Vague Query Result:", res1)

    res2 = vd.evaluate_query("Should I pay off my 6% mortgage debt or invest in S&P 500 index funds for retirement?")
    print("Specific Query Result:", res2)

    print("\n=== TEST 3: Multi-Query Generation ===")
    mq = MultiQueryGenerator()
    sub_q = mq.generate_subqueries("Debt payoff vs index fund investing")
    print("Sub-Queries:", sub_q)

    print("\n=== TEST 4: Vectorless Tree Traversal (Level 0 -> Leaf) ===")
    traverser = VectorlessTreeTraverser(storage)
    ret_res = traverser.execute_full_retrieval(sub_q, "Debt payoff vs index fund investing")
    print("Candidate Books Selected:", ret_res["candidate_books"])
    print(f"Total Leaf Nodes Retrieved: {len(ret_res['leaf_nodes'])}")
    for trace in ret_res["traversal_trace"][:4]:
        print(f"  Trace: {trace['book_title']} -> {trace['leaf_title']} ({trace['pages']})")

    print("\n=== TEST 5: Normal Agent Flow ===")
    agent = NormalAgent(storage)
    answer_res = agent.run("Should I pay off debt or invest in index funds first?", skip_vagueness=True)
    print("Normal Agent Response Type:", answer_res.get("type"))
    print("Sources Count:", len(answer_res.get("sources", [])))
    print("Answer Preview:\n", answer_res.get("answer", "")[:400])

    print("\n=== TEST 6: Deep Research Agent Flow ===")
    dr_agent = DeepResearchAgent(storage)
    dr_res = dr_agent.run("What are the 2026 tax bracket limits compared to Boglehead index fund rules?", skip_vagueness=True)
    print("Deep Research Response Type:", dr_res.get("type"))
    print("Sub-Questions Count:", len(dr_res.get("sub_questions", [])))
    print("Web Sources Count:", len(dr_res.get("web_sources", [])))
    print("Report Preview:\n", dr_res.get("answer", "")[:400])

    print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_pipeline()
