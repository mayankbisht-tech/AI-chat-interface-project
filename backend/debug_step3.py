import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingestion.storage import CorpusStorage
from backend.retrieval.multi_query import MultiQueryGenerator
from backend.retrieval.tree_traverser import VectorlessTreeTraverser
from backend.retrieval.reranker import VectorlessReranker
from backend.agents.normal_agent import NormalAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_step3():
    query = "What is the difference between an asset and a liability according to Rich Dad Poor Dad?"
    print("==================================================")
    print(f"STEP 3: TRACING RETRIEVAL FLOW FOR QUERY:\n'{query}'")
    print("==================================================")

    storage = CorpusStorage()
    
    # 3a: Multi-query generation
    mq = MultiQueryGenerator()
    sub_queries = mq.generate_subqueries(query)
    print("\n[3a] Multi-Query Reformulations:")
    for idx, sq in enumerate(sub_queries, 1):
        print(f"  {idx}. {sq}")

    # 3b: Level 0 Book Routing
    traverser = VectorlessTreeTraverser(storage)
    candidate_books = traverser.route_books_level0(sub_queries)
    print("\n[3b] Level 0 Book Routing Output:")
    print("  Selected Books:")
    for b in candidate_books:
        print(f"   - ID: {b['node_id']} | Title: {b['title']}")

    rdpd_selected = any(b['node_id'] == 'book_rich_dad_poor_dad' for b in candidate_books)
    print(f"  --> Was Rich Dad Poor Dad selected at Level 0? {'YES' if rdpd_selected else 'NO'}")

    # 3c: Level 1+ Lazy Traversal for each candidate book
    print("\n[3c] Level 1+ Traversal & Summaries Shown:")
    all_leaf_nodes = []
    for book in candidate_books:
        print(f"\n--- Traversing Book: {book['title']} ({book['node_id']}) ---")
        chapters = storage.get_children(book["node_id"])
        print(f"  Chapters available in DB ({len(chapters)}):")
        for ch in chapters[:5]:
            print(f"    - ID: {ch['node_id']} | Title: '{ch['title']}' | Pages: {ch['start_page']}-{ch['end_page']} | Summary: '{ch['summary'][:80]}...'")
        
        leaves = traverser.traverse_book_tree(book, query)
        print(f"  Selected Leaf Nodes ({len(leaves)}):")
        for leaf in leaves:
            if leaf:
                all_leaf_nodes.append(leaf)
                print(f"    - Leaf Node ID: {leaf['node_id']} | Title: '{leaf['title']}' | Pages: {leaf['start_page']}-{leaf['end_page']}")

    # 3d: Leaf Text Inspection
    print(f"\n[3d] Final Leaf Chunks Text Content ({len(all_leaf_nodes)} total):")
    for idx, leaf in enumerate(all_leaf_nodes, 1):
        raw = leaf.get("raw_text", "")
        safe_raw = raw[:300].encode('ascii', 'ignore').decode('ascii')
        print(f"\n--- Leaf Chunk #{idx}: {leaf.get('title')} (Pages {leaf.get('start_page')}-{leaf.get('end_page')}) ---")
        print(f"Text Snippet:\n{safe_raw}...")
        has_asset = "asset" in raw.lower()
        has_liability = "liability" in raw.lower()
        print(f"Contains 'asset'? {has_asset} | Contains 'liability'? {has_liability}")

    # 3e: Reranking
    reranker = VectorlessReranker()
    top_k = reranker.rerank_and_filter(all_leaf_nodes, query)
    print(f"\n[3e] Pointwise Reranking Results (Top {len(top_k)}):")
    for idx, chunk in enumerate(top_k, 1):
        print(f"  Top #{idx}: {chunk.get('title')} (Pages {chunk.get('start_page')}-{chunk.get('end_page')}) | Score: {chunk.get('relevance_score')}")

    # Step 3f: Test Answer Generation
    agent = NormalAgent(storage)
    ans = agent.run(query, skip_vagueness=True)
    print("\n==================================================")
    print("FINAL AGENT ANSWER GENERATED:")
    print("==================================================")
    print(ans.get("answer"))

if __name__ == "__main__":
    debug_step3()
