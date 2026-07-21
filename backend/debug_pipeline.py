import sys
import logging
import fitz
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingestion.storage import CorpusStorage
from backend.ingestion.pdf_parser import PDFBookParser
from backend.retrieval.vagueness_detector import VaguenessDetector
from backend.retrieval.multi_query import MultiQueryGenerator
from backend.retrieval.tree_traverser import VectorlessTreeTraverser
from backend.retrieval.reranker import VectorlessReranker
from backend.agents.normal_agent import NormalAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def debug_step1_and_step2():
    print("==================================================")
    print("STEP 1: VERIFYING INGESTION FOR RICH DAD POOR DAD")
    print("==================================================")
    storage = CorpusStorage()
    pdf_path = Path("data/pdf_files/Rich Dad Poor Dad.pdf")

    if not pdf_path.exists():
        print(f"ERROR: PDF file not found at {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    pdf_page_count = len(doc)
    print(f"PDF File: {pdf_path.name}")
    print(f"Actual PDF Page Count: {pdf_page_count}")

    # Check SQLite DB for nodes matching Rich Dad Poor Dad
    books = storage.get_all_books()
    rdpd_book = None
    for b in books:
        if "rich dad" in b["title"].lower() or "rich dad" in b["node_id"].lower():
            rdpd_book = b
            break

    if not rdpd_book:
        print("CRITICAL FAILURE: 'Rich Dad Poor Dad' is NOT present in SQLite Level 0 books table!")
        print("Available books in DB:")
        for b in books:
            print(f" - ID: '{b['node_id']}', Title: '{b['title']}'")
        return

    book_id = rdpd_book["node_id"]
    print(f"\nFound Rich Dad Poor Dad in SQLite DB:")
    print(f"  node_id: {book_id}")
    print(f"  title: {rdpd_book['title']}")
    print(f"  start_page: {rdpd_book['start_page']}, end_page: {rdpd_book['end_page']}")

    # Get all nodes under this book
    with storage._get_connection() as conn:
        all_nodes = conn.execute("SELECT * FROM nodes WHERE book_id = ?", (book_id,)).fetchall()
        total_node_count = len(all_nodes)
        total_text_len = sum(len(r["raw_text"] or "") for r in all_nodes)

    print(f"  Total Node Count for book: {total_node_count}")
    print(f"  Total Ingested Raw Text Length: {total_text_len} chars")

    print("\nBook Summary:")
    print(f"  Summary: {rdpd_book['summary']}")
    print(f"  Stance: {rdpd_book['stance']}")
    print(f"  Topic Tags: {rdpd_book['topic_tags']}")

    # Fetch top-level chapter nodes
    children = storage.get_children(book_id)
    print(f"\nTop-Level Chapter Count: {len(children)}")
    for idx, child in enumerate(children, 1):
        print(f"  [{idx}] Node ID: {child['node_id']} | Title: {child['title']} | Pages: {child['start_page']}-{child['end_page']}")
        print(f"      Summary: {child['summary'][:150]}...")

    print("\n==================================================")
    print("STEP 2: VERIFYING CORPUS INDEX (LEVEL 0)")
    print("==================================================")
    index_path = Path("data/corpus_index.json")
    if not index_path.exists():
        print("ERROR: corpus_index.json does not exist!")
        return

    import json
    with open(index_path, "r", encoding="utf-8") as f:
        corpus_index = json.load(f)

    rdpd_index_entry = None
    for entry in corpus_index:
        if "rich dad" in entry.get("title", "").lower() or "rich dad" in entry.get("book_id", "").lower():
            rdpd_index_entry = entry
            break

    if not rdpd_index_entry:
        print("CRITICAL FAILURE: Rich Dad Poor Dad is MISSING from corpus_index.json!")
    else:
        print("Corpus Index Entry for Rich Dad Poor Dad:")
        print(json.dumps(rdpd_index_entry, indent=2))

if __name__ == "__main__":
    debug_step1_and_step2()
