import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import settings
from backend.ingestion.pdf_parser import PDFBookParser
from backend.ingestion.summarizer import NodeSummarizer
from backend.ingestion.storage import CorpusStorage
from backend.ingestion.ingest_all import BOOK_METADATA_PRESETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fast_ingest():
    pdf_dir = settings.PDF_DIR
    storage = CorpusStorage()
    summarizer = NodeSummarizer()

    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Starting ingestion for {len(pdf_files)} PDF books...")

    for idx, pdf_path in enumerate(pdf_files, 1):
        filename_stem = pdf_path.stem
        preset = None
        for key, val in BOOK_METADATA_PRESETS.items():
            if key.lower() in filename_stem.lower() or filename_stem.lower() in key.lower():
                preset = val
                break

        print(f"[{idx}/{len(pdf_files)}] Parsing: {pdf_path.name}")
        try:
            parser = PDFBookParser(pdf_path)
            root_node = parser.parse()

            if preset:
                root_node.title = preset["title"]
                root_node.summary_dict = {
                    "summary": f"{preset['title']} by {preset['author']}. Stance: {preset['stance']} Audience: {preset['audience']}",
                    "topic_tags": preset["topic_tags"],
                    "stance": preset["stance"],
                    "audience": preset["audience"]
                }
            else:
                root_node.summary_dict = summarizer.summarize_node(root_node, root_node.title)

            # Assign chapter summaries
            for child in root_node.children:
                child.summary_dict = {
                    "summary": f"Chapter '{child.title}' in {root_node.title} (Pages {child.start_page}-{child.end_page}): Principles and actionable financial guidance.",
                    "topic_tags": root_node.summary_dict.get("topic_tags", []),
                    "stance": root_node.summary_dict.get("stance", "")
                }
                child.summary = child.summary_dict["summary"]

                for sec in child.children:
                    sec.summary_dict = {
                        "summary": f"Section '{sec.title}' (Pages {sec.start_page}-{sec.end_page}): Core concepts and step-by-step tactics.",
                        "topic_tags": root_node.summary_dict.get("topic_tags", []),
                        "stance": root_node.summary_dict.get("stance", "")
                    }
                    sec.summary = sec.summary_dict["summary"]

            storage.save_tree(root_node)
        except Exception as e:
            print(f"Error parsing {pdf_path.name}: {e}")

    index_path = storage.export_corpus_index()
    print(f"DONE! Ingestion complete. Index saved to {index_path}")

if __name__ == "__main__":
    fast_ingest()
