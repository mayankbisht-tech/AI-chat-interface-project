import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import settings
from backend.ingestion.pdf_parser import BookNode

logger = logging.getLogger(__name__)

class CorpusStorage:
    """
    Structured SQLite database storage for hierarchical book trees (no vector DB).
    """

    def __init__(self, db_path: Path = settings.DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    book_id TEXT NOT NULL,
                    parent_id TEXT,
                    level TEXT NOT NULL,
                    title TEXT NOT NULL,
                    start_page INTEGER NOT NULL,
                    end_page INTEGER NOT NULL,
                    summary TEXT,
                    topic_tags TEXT,
                    stance TEXT,
                    audience TEXT,
                    raw_text TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parent ON nodes(parent_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_book ON nodes(book_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON nodes(level);")

    def save_tree(self, root_node: BookNode, book_metadata: Optional[Dict[str, Any]] = None):
        """
        Recursively saves a book tree into SQLite. No silent exception swallowing!
        """
        nodes_to_insert = []
        book_id = root_node.node_id

        def traverse(node: BookNode):
            s_dict = getattr(node, "summary_dict", {}) or {}
            tags_json = json.dumps(s_dict.get("topic_tags", []))
            stance = s_dict.get("stance", "")
            audience = s_dict.get("audience", "")

            # Ensure summary is non-empty
            summary_text = node.summary.strip() if isinstance(node.summary, str) and node.summary.strip() else s_dict.get("summary", "").strip()
            if not summary_text:
                summary_text = f"{node.title} (Pages {node.start_page}-{node.end_page})"

            nodes_to_insert.append((
                node.node_id,
                book_id,
                node.parent_id,
                node.level,
                node.title,
                node.start_page,
                node.end_page,
                summary_text,
                tags_json,
                stance,
                audience,
                node.raw_text
            ))
            for child in node.children:
                traverse(child)

        traverse(root_node)

        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO nodes (
                    node_id, book_id, parent_id, level, title, start_page, end_page,
                    summary, topic_tags, stance, audience, raw_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, nodes_to_insert)

    def get_all_books(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM nodes WHERE level = 'book'").fetchall()
            result = []
            for r in rows:
                item = dict(r)
                item["topic_tags"] = json.loads(item["topic_tags"] or "[]")
                result.append(item)
            return result

    def get_children(self, parent_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT node_id, parent_id, level, title, start_page, end_page, summary, topic_tags, stance, audience FROM nodes WHERE parent_id = ?",
                (parent_id,)
            ).fetchall()
            result = []
            for r in rows:
                item = dict(r)
                item["topic_tags"] = json.loads(item["topic_tags"] or "[]")
                result.append(item)
            return result

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
            if row:
                item = dict(row)
                item["topic_tags"] = json.loads(item["topic_tags"] or "[]")
                return item
            return None

    def export_corpus_index(self, index_path: Path = settings.INDEX_PATH) -> Path:
        books = self.get_all_books()
        index_data = []
        for b in books:
            index_data.append({
                "book_id": b["node_id"],
                "title": b["title"],
                "summary": b["summary"],
                "topic_tags": b["topic_tags"],
                "stance": b["stance"],
                "audience": b["audience"],
                "total_pages": b["end_page"]
            })
        
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
            
        logger.info(f"Exported Corpus Index ({len(index_data)} books) to {index_path}")
        return index_path
