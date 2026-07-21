import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

class BookNode:
    def __init__(
        self,
        node_id: str,
        title: str,
        level: str,  # 'book', 'part', 'chapter', 'section', 'subsection'
        start_page: int,
        end_page: int,
        parent_id: Optional[str] = None,
        raw_text: str = "",
        summary: str = ""
    ):
        self.node_id = node_id
        self.title = title
        self.level = level
        self.start_page = start_page
        self.end_page = end_page
        self.parent_id = parent_id
        self.raw_text = raw_text
        self.summary = summary
        self.summary_dict: Dict[str, Any] = {}
        self.children: List['BookNode'] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "title": self.title,
            "level": self.level,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "summary": self.summary,
            "children": [c.to_dict() for c in self.children]
        }

class PDFBookParser:
    """
    Parses a PDF book into a hierarchical tree structure:
    Book -> Parts -> Chapters -> Sections -> Subsections
    """
    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        self.doc = fitz.open(self.pdf_path)
        self.book_title = self._clean_title(self.pdf_path.stem)
        self.total_pages = len(self.doc)

    def _clean_title(self, filename: str) -> str:
        cleaned = re.sub(r'\(.*?\)|\[.*?\]', '', filename)
        cleaned = cleaned.replace('_', ' ').replace('-', ' ').strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned

    def extract_full_text_range(self, start_page: int, end_page: int) -> str:
        text_parts = []
        for pno in range(max(0, start_page - 1), min(self.total_pages, end_page)):
            try:
                page_text = self.doc[pno].get_text()
                if page_text.strip():
                    text_parts.append(page_text.strip())
            except Exception as e:
                print(f"Warning: error reading page {pno+1} in {self.pdf_path.name}: {e}")
        return "\n\n".join(text_parts)

    def parse(self) -> BookNode:
        book_id = f"book_{re.sub(r'[^a-zA-Z0-9]', '_', self.book_title).lower()}"
        root = BookNode(
            node_id=book_id,
            title=self.book_title,
            level="book",
            start_page=1,
            end_page=self.total_pages,
            raw_text=self.extract_full_text_range(1, min(10, self.total_pages))
        )

        toc = self.doc.get_toc()
        if toc and len(toc) >= 3:
            self._build_tree_from_toc(root, toc)
        else:
            self._build_tree_from_headings(root)

        return root

    def _build_tree_from_toc(self, root: BookNode, toc: List[List[Any]]):
        node_stack = [(0, root)]

        for idx, item in enumerate(toc):
            lvl, title, page_num = item[0], item[1].strip(), item[2]
            if page_num <= 0 or not title:
                continue

            level_str = "part" if "part" in title.lower() else "chapter" if lvl == 1 else "section"

            end_page = self.total_pages
            for next_item in toc[idx + 1:]:
                if next_item[2] >= page_num:
                    end_page = next_item[2] - 1
                    break
            if end_page < page_num:
                end_page = page_num

            while node_stack and node_stack[-1][0] >= lvl:
                node_stack.pop()

            parent_node = node_stack[-1][1] if node_stack else root
            node_id = f"{parent_node.node_id}_n{idx+1}"
            raw_text = self.extract_full_text_range(page_num, end_page)

            node = BookNode(
                node_id=node_id,
                title=title,
                level=level_str,
                start_page=page_num,
                end_page=end_page,
                parent_id=parent_node.node_id,
                raw_text=raw_text
            )
            parent_node.children.append(node)
            node_stack.append((lvl, node))

    def _normalize_heading(self, h_text: str) -> str:
        # Normalize titles like "Chapter Two: Lesson 2" -> "Chapter Two: Lesson 2"
        h = re.sub(r'\s+', ' ', h_text).strip()
        h = re.sub(r'^(Chapter\s+\w+):?\s*(Lesson\s+\d+).*', r'\1: \2', h, flags=re.IGNORECASE)
        h = re.sub(r'^(Lesson\s+\d+).*', r'\1', h, flags=re.IGNORECASE)
        return h

    def _build_tree_from_headings(self, root: BookNode):
        headings: List[Dict[str, Any]] = []

        pattern = re.compile(
            r'^(?:CHAPTER|Chapter|LESSON|Lesson|PART|Part|RULE|Rule)\s+(?:[0-9]+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen)\b.*',
            re.IGNORECASE | re.MULTILINE
        )

        for pno in range(self.total_pages):
            text = self.doc[pno].get_text()
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            for l in lines[:5]:
                match = pattern.match(l)
                if match:
                    title_text = self._normalize_heading(match.group(0))
                    if len(title_text) >= 5:
                        headings.append({
                            "page": pno + 1,
                            "title": title_text
                        })

        # Consolidate headings by main chapter title
        consolidated = []
        for h in headings:
            if not consolidated:
                consolidated.append(h)
            else:
                last = consolidated[-1]
                if h["title"] == last["title"]:
                    continue  # Same chapter continuation
                elif h["page"] - last["page"] < 3:
                    continue  # Ignore sub-headers occurring within < 3 pages
                else:
                    consolidated.append(h)

        if len(consolidated) < 3:
            self._build_tree_fallback_synthetic(root)
            return

        for idx, h in enumerate(consolidated):
            start_p = h["page"]
            end_p = consolidated[idx + 1]["page"] - 1 if idx + 1 < len(consolidated) else self.total_pages
            if end_p < start_p:
                end_p = start_p

            raw_text = self.extract_full_text_range(start_p, end_p)
            chap_node = BookNode(
                node_id=f"{root.node_id}_ch{idx+1}",
                title=h["title"],
                level="chapter",
                start_page=start_p,
                end_page=end_p,
                parent_id=root.node_id,
                raw_text=raw_text
            )

            # Divide chapter into 2 sections (Core Principles vs Practical Tactics)
            mid_p = start_p + (end_p - start_p) // 2
            if end_p - start_p >= 4:
                sec1 = BookNode(
                    node_id=f"{chap_node.node_id}_sec1",
                    title=f"{h['title']} - Core Principles & Definitions",
                    level="section",
                    start_page=start_p,
                    end_page=mid_p,
                    parent_id=chap_node.node_id,
                    raw_text=self.extract_full_text_range(start_p, mid_p)
                )
                sec2 = BookNode(
                    node_id=f"{chap_node.node_id}_sec2",
                    title=f"{h['title']} - Practical Application & Examples",
                    level="section",
                    start_page=mid_p + 1,
                    end_page=end_p,
                    parent_id=chap_node.node_id,
                    raw_text=self.extract_full_text_range(mid_p + 1, end_p)
                )
                chap_node.children.extend([sec1, sec2])

            root.children.append(chap_node)

    def _build_tree_fallback_synthetic(self, root: BookNode):
        chunk_size = 20
        total_chunks = (self.total_pages + chunk_size - 1) // chunk_size

        for i in range(total_chunks):
            start_p = i * chunk_size + 1
            end_p = min((i + 1) * chunk_size, self.total_pages)
            raw_text = self.extract_full_text_range(start_p, end_p)

            chap_node = BookNode(
                node_id=f"{root.node_id}_ch{i+1}",
                title=f"Chapter {i+1} (Pages {start_p}-{end_p})",
                level="chapter",
                start_page=start_p,
                end_page=end_p,
                parent_id=root.node_id,
                raw_text=raw_text
            )

            sec1_end = start_p + (end_p - start_p) // 2
            sec1 = BookNode(
                node_id=f"{chap_node.node_id}_sec1",
                title=f"Chapter {i+1} Part A (Pages {start_p}-{sec1_end})",
                level="section",
                start_page=start_p,
                end_page=sec1_end,
                parent_id=chap_node.node_id,
                raw_text=self.extract_full_text_range(start_p, sec1_end)
            )
            sec2 = BookNode(
                node_id=f"{chap_node.node_id}_sec2",
                title=f"Chapter {i+1} Part B (Pages {sec1_end+1}-{end_p})",
                level="section",
                start_page=sec1_end + 1,
                end_page=end_p,
                parent_id=chap_node.node_id,
                raw_text=self.extract_full_text_range(sec1_end + 1, end_p)
            )
            chap_node.children.extend([sec1, sec2])
            root.children.append(chap_node)
