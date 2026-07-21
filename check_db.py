from backend.ingestion.storage import CorpusStorage
s = CorpusStorage()
books = s.get_all_books()
print(f"Books in DB: {len(books)}")
for b in books[:5]:
    print(f"  - {b['title']} | summary_len={len(b.get('summary', ''))} | tags={b['topic_tags'][:3]}")
