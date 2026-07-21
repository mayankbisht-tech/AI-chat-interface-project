import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.tools.tavily_search import TavilySearchTool

def test_tavily_isolated():
    print("==================================================")
    print("PART A - STEP 5: ISOLATED TAVILY API TEST")
    print("==================================================")
    key = settings.TAVILY_API_KEY or os.getenv("TAVILY_API_KEY", "")
    print(f"Loaded Tavily API Key: '{key[:6]}...' (length: {len(key)})")

    tool = TavilySearchTool(api_key=key)
    query = "current 401k contribution limit 2026"
    print(f"Executing search query: '{query}'")

    results = tool.search(query, max_results=3)
    print(f"Total Results Returned: {len(results)}")
    for idx, r in enumerate(results, 1):
        print(f"\nResult #{idx}:")
        print(f"  Title: {r.get('title')}")
        print(f"  URL: {r.get('url')}")
        print(f"  Snippet: {r.get('snippet')[:200]}...")

if __name__ == "__main__":
    test_tavily_isolated()
