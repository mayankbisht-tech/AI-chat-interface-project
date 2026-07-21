import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from backend.config import BASE_DIR, settings
from backend.logging_config import setup_logger

logger = setup_logger("TavilySearchTool")

class TavilySearchTool:
    """
    Tavily Web Search Tool for real-time rates, tax brackets, and recent market updates.
    Dynamically re-checks .env on every search to guarantee newly pasted API keys work immediately.
    """

    def __init__(self, api_key: str = None):
        self.manual_api_key = api_key
        self.client = None
        self.current_key = None

    def _get_client(self):
        # Reload .env dynamically in case user updated API keys in the file
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)

        key = (
            self.manual_api_key
            or os.getenv("TAVILY_API_KEY", "")
            or getattr(settings, "TAVILY_API_KEY", "")
        )

        if not key or key.startswith("your_"):
            logger.warning("[Tavily] TAVILY_API_KEY missing or placeholder in .env.")
            return None

        # Re-initialize client if key changed or client not created yet
        if self.client is None or self.current_key != key:
            try:
                from tavily import TavilyClient
                self.client = TavilyClient(api_key=key)
                self.current_key = key
                logger.info(f"[Tavily] TavilyClient successfully initialized with API key: {key[:6]}...")
            except Exception as e:
                logger.error(f"[Tavily] Failed to initialize TavilyClient: {e}")
                self.client = None

        return self.client

    def search(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """
        Executes web search query using Tavily API.
        """
        client = self._get_client()
        if not client:
            logger.warning(f"[Tavily] Search skipped for '{query}': TAVILY_API_KEY missing or invalid in .env.")
            return []

        try:
            logger.info(f"[Tavily] Executing live Tavily web search API call for: '{query}'")
            response = client.search(query=query, max_results=max_results, search_depth="basic")
            results = []
            for res in response.get("results", []):
                results.append({
                    "title": res.get("title", "Web Source"),
                    "url": res.get("url", ""),
                    "snippet": res.get("content", ""),
                    "source_type": "web"
                })
            logger.info(f"[Tavily] Tavily API returned {len(results)} web results for: '{query}'")
            return results
        except Exception as e:
            logger.error(f"[Tavily] Tavily API call FAILED for query '{query}': {e}", exc_info=True)
            return []
