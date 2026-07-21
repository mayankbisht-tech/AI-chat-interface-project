import sys
import unittest
import asyncio
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.ingestion.storage import CorpusStorage
from backend.agents.normal_agent import NormalAgent
from backend.agents.deep_research_agent import DeepResearchAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPartABC(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.storage = CorpusStorage()
        self.agent = NormalAgent(self.storage)
        self.deep_agent = DeepResearchAgent(self.storage)

    async def test_part_b_in_corpus_query(self):
        query = "What is the difference between an asset and a liability according to Rich Dad Poor Dad?"
        events = []
        async for evt in self.agent.run_stream(query, skip_vagueness=True):
            events.append(evt)

        event_types = [e.get("event") for e in events]
        self.assertIn("traversal", event_types)
        self.assertIn("sources", event_types)
        self.assertIn("answer_chunk", event_types)

        # Verify Rich Dad Poor Dad was selected
        traversal_evt = next(e for e in events if e.get("event") == "traversal")
        self.assertIn("Rich Dad Poor Dad", traversal_evt.get("candidate_books", []))

    async def test_part_a_web_search_triggering(self):
        query = "What is the current Fed interest rate in 2026?"
        events = []
        async for evt in self.agent.run_stream(query, skip_vagueness=True):
            events.append(evt)

        statuses = [e.get("message", "") for e in events if e.get("event") == "status"]
        self.assertTrue(any("Step 2" in s or "Tavily" in s or "internet search" in s for s in statuses))

    async def test_part_b_out_of_scope_honest_reporting(self):
        query = "How to repair a Toyota Camry engine?"
        events = []
        async for evt in self.agent.run_stream(query, skip_vagueness=True):
            events.append(evt)

        statuses = [e.get("message", "") for e in events if e.get("event") == "status"]
        self.assertTrue(any("Step 2" in s or "Tavily" in s or "internet search" in s for s in statuses))

if __name__ == "__main__":
    unittest.main()
