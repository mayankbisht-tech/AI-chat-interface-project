import sys
import unittest
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.ingestion.storage import CorpusStorage
from backend.retrieval.multi_query import MultiQueryGenerator
from backend.retrieval.tree_traverser import VectorlessTreeTraverser
from backend.retrieval.reranker import VectorlessReranker
from backend.agents.normal_agent import NormalAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestVectorlessRAGRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.storage = CorpusStorage()
        cls.traverser = VectorlessTreeTraverser(cls.storage)
        cls.reranker = VectorlessReranker()
        cls.agent = NormalAgent(cls.storage)

    def test_rich_dad_poor_dad_asset_vs_liability(self):
        query = "What is the difference between an asset and a liability according to Rich Dad Poor Dad?"
        res = self.agent.run(query, skip_vagueness=True)

        self.assertEqual(res.get("type"), "answer")
        self.assertIn("Rich Dad Poor Dad", res.get("candidate_books", []))

        # Assert top leaf node points to Chapter Two: Lesson 2
        top_source_titles = [s.get("title") for s in res.get("sources", [])]
        self.assertTrue(any("Lesson 2" in t for t in top_source_titles if t))

        # Assert answer contains core definitions
        answer = res.get("answer", "").lower()
        self.assertTrue("asset" in answer and "liability" in answer)

    def test_richest_man_in_babylon_ten_percent_rule(self):
        query = "What is the 10 percent gold saving rule in The Richest Man in Babylon?"
        res = self.agent.run(query, skip_vagueness=True)
        self.assertIn("The Richest Man in Babylon", res.get("candidate_books", []))
        self.assertTrue(len(res.get("sources", [])) > 0)

    def test_simple_path_to_wealth_vtsax(self):
        query = "Why does JL Collins recommend total stock market index funds in The Simple Path to Wealth?"
        res = self.agent.run(query, skip_vagueness=True)
        self.assertIn("The Simple Path to Wealth", res.get("candidate_books", []))
        self.assertTrue(len(res.get("sources", [])) > 0)

    def test_total_money_makeover_debt_snowball(self):
        query = "What is the Debt Snowball method in The Total Money Makeover?"
        res = self.agent.run(query, skip_vagueness=True)
        self.assertIn("The Total Money Makeover", res.get("candidate_books", []))
        self.assertTrue(len(res.get("sources", [])) > 0)

if __name__ == "__main__":
    unittest.main()
