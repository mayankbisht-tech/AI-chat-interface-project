import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import get_llm_client
from backend.ingestion.storage import CorpusStorage
from backend.agents.normal_agent import NormalAgent

async def test_stream():
    llm = get_llm_client()
    storage = CorpusStorage()
    agent = NormalAgent(storage, llm_client=llm)

    query = "What is the 10 percent rule in The Richest Man in Babylon?"
    print(f"Testing streaming for query: '{query}'")
    print("=" * 50)

    async for event in agent.run_stream(query, skip_vagueness=True):
        evt_type = event.get("event")
        if evt_type == "status":
            print(f"[STATUS] {event.get('message')}")
        elif evt_type == "traversal":
            print(f"[TRAVERSAL] Selected Books: {event.get('candidate_books')}")
        elif evt_type == "answer_chunk":
            print(event.get("chunk"), end="", flush=True)

    print("\n" + "=" * 50)
    print("STREAMING TEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(test_stream())
