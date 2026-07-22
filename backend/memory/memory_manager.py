"""
Memory Manager — Short-term + Long-term memory backed by Neon PostgreSQL.

Short-term memory:  Last MAX_SHORT_TERM_TURNS (default 10) messages for the current
                    session. Stored in the `conversation_turns` table. Cleared when the
                    session ends (browser close / new session_id).

Long-term memory:   A rolling LLM-generated summary of the user's financial profile
                    (goals, risk tolerance, income situation, debt status, etc.) stored
                    in the `user_profiles` table.  Updated after every N turns so the
                    next session can pick up context without re-reading every message.

Usage:
    mm = MemoryManager()
    await mm.init()                               # create tables if not exist
    history = await mm.get_short_term(session_id) # list of {role, content}
    profile = await mm.get_long_term(session_id)  # str or ""
    await mm.add_turn(session_id, "user", text)
    await mm.add_turn(session_id, "assistant", text)
    await mm.maybe_update_long_term(session_id, llm_client)
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger("MemoryManager")

# How many recent turns to keep in short-term memory
MAX_SHORT_TERM_TURNS = 10
# Update long-term summary every N turns
LONG_TERM_UPDATE_EVERY = 6


class MemoryManager:
    """
    Async Neon PostgreSQL memory manager.
    Falls back to in-process dict if NEON_DATABASE_URL is not set (dev / no-DB mode).
    """

    def __init__(self):
        self.db_url = os.getenv("NEON_DATABASE_URL", "").strip()
        self._pool = None
        self._fallback: Dict[str, List[Dict]] = {}      # session_id -> turns list
        self._fallback_profile: Dict[str, str] = {}     # session_id -> profile str
        self._use_fallback = not bool(self.db_url)
        if self._use_fallback:
            logger.warning(
                "[Memory] NEON_DATABASE_URL not set — using in-process dict (no persistence)."
            )

    # ── Initialisation ──────────────────────────────────────────────────────

    async def init(self):
        """Create tables and connection pool. Safe to call multiple times."""
        if self._use_fallback:
            return
        if self._pool is not None:
            return
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                dsn=self.db_url,
                min_size=1,
                max_size=5,
                ssl="require",
            )
            await self._create_tables()
            logger.info("[Memory] Connected to Neon PostgreSQL and tables verified.")
        except Exception as e:
            logger.error(f"[Memory] Neon DB init failed — falling back to in-process dict: {e}")
            self._use_fallback = True
            self._pool = None

    async def _create_tables(self):
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id          BIGSERIAL PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,          -- 'user' | 'assistant'
                    content     TEXT NOT NULL,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_turns_session
                    ON conversation_turns(session_id, created_at DESC);
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    session_id   TEXT PRIMARY KEY,
                    profile_text TEXT NOT NULL DEFAULT '',
                    turn_count   INT  NOT NULL DEFAULT 0,
                    updated_at   TIMESTAMPTZ DEFAULT NOW()
                );
            """)

    # ── Short-term memory ───────────────────────────────────────────────────

    async def get_short_term(self, session_id: str) -> List[Dict[str, str]]:
        """Return last MAX_SHORT_TERM_TURNS turns as [{role, content}, ...]."""
        if self._use_fallback:
            return list(self._fallback.get(session_id, []))[-MAX_SHORT_TERM_TURNS:]

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT role, content FROM (
                        SELECT role, content, created_at
                        FROM conversation_turns
                        WHERE session_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                    ) sub
                    ORDER BY created_at ASC
                    """,
                    session_id,
                    MAX_SHORT_TERM_TURNS,
                )
                return [{"role": r["role"], "content": r["content"]} for r in rows]
        except Exception as e:
            logger.error(f"[Memory] get_short_term failed: {e}")
            return []

    async def add_turn(self, session_id: str, role: str, content: str):
        """Persist a single conversation turn."""
        if self._use_fallback:
            turns = self._fallback.setdefault(session_id, [])
            turns.append({"role": role, "content": content})
            # Keep only last MAX_SHORT_TERM_TURNS * 2 to bound memory
            if len(turns) > MAX_SHORT_TERM_TURNS * 2:
                self._fallback[session_id] = turns[-MAX_SHORT_TERM_TURNS * 2:]
            return

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO conversation_turns(session_id, role, content) VALUES($1,$2,$3)",
                    session_id, role, content,
                )
        except Exception as e:
            logger.error(f"[Memory] add_turn failed: {e}")

    # ── Long-term memory ────────────────────────────────────────────────────

    async def get_long_term(self, session_id: str) -> str:
        """Return stored long-term profile string (empty if none)."""
        if self._use_fallback:
            return self._fallback_profile.get(session_id, "")

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT profile_text FROM user_profiles WHERE session_id = $1",
                    session_id,
                )
                return row["profile_text"] if row else ""
        except Exception as e:
            logger.error(f"[Memory] get_long_term failed: {e}")
            return ""

    async def _get_turn_count(self, session_id: str) -> int:
        if self._use_fallback:
            return len(self._fallback.get(session_id, []))
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT turn_count FROM user_profiles WHERE session_id = $1",
                    session_id,
                )
                return row["turn_count"] if row else 0
        except Exception as e:
            logger.error(f"[Memory] _get_turn_count failed: {e}")
            return 0

    async def maybe_update_long_term(self, session_id: str, llm_client=None):
        """
        After every LONG_TERM_UPDATE_EVERY turns, regenerate the long-term profile
        summary from the last MAX_SHORT_TERM_TURNS conversation turns.
        """
        if not llm_client:
            return

        turns = await self.get_short_term(session_id)
        total_turns = len(turns)

        if total_turns == 0 or total_turns % LONG_TERM_UPDATE_EVERY != 0:
            return

        # Build conversation text for the LLM to summarise
        conv_text = "\n".join(
            f"{t['role'].upper()}: {t['content'][:400]}" for t in turns
        )

        existing_profile = await self.get_long_term(session_id)
        prompt = f"""You are a financial profile extractor. 
Given the conversation below, extract and update the user's financial profile.
Include: financial goals, time horizon, risk tolerance, current situation (debt/savings/income level if mentioned), 
topics of interest, and any specific preferences or constraints mentioned.
Keep the profile concise (max 150 words). If no financial info was discussed, write "No profile data yet."

Existing profile:
{existing_profile or "None"}

Recent conversation:
{conv_text}

Updated profile (plain text, no JSON):"""

        try:
            import asyncio
            if hasattr(llm_client, "ainvoke"):
                res = await llm_client.ainvoke(prompt)
                profile_text = res.content if hasattr(res, "content") else str(res)
            else:
                res = await asyncio.to_thread(llm_client.invoke, prompt)
                profile_text = res.content if hasattr(res, "content") else str(res)

            profile_text = profile_text.strip()[:800]  # hard cap
            await self._save_long_term(session_id, profile_text, total_turns)
            logger.info(f"[Memory] Long-term profile updated for session {session_id[:8]}...")
        except Exception as e:
            logger.error(f"[Memory] maybe_update_long_term LLM call failed: {e}")

    async def _save_long_term(self, session_id: str, profile_text: str, turn_count: int):
        if self._use_fallback:
            self._fallback_profile[session_id] = profile_text
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO user_profiles(session_id, profile_text, turn_count, updated_at)
                    VALUES($1, $2, $3, NOW())
                    ON CONFLICT(session_id) DO UPDATE
                        SET profile_text = EXCLUDED.profile_text,
                            turn_count   = EXCLUDED.turn_count,
                            updated_at   = NOW()
                    """,
                    session_id, profile_text, turn_count,
                )
        except Exception as e:
            logger.error(f"[Memory] _save_long_term failed: {e}")

    # ── Cleanup ─────────────────────────────────────────────────────────────

    async def clear_session(self, session_id: str):
        """Delete all short-term turns for a session (called on explicit reset)."""
        if self._use_fallback:
            self._fallback.pop(session_id, None)
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM conversation_turns WHERE session_id = $1",
                    session_id,
                )
            logger.info(f"[Memory] Session {session_id[:8]}... cleared.")
        except Exception as e:
            logger.error(f"[Memory] clear_session failed: {e}")

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None
