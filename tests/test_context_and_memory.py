"""
Unit tests for Context Bloat Management, Persistent Database, and Async Memory.
"""

import asyncio
import os
import tempfile
import pytest

from memory import (
    ConversationTurn,
    MemoryStore,
    PersistentSessionStore,
    SessionMemory,
    MAX_RECENT_TURNS,
)


def test_sliding_window_eviction():
  mem = SessionMemory("test-sliding-window")
  # Add 10 turns (exceeding MAX_RECENT_TURNS=6)
  for i in range(10):
    mem.add_user_turn(f"Query {i}")
    mem.add_agent_turn(f"Response {i}")

  # Total turns should be 20
  assert len(mem.turns) == 20

  # Sliding window should only return the last MAX_RECENT_TURNS turns
  window = mem.get_sliding_window_turns(window_size=6)
  assert len(window) == 6
  assert window[-1].content == "Response 9"


def test_progressive_summarization():
  mem = SessionMemory("test-summary")
  # Add turns with realistic content
  mem.add_user_turn("Can you give me rates for 5 lb package to 90210?")
  mem.add_agent_turn("Ground starts at $5.99, Express at $12.99.")
  mem.add_user_turn("My tracking number is SW-123456789.")
  mem.add_agent_turn("Your parcel is currently in transit to Chicago.")
  mem.add_user_turn("What is your refund policy?")
  mem.add_agent_turn("Returns accepted within 30 days.")
  mem.add_user_turn("Do you deliver on Saturdays?")
  mem.add_agent_turn("Yes, 8am to 8pm.")

  # Trigger compression
  mem.compress_older_turns()
  assert mem.running_summary != ""
  assert "Customer" in mem.running_summary or "Agent" in mem.running_summary


def test_persistent_sqlite_storage():
  with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
    db_path = tf.name

  try:
    store = PersistentSessionStore(db_path=db_path)
    turn1 = ConversationTurn(role="user", content="Where is SW-987654321?", entities_detected={"tracking_numbers": ["SW-987654321"]})
    turn2 = ConversationTurn(role="agent", content="It is out for delivery.", route="shipping", tools_used=["track_package"])

    store.save_session(
        session_id="persisted-sess-1",
        created_at=100.0,
        last_active=200.0,
        turn_count=2,
        active_tracking_id="SW-987654321",
        running_summary="Customer asked about package SW-987654321",
        entities={"tracking_numbers": ["SW-987654321"]},
    )
    store.save_turn("persisted-sess-1", turn1)
    store.save_turn("persisted-sess-1", turn2)

    # Re-open database from scratch
    reopened = PersistentSessionStore(db_path=db_path)
    loaded = reopened.load_session("persisted-sess-1")

    assert loaded is not None
    assert loaded["session_id"] == "persisted-sess-1"
    assert loaded["active_tracking_id"] == "SW-987654321"
    assert len(loaded["turns"]) == 2
    assert loaded["turns"][0].content == "Where is SW-987654321?"
    assert loaded["turns"][1].tools_used == ["track_package"]

  finally:
    if os.path.exists(db_path):
      os.remove(db_path)


@pytest.mark.asyncio
async def test_async_background_save():
  sess_id = "async-bg-test-1"
  mem = MemoryStore.get_or_create(sess_id)
  mem.add_user_turn("Track SW-112233445")
  turn = mem.turns[-1]

  # Run async background task
  await MemoryStore.save_turn_async(sess_id, turn)

  # Check that session can be reloaded
  loaded = MemoryStore.get(sess_id)
  assert loaded is not None
  assert loaded.active_tracking_number == "SW-112233445"
