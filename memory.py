"""
Context & Memory Engine for SwiftShip Customer Support Agent.

Implements:
  1. Context Bloat Management: Sliding window truncation and running background summarization.
  2. Persistent Storage: ACID-compliant SQLite session database (`data/sessions.db`) persisting state across server restarts.
  3. Asynchronous Background Execution: Non-blocking background tasks for persistence, entity extraction, and memory summarization.
"""

import asyncio
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
  from customer_support_agent.observability import PIIRedactor, get_logger, get_tracer
  logger = get_logger("swiftship.memory")
  tracer = get_tracer("swiftship.memory")
except ImportError:
  try:
    from observability import PIIRedactor, get_logger, get_tracer
    logger = get_logger("swiftship.memory")
    tracer = get_tracer("swiftship.memory")
  except ImportError:
    logger = None
    tracer = None

# Context Window Configuration
MAX_RECENT_TURNS = int(os.getenv("MEMORY_MAX_RECENT_TURNS", "6"))  # 3 exchanges
MAX_SUMMARY_LENGTH = int(os.getenv("MEMORY_MAX_SUMMARY_LENGTH", "500"))
DEFAULT_DB_PATH = os.getenv("SESSION_DB_PATH", str(Path(__file__).resolve().parent / "data" / "sessions.db"))


@dataclass
class ConversationTurn:
  role: str  # "user" or "agent"
  content: str
  timestamp: float = field(default_factory=time.time)
  route: Optional[str] = None
  tools_used: List[str] = field(default_factory=list)
  entities_detected: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 1. Persistent Storage Backend (SQLite Engine)
# ==============================================================================

class PersistentSessionStore:
  """ACID-compliant persistent SQLite database for session state and conversation history."""

  def __init__(self, db_path: str = DEFAULT_DB_PATH):
    self.db_path = db_path
    db_dir = os.path.dirname(self.db_path)
    if db_dir:
      os.makedirs(db_dir, exist_ok=True)
    self._init_db()

  def _get_connection(self) -> sqlite3.Connection:
    conn = sqlite3.connect(self.db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

  def _init_db(self) -> None:
    """Initializes tables with WAL mode for fast concurrent async writes."""
    with self._get_connection() as conn:
      conn.execute("PRAGMA journal_mode=WAL;")
      conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
          session_id TEXT PRIMARY KEY,
          created_at REAL,
          last_active REAL,
          turn_count INTEGER,
          active_tracking_id TEXT,
          running_summary TEXT,
          entities_json TEXT
        );
      """)
      conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT,
          role TEXT,
          content TEXT,
          timestamp REAL,
          route TEXT,
          tools_json TEXT,
          entities_json TEXT,
          FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
      """)
      conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);")
      conn.commit()

  def save_session(
      self,
      session_id: str,
      created_at: float,
      last_active: float,
      turn_count: int,
      active_tracking_id: Optional[str],
      running_summary: str,
      entities: Dict[str, Any],
  ) -> None:
    with self._get_connection() as conn:
      conn.execute(
          """
          INSERT INTO sessions (session_id, created_at, last_active, turn_count, active_tracking_id, running_summary, entities_json)
          VALUES (?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(session_id) DO UPDATE SET
            last_active=excluded.last_active,
            turn_count=excluded.turn_count,
            active_tracking_id=excluded.active_tracking_id,
            running_summary=excluded.running_summary,
            entities_json=excluded.entities_json;
          """,
          (
              session_id,
              created_at,
              last_active,
              turn_count,
              active_tracking_id,
              running_summary,
              json.dumps(entities),
          ),
      )
      conn.commit()

  def save_turn(self, session_id: str, turn: ConversationTurn) -> None:
    with self._get_connection() as conn:
      conn.execute(
          """
          INSERT INTO turns (session_id, role, content, timestamp, route, tools_json, entities_json)
          VALUES (?, ?, ?, ?, ?, ?, ?);
          """,
          (
              session_id,
              turn.role,
              turn.content,
              turn.timestamp,
              turn.route,
              json.dumps(turn.tools_used),
              json.dumps(turn.entities_detected),
          ),
      )
      conn.commit()

  def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
    with self._get_connection() as conn:
      cursor = conn.execute("SELECT * FROM sessions WHERE session_id = ?;", (session_id,))
      row = cursor.fetchone()
      if not row:
        return None

      # Load associated turns
      t_cursor = conn.execute(
          "SELECT * FROM turns WHERE session_id = ? ORDER BY id ASC;",
          (session_id,),
      )
      turns = []
      for t in t_cursor.fetchall():
        turns.append(
            ConversationTurn(
                role=t["role"],
                content=t["content"],
                timestamp=t["timestamp"],
                route=t["route"],
                tools_used=json.loads(t["tools_json"] or "[]"),
                entities_detected=json.loads(t["entities_json"] or "{}"),
            )
        )

      return {
          "session_id": row["session_id"],
          "created_at": row["created_at"],
          "last_active": row["last_active"],
          "turn_count": row["turn_count"],
          "active_tracking_id": row["active_tracking_id"],
          "running_summary": row["running_summary"] or "",
          "entities": json.loads(row["entities_json"] or "{}"),
          "turns": turns,
      }

  def delete_session(self, session_id: str) -> None:
    with self._get_connection() as conn:
      conn.execute("DELETE FROM turns WHERE session_id = ?;", (session_id,))
      conn.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
      conn.commit()

  def count_sessions(self) -> int:
    with self._get_connection() as conn:
      cursor = conn.execute("SELECT COUNT(*) FROM sessions;")
      return cursor.fetchone()[0]


# Initialize persistent database singleton
_storage = PersistentSessionStore()


# ==============================================================================
# 2. Session Memory with Sliding Window & Context Bloat Management
# ==============================================================================

class SessionMemory:
  """
  Maintains stateful memory with sliding window eviction, entity resolution,
  and progressive summarization to avoid LLM context bloat.
  """

  def __init__(self, session_id: str):
    self.session_id = session_id
    self.created_at = time.time()
    self.last_active = time.time()
    self.turns: List[ConversationTurn] = []
    self.running_summary: str = ""

    # Tracked Entities across conversation
    self.tracking_numbers: List[str] = []
    self.active_tracking_number: Optional[str] = None
    self.zip_codes: List[str] = []
    self.weights: List[float] = []
    self.order_ids: List[str] = []
    self.customer_intent_history: List[str] = []

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "SessionMemory":
    mem = cls(data["session_id"])
    mem.created_at = data.get("created_at", time.time())
    mem.last_active = data.get("last_active", time.time())
    mem.running_summary = data.get("running_summary", "")
    mem.active_tracking_number = data.get("active_tracking_id")
    mem.turns = data.get("turns", [])

    entities = data.get("entities", {})
    mem.tracking_numbers = entities.get("tracking_numbers", [])
    mem.zip_codes = entities.get("zip_codes", [])
    mem.weights = entities.get("weights", [])
    mem.order_ids = entities.get("order_ids", [])
    mem.customer_intent_history = entities.get("intent_history", [])
    return mem

  def extract_entities(self, text: str) -> Dict[str, Any]:
    """Extracts logistics domain entities using strict regex patterns."""
    detected = {}

    # Tracking numbers (SW-XXXXXXXXX)
    tracking_matches = re.findall(r"\bSW-[0-9A-Z]{8,12}\b", text, re.IGNORECASE)
    if tracking_matches:
      upper_tracks = [t.upper() for t in tracking_matches]
      detected["tracking_numbers"] = upper_tracks
      for t in upper_tracks:
        if t not in self.tracking_numbers:
          self.tracking_numbers.append(t)
      self.active_tracking_number = upper_tracks[-1]

    # ZIP codes (5 consecutive digits)
    zip_matches = re.findall(r"\b[0-9]{5}\b", text)
    if zip_matches:
      detected["zip_codes"] = zip_matches
      for z in zip_matches:
        if z not in self.zip_codes:
          self.zip_codes.append(z)

    # Weights
    weight_matches = re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:lbs?|pounds?|kg)\b", text, re.IGNORECASE)
    if weight_matches:
      weights = [float(w) for w in weight_matches]
      detected["weights"] = weights
      self.weights.extend(weights)

    # Order numbers
    order_matches = re.findall(r"\b(?:ORD-|Order\s*#?)\s*([0-9A-Z]{4,10})\b", text, re.IGNORECASE)
    if order_matches:
      orders = [f"ORD-{o.upper()}" for o in order_matches]
      detected["order_ids"] = orders
      for o in orders:
        if o not in self.order_ids:
          self.order_ids.append(o)

    return detected

  def add_user_turn(self, query: str) -> Dict[str, Any]:
    """Registers incoming query, extracts entities, and records timestamp."""
    self.last_active = time.time()
    entities = self.extract_entities(query)
    turn = ConversationTurn(
        role="user",
        content=query,
        timestamp=time.time(),
        entities_detected=entities,
    )
    self.turns.append(turn)
    return entities

  def add_agent_turn(
      self,
      response: str,
      route: Optional[str] = None,
      tools_used: Optional[List[str]] = None,
  ) -> None:
    """Registers agent output and updates intent history."""
    self.last_active = time.time()
    if route:
      self.customer_intent_history.append(route)
    turn = ConversationTurn(
        role="agent",
        content=response,
        timestamp=time.time(),
        route=route,
        tools_used=tools_used or [],
    )
    self.turns.append(turn)

  # ============================================================================
  # Context Bloat Management: Sliding Window & Summarization
  # ============================================================================

  def get_sliding_window_turns(self, window_size: int = MAX_RECENT_TURNS) -> List[ConversationTurn]:
    """Returns the most recent N turns to prevent prompt context bloat."""
    return self.turns[-window_size:] if len(self.turns) > window_size else self.turns

  def compress_older_turns(self) -> None:
    """
    Condenses turns outside the sliding window into the running summary,
    keeping prompt payload lightweight while retaining full conversational context.
    """
    if len(self.turns) <= MAX_RECENT_TURNS:
      return

    older_turns = self.turns[:-MAX_RECENT_TURNS]
    # Build incremental summary snippet
    summary_snippets = []
    for t in older_turns:
      role_tag = "Customer" if t.role == "user" else "Agent"
      snippet = t.content.strip().replace("\n", " ")
      if len(snippet) > 100:
        snippet = snippet[:100] + "..."
      summary_snippets.append(f"{role_tag}: {snippet}")

    new_summary = " | ".join(summary_snippets)
    if self.running_summary:
      combined = f"{self.running_summary} || {new_summary}"
    else:
      combined = new_summary

    if len(combined) > MAX_SUMMARY_LENGTH:
      combined = "..." + combined[-(MAX_SUMMARY_LENGTH - 3):]

    self.running_summary = combined

  def enrich_query_with_context(self, query: str) -> str:
    """
    Enriches a user query with sliding window memory and active entity references
    to resolve cross-turn pronouns without blowing up context size.
    """
    context_notes = []

    # 1. Resolve implicit pronouns ('it', 'my parcel', 'the shipment')
    implicit_refs = ["it", "this package", "my parcel", "the shipment", "where is it", "when will it arrive", "track it"]
    has_implicit = any(ref in query.lower() for ref in implicit_refs)

    if has_implicit and self.active_tracking_number and self.active_tracking_number not in query:
      context_notes.append(f"Active Subject: Tracking #{self.active_tracking_number}")

    if self.zip_codes and len(self.zip_codes) >= 2 and any(k in query.lower() for k in ["rate", "cost", "quote"]):
      context_notes.append(f"Active Postal Zones: Origin={self.zip_codes[0]}, Dest={self.zip_codes[1]}")

    if self.order_ids and "return" in query.lower() and not any(o in query for o in self.order_ids):
      context_notes.append(f"Active Order: {self.order_ids[-1]}")

    # 2. Add concise running summary if past turns exist
    if self.running_summary:
      context_notes.append(f"Prior Conversation: {self.running_summary}")

    if context_notes:
      injected = " [Context: " + "; ".join(context_notes) + "]"
      return f"{query}{injected}"
    return query

  def get_context_summary(self) -> Dict[str, Any]:
    """Returns structured metadata of active session context."""
    return {
        "session_id": self.session_id,
        "turn_count": len(self.turns),
        "sliding_window_active_turns": len(self.get_sliding_window_turns()),
        "has_running_summary": bool(self.running_summary),
        "running_summary": self.running_summary,
        "active_tracking_number": self.active_tracking_number,
        "all_tracking_numbers": self.tracking_numbers,
        "zip_codes": self.zip_codes,
        "weights": self.weights,
        "order_ids": self.order_ids,
        "recent_intents": self.customer_intent_history[-3:],
    }

  def get_history_messages(self) -> List[Dict[str, Any]]:
    """Returns serialized list of all conversation turns."""
    return [asdict(t) for t in self.turns]


# ==============================================================================
# 3. Asynchronous Background Memory Manager
# ==============================================================================

class MemoryStore:
  """
  High-performance memory coordinator combining fast in-memory caching
  with asynchronous background persistence and non-blocking summarization.
  """

  _cache: Dict[str, SessionMemory] = {}

  @classmethod
  def get_or_create(cls, session_id: str) -> SessionMemory:
    """Loads session from in-memory cache or restores from persistent SQLite database."""
    if session_id in cls._cache:
      return cls._cache[session_id]

    # Attempt restore from persistent storage
    persisted = _storage.load_session(session_id)
    if persisted:
      mem = SessionMemory.from_dict(persisted)
      cls._cache[session_id] = mem
      if logger:
        logger.info(
            f"Restored session '{session_id}' from persistent database",
            extra={"event_type": "memory_restored", "structured_context": {"session_id": session_id, "turn_count": len(mem.turns)}},
        )
      return mem

    # Create new session memory
    mem = SessionMemory(session_id)
    cls._cache[session_id] = mem
    return mem

  @classmethod
  def get(cls, session_id: str) -> Optional[SessionMemory]:
    """Retrieves session from cache or persistent database."""
    if session_id in cls._cache:
      return cls._cache[session_id]
    persisted = _storage.load_session(session_id)
    if persisted:
      mem = SessionMemory.from_dict(persisted)
      cls._cache[session_id] = mem
      return mem
    return None

  @classmethod
  async def save_turn_async(cls, session_id: str, turn: ConversationTurn) -> None:
    """
    Executes SQLite persistence and context bloat compression asynchronously in a background thread,
    preventing any I/O latency from blocking the main chat request loop.
    """
    def _background_worker():
      mem = cls._cache.get(session_id)
      if mem:
        mem.compress_older_turns()
        entities = {
            "tracking_numbers": mem.tracking_numbers,
            "zip_codes": mem.zip_codes,
            "weights": mem.weights,
            "order_ids": mem.order_ids,
            "intent_history": mem.customer_intent_history,
        }
        _storage.save_session(
            session_id=session_id,
            created_at=mem.created_at,
            last_active=mem.last_active,
            turn_count=len(mem.turns),
            active_tracking_id=mem.active_tracking_number,
            running_summary=mem.running_summary,
            entities=entities,
        )
        _storage.save_turn(session_id, turn)

    # Run in default executor without blocking asyncio event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _background_worker)

  @classmethod
  def schedule_background_save(cls, session_id: str, turn: ConversationTurn) -> None:
    """Schedules a non-blocking background task to persist the turn."""
    try:
      asyncio.create_task(cls.save_turn_async(session_id, turn))
    except RuntimeError:
      # If no running loop, execute synchronously as fallback
      mem = cls._cache.get(session_id)
      if mem:
        mem.compress_older_turns()
        _storage.save_turn(session_id, turn)

  @classmethod
  def clear(cls, session_id: Optional[str] = None) -> None:
    """Removes session from both memory cache and persistent database."""
    if session_id:
      cls._cache.pop(session_id, None)
      _storage.delete_session(session_id)
    else:
      cls._cache.clear()

  @classmethod
  def active_session_count(cls) -> int:
    return len(cls._cache)
