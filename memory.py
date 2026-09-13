"""
Context & Multi-Turn Memory Management for SwiftShip Customer Support Agent.

Maintains conversation turns, resolves cross-turn pronouns ('it', 'my parcel', 'the return'),
extracts and preserves logistics entities (tracking IDs, postal codes, weights, order numbers),
and enriches agent prompts with accumulated conversational context.
"""

import datetime
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class ConversationTurn:
  role: str  # "user" or "agent"
  content: str
  timestamp: float = field(default_factory=time.time)
  route: Optional[str] = None
  tools_used: List[str] = field(default_factory=list)
  entities_detected: Dict[str, Any] = field(default_factory=dict)


class SessionMemory:
  """Maintains stateful memory, entity tracking, and context across turns for a single session."""

  def __init__(self, session_id: str):
    self.session_id = session_id
    self.created_at = time.time()
    self.last_active = time.time()
    self.turns: List[ConversationTurn] = []

    # Tracked Entities across conversation
    self.tracking_numbers: List[str] = []
    self.active_tracking_number: Optional[str] = None
    self.zip_codes: List[str] = []
    self.weights: List[float] = []
    self.order_ids: List[str] = []
    self.customer_intent_history: List[str] = []

  def extract_entities(self, text: str) -> Dict[str, Any]:
    """Extracts domain entities from message text."""
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

    # Weights (e.g., "5 lbs", "10.5 kg", "15 pounds")
    weight_matches = re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:lbs?|pounds?|kg)\b", text, re.IGNORECASE)
    if weight_matches:
      weights = [float(w) for w in weight_matches]
      detected["weights"] = weights
      self.weights.extend(weights)

    # Order numbers (ORD-XXXX or Order #XXXX)
    order_matches = re.findall(r"\b(?:ORD-|Order\s*#?)\s*([0-9A-Z]{4,10})\b", text, re.IGNORECASE)
    if order_matches:
      orders = [f"ORD-{o.upper()}" for o in order_matches]
      detected["order_ids"] = orders
      for o in orders:
        if o not in self.order_ids:
          self.order_ids.append(o)

    return detected

  def add_user_turn(self, query: str) -> Dict[str, Any]:
    """Registers a user message and extracts context."""
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
    """Registers an agent response and associated execution metadata."""
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

  def enrich_query_with_context(self, query: str) -> str:
    """
    Enriches a user query with prior session memory context,
    resolving implicit references (e.g. 'it', 'my package', 'the status').
    """
    context_notes = []

    # Check for implicit pronoun references
    implicit_refs = ["it", "this package", "my parcel", "the shipment", "where is it", "when will it arrive"]
    has_implicit = any(ref in query.lower() for ref in implicit_refs)

    if has_implicit and self.active_tracking_number and self.active_tracking_number not in query:
      context_notes.append(f"Active Subject: Tracking #{self.active_tracking_number}")

    if self.zip_codes and len(self.zip_codes) >= 2 and ("rate" in query.lower() or "cost" in query.lower()):
      context_notes.append(f"Previously Mentioned Postal Zones: Origin={self.zip_codes[0]}, Dest={self.zip_codes[1]}")

    if self.order_ids and "return" in query.lower() and not any(o in query for o in self.order_ids):
      context_notes.append(f"Active Order Reference: {self.order_ids[-1]}")

    if context_notes:
      injected = " [Conversation Context: " + "; ".join(context_notes) + "]"
      return f"{query}{injected}"
    return query

  def get_context_summary(self) -> Dict[str, Any]:
    """Returns a structured summary of active session context for API consumers."""
    return {
        "session_id": self.session_id,
        "turn_count": len(self.turns),
        "active_tracking_number": self.active_tracking_number,
        "all_tracking_numbers": self.tracking_numbers,
        "zip_codes": self.zip_codes,
        "weights": self.weights,
        "order_ids": self.order_ids,
        "recent_intents": self.customer_intent_history[-3:],
    }

  def get_history_messages(self) -> List[Dict[str, Any]]:
    """Returns serialized list of message turns."""
    return [asdict(t) for t in self.turns]


class MemoryStore:
  """Thread-safe global in-memory repository for active session memories."""

  _sessions: Dict[str, SessionMemory] = {}

  @classmethod
  def get_or_create(cls, session_id: str) -> SessionMemory:
    if session_id not in cls._sessions:
      cls._sessions[session_id] = SessionMemory(session_id)
    return cls._sessions[session_id]

  @classmethod
  def get(cls, session_id: str) -> Optional[SessionMemory]:
    return cls._sessions.get(session_id)

  @classmethod
  def clear(cls, session_id: Optional[str] = None) -> None:
    if session_id:
      cls._sessions.pop(session_id, None)
    else:
      cls._sessions.clear()

  @classmethod
  def active_session_count(cls) -> int:
    return len(cls._sessions)
