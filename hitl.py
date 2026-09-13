"""
Human-in-the-Loop (HITL) Confirmation & Governance Engine for SwiftShip Agent.

Provides policy safeguards and confirmation hooks for high-impact, sensitive,
or irreversible actions:
  1. High-Value Refunds / Claims (over configurable $100 threshold)
  2. In-Transit Address Modifications & Package Rerouting (anti-fraud protection)
  3. Human Supervisor Escalations (low intent confidence or explicit requests)
"""

import datetime
import os
import re
import secrets
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

try:
  from customer_support_agent.observability import PIIRedactor, get_logger, get_tracer
  logger = get_logger("swiftship.hitl")
  tracer = get_tracer("swiftship.hitl")
except ImportError:
  try:
    from observability import PIIRedactor, get_logger, get_tracer
    logger = get_logger("swiftship.hitl")
    tracer = get_tracer("swiftship.hitl")
  except ImportError:
    logger = None
    tracer = None

# Configurable thresholds
HIGH_VALUE_REFUND_THRESHOLD = float(os.getenv("HITL_REFUND_THRESHOLD", "100.0"))


@dataclass
class HITLConfirmationRequest:
  """Represents an action requiring human supervisor review and confirmation."""

  ticket_id: str
  session_id: str
  action_type: str  # 'high_value_refund', 'address_reroute', 'supervisor_escalation'
  status: Literal["pending_approval", "approved", "rejected"]
  summary: str
  details: Dict[str, Any]
  confirmation_token: str
  created_at: float = field(default_factory=time.time)
  resolved_at: Optional[float] = None
  resolved_by: Optional[str] = None
  resolution_notes: Optional[str] = None


class HITLManager:
  """Centralized manager for human-in-the-loop confirmation tickets and approval workflows."""

  _pending_tickets: Dict[str, HITLConfirmationRequest] = {}

  @classmethod
  def evaluate_hitl_requirement(
      cls,
      session_id: str,
      query: str,
      intent: str,
      entities: Optional[Dict[str, Any]] = None,
  ) -> Optional[HITLConfirmationRequest]:
    """
    Evaluates whether an incoming query triggers a Human-in-the-Loop (HITL) confirmation hook.
    """
    q = query.lower()

    # 1. In-Transit Address Changes / Rerouting (Severe Fraud Risk)
    reroute_triggers = ["change address", "reroute", "new address", "divert package", "different delivery address"]
    if any(t in q for t in reroute_triggers):
      tracking_id = (entities or {}).get("active_tracking_number")
      if not tracking_id:
        m = re.findall(r"\bSW-[0-9A-Z]{8,12}\b", query, re.IGNORECASE)
        tracking_id = m[0].upper() if m else "SW-UNKNOWN"

      return cls._create_ticket(
          session_id=session_id,
          action_type="address_reroute",
          summary=f"In-transit address change requested for shipment {tracking_id}",
          details={
              "query": query,
              "tracking_id": tracking_id,
              "risk_level": "HIGH (Address Diversion Fraud Risk)",
              "policy_rule": "Rerouting in transit requires dispatch supervisor verification of customer identity.",
          },
      )

    # 2. High-Value Damage Claims / Returns / Refunds
    refund_triggers = ["refund", "claim", "damage compensation", "stolen", "lost item", "reimburse"]
    if any(t in q for t in refund_triggers):
      # Look for explicit dollar values
      amounts = re.findall(r"\$([0-9]+(?:\.[0-9]+)?)", query)
      val = float(amounts[0]) if amounts else 0.0

      if val >= HIGH_VALUE_REFUND_THRESHOLD or "thousand" in q or "expensive" in q or val == 0.0 and any(w in q for w in ["claim", "stolen", "damaged item"]):
        return cls._create_ticket(
            session_id=session_id,
            action_type="high_value_refund",
            summary=f"Refund/damage claim authorization exceeding approval limit (${val:.2f})",
            details={
                "query": query,
                "claimed_amount": val,
                "threshold": HIGH_VALUE_REFUND_THRESHOLD,
                "risk_level": "FINANCIAL",
                "policy_rule": f"Refunds/claims exceeding ${HIGH_VALUE_REFUND_THRESHOLD:.2f} require supervisor sign-off.",
            },
        )

    # 3. Explicit Human Escalation Request
    human_triggers = ["talk to human", "speak to representative", "supervisor", "manager", "human agent", "real person", "representative"]
    if any(t in q for t in human_triggers):
      return cls._create_ticket(
          session_id=session_id,
          action_type="supervisor_escalation",
          summary="Customer explicitly requested transfer to a human support supervisor",
          details={
              "query": query,
              "reason": "Customer requested human representative",
              "priority": "HIGH",
          },
      )

    return None

  @classmethod
  def _create_ticket(
      cls,
      session_id: str,
      action_type: str,
      summary: str,
      details: Dict[str, Any],
  ) -> HITLConfirmationRequest:
    """Generates a secure pending confirmation ticket."""
    ticket_id = f"HITL-{datetime.datetime.utcnow().strftime('%Y%m%d')}-{secrets.token_hex(3).upper()}"
    token = secrets.token_urlsafe(16)

    ticket = HITLConfirmationRequest(
        ticket_id=ticket_id,
        session_id=session_id,
        action_type=action_type,
        status="pending_approval",
        summary=summary,
        details=details,
        confirmation_token=token,
    )
    cls._pending_tickets[ticket_id] = ticket

    if logger:
      logger.info(
          f"Created Human-in-the-Loop confirmation ticket '{ticket_id}' ({action_type})",
          extra={
              "event_type": "hitl_ticket_created",
              "structured_context": {
                  "ticket_id": ticket_id,
                  "session_id": session_id,
                  "action_type": action_type,
                  "summary": summary,
              },
          },
      )

    return ticket

  @classmethod
  def approve_ticket(cls, ticket_id: str, approver: str, notes: Optional[str] = None) -> Optional[HITLConfirmationRequest]:
    """Approves a pending confirmation ticket."""
    ticket = cls._pending_tickets.get(ticket_id)
    if not ticket:
      return None

    ticket.status = "approved"
    ticket.resolved_at = time.time()
    ticket.resolved_by = approver
    ticket.resolution_notes = notes or "Approved by human supervisor."

    if logger:
      logger.info(
          f"HITL ticket '{ticket_id}' approved by {approver}",
          extra={"event_type": "hitl_approved", "structured_context": {"ticket_id": ticket_id, "approver": approver}},
      )
    return ticket

  @classmethod
  def reject_ticket(cls, ticket_id: str, reviewer: str, reason: str) -> Optional[HITLConfirmationRequest]:
    """Rejects a pending confirmation ticket."""
    ticket = cls._pending_tickets.get(ticket_id)
    if not ticket:
      return None

    ticket.status = "rejected"
    ticket.resolved_at = time.time()
    ticket.resolved_by = reviewer
    ticket.resolution_notes = reason

    if logger:
      logger.info(
          f"HITL ticket '{ticket_id}' rejected by {reviewer}: {reason}",
          extra={"event_type": "hitl_rejected", "structured_context": {"ticket_id": ticket_id, "reviewer": reviewer, "reason": reason}},
      )
    return ticket

  @classmethod
  def get_ticket(cls, ticket_id: str) -> Optional[HITLConfirmationRequest]:
    return cls._pending_tickets.get(ticket_id)

  @classmethod
  def list_pending_tickets(cls) -> List[Dict[str, Any]]:
    return [
        asdict(t)
        for t in cls._pending_tickets.values()
        if t.status == "pending_approval"
    ]

  @classmethod
  def clear(cls) -> None:
    cls._pending_tickets.clear()
