"""
Unit tests for Strategic Model Routing and Human-in-the-Loop (HITL) confirmation hooks.
"""

import pytest
from agent import get_strategic_model
from hitl import HITLManager, HIGH_VALUE_REFUND_THRESHOLD


def setup_function():
  HITLManager.clear()


def test_strategic_model_routing():
  fast_model = get_strategic_model("fast_triage")
  synthesis_model = get_strategic_model("synthesis")
  complex_model = get_strategic_model("complex_reasoning")

  assert "flash" in fast_model.lower()
  assert "flash" in synthesis_model.lower() or "pro" in synthesis_model.lower()
  assert "pro" in complex_model.lower() or "flash" in complex_model.lower()
  assert fast_model != complex_model or "gemini" in fast_model


def test_hitl_address_reroute_detection():
  ticket = HITLManager.evaluate_hitl_requirement(
      session_id="sess-reroute-1",
      query="I need to change address and reroute package SW-123456789 to Miami.",
      intent="shipping",
      entities={"active_tracking_number": "SW-123456789"},
  )
  assert ticket is not None
  assert ticket.action_type == "address_reroute"
  assert ticket.status == "pending_approval"
  assert "SW-123456789" in ticket.summary


def test_hitl_high_value_refund_detection():
  ticket = HITLManager.evaluate_hitl_requirement(
      session_id="sess-refund-1",
      query="My package was completely destroyed, I demand a full refund of $350.00 right now.",
      intent="shipping",
  )
  assert ticket is not None
  assert ticket.action_type == "high_value_refund"
  assert ticket.details["claimed_amount"] == 350.0
  assert ticket.status == "pending_approval"


def test_hitl_supervisor_escalation_detection():
  ticket = HITLManager.evaluate_hitl_requirement(
      session_id="sess-human-1",
      query="This bot is not understanding me, let me talk to a human supervisor immediately.",
      intent="shipping",
  )
  assert ticket is not None
  assert ticket.action_type == "supervisor_escalation"


def test_hitl_approval_and_rejection_lifecycle():
  ticket = HITLManager.evaluate_hitl_requirement(
      session_id="sess-life-1",
      query="Please reroute parcel SW-554433221 to a new address.",
      intent="shipping",
  )
  assert ticket is not None

  # Approve ticket
  approved = HITLManager.approve_ticket(ticket.ticket_id, approver="supervisor_dan", notes="Verified caller ID")
  assert approved is not None
  assert approved.status == "approved"
  assert approved.resolved_by == "supervisor_dan"

  # Create another ticket and reject
  ticket2 = HITLManager.evaluate_hitl_requirement(
      session_id="sess-life-2",
      query="I want a $500 claim refund.",
      intent="shipping",
  )
  rejected = HITLManager.reject_ticket(ticket2.ticket_id, reviewer="supervisor_alice", reason="Missing receipt")
  assert rejected is not None
  assert rejected.status == "rejected"
  assert rejected.resolution_notes == "Missing receipt"
