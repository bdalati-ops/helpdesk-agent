"""Unit tests for Observability, OpenTelemetry Tracing, Structured Logging, and PII Redaction."""

import json
import logging
import os
import sys
from opentelemetry import trace
from opentelemetry.trace import StatusCode

# Ensure package importability
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in (root_dir, parent_dir, current_dir):
  if p not in sys.path:
    sys.path.insert(0, p)

from customer_support_agent.observability import (
    PIIRedactor,
    StructuredJsonFormatter,
    setup_observability,
    get_tracer,
    get_logger,
    track_intent_vs_outcome,
    trace_span,
    test_span_exporter,
)
from customer_support_agent.agent import (
    QueryClassification,
    process_user_query,
    route_query,
    decline_unrelated_query,
)


# =====================================================================
# 1. PII Redactor Tests
# =====================================================================

def test_pii_redact_emails():
  raw = "Contact support@swiftship.com or customer.service@sub.example.org for help."
  redacted = PIIRedactor.redact(raw)
  assert "support@swiftship.com" not in redacted
  assert "customer.service@sub.example.org" not in redacted
  assert "[REDACTED_EMAIL]" in redacted


def test_pii_redact_phone_numbers():
  samples = [
      "Call me at 555-123-4567 please.",
      "Support line: (800) 555-0199.",
      "International: +1 555 123 4567.",
      "Direct: 555.987.6543 today.",
  ]
  for s in samples:
    redacted = PIIRedactor.redact(s)
    assert "[REDACTED_PHONE]" in redacted, f"Failed for {s}: {redacted}"


def test_pii_redact_credit_cards():
  raw = "Charged to card 4111-2222-3333-4444 and backup 5555 6666 7777 8888."
  redacted = PIIRedactor.redact(raw)
  assert "4111-2222-3333-4444" not in redacted
  assert "5555 6666 7777 8888" not in redacted
  assert "[REDACTED_CARD]" in redacted


def test_pii_redact_ssn():
  raw = "Customer SSN is 123-45-6789 on file."
  redacted = PIIRedactor.redact(raw)
  assert "123-45-6789" not in redacted
  assert "[REDACTED_SSN]" in redacted


def test_pii_redact_api_keys():
  samples = [
      "Token: ghp_" + ("mocktoken" * 4),
      "Key: AIza" + ("mockapikey" * 3) + "12345",
      "OAuth: ya29." + ("mockoauthtoken" * 3),
      "Secret: AQ." + ("mocksecretkey" * 3),
  ]
  for s in samples:
    redacted = PIIRedactor.redact(s)
    assert "[REDACTED_SECRET]" in redacted, f"Failed for {s}: {redacted}"


def test_pii_preserves_tracking_numbers():
  raw = "Where is package SW-123456789 and express parcel SW-987654321?"
  redacted = PIIRedactor.redact(raw)
  assert "SW-123456789" in redacted
  assert "SW-987654321" in redacted


def test_pii_redact_nested_structures():
  data = {
      "user": {"email": "user@test.com", "phone": "555-123-4567"},
      "notes": ["Call 800-555-1212", "Card: 4111 2222 3333 4444"],
      "safe_id": "SW-12345",
  }
  clean = PIIRedactor.redact_data(data)
  assert clean["user"]["email"] == "[REDACTED_EMAIL]"
  assert clean["user"]["phone"] == "[REDACTED_PHONE]"
  assert "[REDACTED_PHONE]" in clean["notes"][0]
  assert "[REDACTED_CARD]" in clean["notes"][1]
  assert clean["safe_id"] == "SW-12345"


# =====================================================================
# 2. Structured JSON Logging Tests
# =====================================================================

def test_structured_json_formatter():
  formatter = StructuredJsonFormatter(project_id="test-project")
  record = logging.LogRecord(
      name="test.logger",
      level=logging.INFO,
      pathname="/path/to/test.py",
      lineno=42,
      msg="Customer email is john@example.com",
      args=(),
      exc_info=None,
  )
  record.structured_context = {"order_id": "12345", "contact": "555-123-4567"}

  output = formatter.format(record)
  parsed = json.loads(output)

  assert parsed["severity"] == "INFO"
  assert parsed["logger"] == "test.logger"
  assert "[REDACTED_EMAIL]" in parsed["message"]
  assert "john@example.com" not in parsed["message"]
  assert parsed["context"]["order_id"] == "12345"
  assert parsed["context"]["contact"] == "[REDACTED_PHONE]"
  assert "timestamp" in parsed


def test_structured_json_trace_correlation():
  tracer = setup_observability(service_name="test-service", project_id="test-proj")
  formatter = StructuredJsonFormatter(project_id="test-proj")

  with tracer.start_as_current_span("test_span") as span:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.WARNING,
        pathname="/test.py",
        lineno=10,
        msg="Warning event",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)

    span_ctx = span.get_span_context()
    expected_trace_id = format(span_ctx.trace_id, "032x")
    expected_span_id = format(span_ctx.span_id, "016x")

    assert parsed["trace_id"] == expected_trace_id
    assert parsed["span_id"] == expected_span_id
    assert parsed["logging.googleapis.com/trace"] == f"projects/test-proj/traces/{expected_trace_id}"
    assert parsed["logging.googleapis.com/spanId"] == expected_span_id


# =====================================================================
# 3. OpenTelemetry Distributed Tracing Tests
# =====================================================================

def test_trace_span_context_manager():
  tracer = get_tracer("test-tracer")
  test_span_exporter.clear()

  with trace_span("custom_operation", attributes={"user.email": "test@domain.com"}) as span:
    span_ctx = span.get_span_context()
    assert span_ctx.is_valid

  # Exporter should capture the finished span
  assert len(test_span_exporter.spans) > 0
  exported = test_span_exporter.spans[-1]
  assert exported.name == "custom_operation"
  assert exported.status.status_code == StatusCode.OK
  # Verify attribute was sanitized
  assert exported.attributes["user.email"] == "[REDACTED_EMAIL]"


# =====================================================================
# 4. Intent vs. Outcome Tracking Tests
# =====================================================================

def test_track_intent_vs_outcome_aligned_shipping():
  tracer = get_tracer("test-tracer")
  with tracer.start_as_current_span("parent_turn") as parent:
    res = track_intent_vs_outcome(
        session_id="sess-101",
        user_id="cust-1",
        customer_query="How much to ship 5lbs to California? Call 555-0100",
        intent_category="shipping",
        intent_reasoning="User asking for rate estimate",
        target_node="shipping_faq_agent",
        outcome_action="answered_faq",
        nodes_visited=["process_user_query", "query_classifier", "route_query", "shipping_faq_agent"],
        latency_ms=185.2,
        response_summary="Standard Ground starts at $5.99",
    )

    assert res["outcome"]["alignment"] == "aligned"
    assert res["outcome"]["status"] == "success"
    assert "[REDACTED_PHONE]" in res["customer_query"]
    assert res["performance"]["latency_ms"] == 185.2


def test_track_intent_vs_outcome_aligned_unrelated():
  res = track_intent_vs_outcome(
      session_id="sess-102",
      user_id="cust-2",
      customer_query="What is the recipe for brownies?",
      intent_category="unrelated",
      intent_reasoning="Cooking question",
      target_node="decline_unrelated_query",
      outcome_action="declined_unrelated",
      nodes_visited=["process_user_query", "query_classifier", "route_query", "decline_unrelated_query"],
      latency_ms=95.0,
      response_summary="Politely declined non-shipping question",
  )

  assert res["outcome"]["alignment"] == "aligned"
  assert res["outcome"]["target_node"] == "decline_unrelated_query"
  assert res["outcome"]["action"] == "declined_unrelated"


def test_track_intent_vs_outcome_error():
  res = track_intent_vs_outcome(
      session_id="sess-103",
      user_id="cust-3",
      customer_query="Track SW-111",
      intent_category="shipping",
      intent_reasoning="Tracking inquiry",
      target_node="shipping_faq_agent",
      outcome_action="failed",
      nodes_visited=["process_user_query"],
      latency_ms=50.0,
      error="API quota exceeded",
  )

  assert res["outcome"]["alignment"] == "mismatched"
  assert res["outcome"]["status"] == "error"
  assert res["outcome"]["error"] == "API quota exceeded"


# =====================================================================
# 5. Workflow Node Instrumentation Tests
# =====================================================================

def test_process_user_query_instrumentation():
  event = process_user_query("Send package to john@test.com")
  assert event.output == "Send package to john@test.com"
  assert event.actions.state_delta["user_query"] == "Send package to john@test.com"
  assert "workflow_start_time" in event.actions.state_delta


def test_route_query_instrumentation():
  classification = QueryClassification(category="shipping", reasoning="Rate check")
  events = list(route_query(classification))
  assert len(events) == 1
  assert events[0].actions.route == "shipping"


def test_decline_unrelated_query_instrumentation():
  events = list(decline_unrelated_query())
  assert len(events) == 2
  assert "SwiftShip Customer Support" in events[1].output


if __name__ == "__main__":
  test_pii_redact_emails()
  test_pii_redact_phone_numbers()
  test_pii_redact_credit_cards()
  test_pii_redact_ssn()
  test_pii_redact_api_keys()
  test_pii_preserves_tracking_numbers()
  test_pii_redact_nested_structures()
  test_structured_json_formatter()
  test_structured_json_trace_correlation()
  test_trace_span_context_manager()
  test_track_intent_vs_outcome_aligned_shipping()
  test_track_intent_vs_outcome_aligned_unrelated()
  test_track_intent_vs_outcome_error()
  test_process_user_query_instrumentation()
  test_route_query_instrumentation()
  test_decline_unrelated_query_instrumentation()
  print("All 16 observability and tracing unit tests PASSED successfully!")
