"""Observability & Tracing Module for SwiftShip Customer Support Agent.

Provides:
- OpenTelemetry distributed tracing with span instrumentation.
- Structured JSON logging with Google Cloud Logging / W3C trace correlation.
- PII redaction (email, phone, credit cards, SSN, API tokens).
- Intent vs. Outcome tracking for agent decision auditing and evaluation.
"""

from contextlib import contextmanager
import datetime
import json
import logging
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, StatusCode, Tracer


# =====================================================================
# 1. PII Redaction Engine
# =====================================================================

class PIIRedactor:
  """Sanitizes text and structured payloads to remove personally identifiable info."""

  # Email pattern (e.g. customer@example.com)
  _EMAIL_RE = re.compile(
      r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE
  )

  # US / International telephone patterns (e.g. 555-123-4567, (800) 555-0199, +1 555 123 4567, 555-0100)
  _PHONE_RE = re.compile(
      r"(?:\+?1[-. ]?)?(?:\([0-9]{3}\)|[0-9]{3})[-. ]?[0-9]{3}[-. ][0-9]{4}|\b[0-9]{3}[-. ][0-9]{4}\b"
  )

  # Credit / Debit Cards (13-16 digits with optional spaces or hyphens)
  _CARD_RE = re.compile(
      r"\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{4}[ -]?\d{6}[ -]?\d{5}\b"
  )

  # Social Security Number (SSN: ###-##-####)
  _SSN_RE = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")

  # API Keys & Auth Tokens (Google AI, GitHub, GCP access tokens)
  _API_KEY_RE = re.compile(
      r"\b(?:ghp_[a-zA-Z0-9]{36}|AIza[0-9A-Za-z-_]{35}|ya29\.[0-9A-Za-z-_]+|AQ\.[0-9A-Za-z-_]{30,})\b"
  )

  @classmethod
  def redact(cls, text: Optional[str]) -> str:
    """Redacts known PII patterns from the input string."""
    if not text or not isinstance(text, str):
      return "" if text is None else str(text)

    # Note: SwiftShip tracking numbers like "SW-123456789" are preserved
    # because they do not match SSN, phone, card, or email patterns.
    s = cls._EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    s = cls._PHONE_RE.sub("[REDACTED_PHONE]", s)
    s = cls._CARD_RE.sub("[REDACTED_CARD]", s)
    s = cls._SSN_RE.sub("[REDACTED_SSN]", s)
    s = cls._API_KEY_RE.sub("[REDACTED_SECRET]", s)
    return s

  @classmethod
  def redact_data(cls, data: Any) -> Any:
    """Recursively redacts PII strings within dictionaries, lists, or primitives."""
    if isinstance(data, str):
      return cls.redact(data)
    elif isinstance(data, dict):
      return {k: cls.redact_data(v) for k, v in data.items()}
    elif isinstance(data, list):
      return [cls.redact_data(item) for item in data]
    elif isinstance(data, tuple):
      return tuple(cls.redact_data(item) for item in data)
    return data


# =====================================================================
# 2. Structured JSON Logging with Trace Correlation
# =====================================================================

class StructuredJsonFormatter(logging.Formatter):
  """Formats log records as single-line JSON objects with trace correlation."""

  def __init__(self, project_id: Optional[str] = None):
    super().__init__()
    self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")

  def format(self, record: logging.LogRecord) -> str:
    # Basic log entry fields
    entry: Dict[str, Any] = {
        "timestamp": datetime.datetime.fromtimestamp(
            record.created, tz=datetime.timezone.utc
        ).isoformat(),
        "severity": record.levelname,
        "logger": record.name,
        "message": PIIRedactor.redact(record.getMessage()),
        "source_location": {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        },
    }

    # OpenTelemetry trace and span correlation
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
      ctx = span.get_span_context()
      trace_id_hex = format(ctx.trace_id, "032x")
      span_id_hex = format(ctx.span_id, "016x")

      entry["trace_id"] = trace_id_hex
      entry["span_id"] = span_id_hex
      entry["trace_sampled"] = bool(ctx.trace_flags.sampled)

      # Google Cloud Logging trace correlation format
      if self.project_id:
        entry["logging.googleapis.com/trace"] = (
            f"projects/{self.project_id}/traces/{trace_id_hex}"
        )
      else:
        entry["logging.googleapis.com/trace"] = trace_id_hex
      entry["logging.googleapis.com/spanId"] = span_id_hex
      entry["logging.googleapis.com/trace_sampled"] = bool(ctx.trace_flags.sampled)

    # Attach structured context / extra attributes
    if hasattr(record, "structured_context") and isinstance(
        record.structured_context, dict
    ):
      entry["context"] = PIIRedactor.redact_data(record.structured_context)

    if hasattr(record, "event_type"):
      entry["event_type"] = record.event_type

    # Exception and traceback details
    if record.exc_info:
      entry["exception"] = {
          "type": record.exc_info[0].__name__ if record.exc_info[0] else "Exception",
          "message": str(record.exc_info[1]) if record.exc_info[1] else "",
          "stack": self.formatException(record.exc_info),
      }

    return json.dumps(entry, ensure_ascii=False)


# =====================================================================
# 3. OpenTelemetry Tracer & Provider Setup
# =====================================================================

_IS_INITIALIZED = False
_TRACER: Optional[Tracer] = None


class InMemorySpanExporter(SpanExporter):
  """In-memory exporter for hermetic testing and verification of spans."""

  def __init__(self, max_spans: int = 1000):
    self.spans: List[Any] = []
    self.max_spans = max_spans

  def export(self, spans: List[Any]) -> SpanExportResult:
    self.spans.extend(spans)
    if len(self.spans) > self.max_spans:
      self.spans = self.spans[-self.max_spans:]
    return SpanExportResult.SUCCESS

  def shutdown(self) -> None:
    self.spans.clear()

  def clear(self) -> None:
    self.spans.clear()


# Global in-memory collector for inspections and tests
test_span_exporter = InMemorySpanExporter()


def setup_observability(
    service_name: str = "customer-support-agent",
    log_level: int = logging.INFO,
    enable_console_trace: bool = False,
    project_id: Optional[str] = None,
) -> Tracer:
  """Configures OpenTelemetry distributed tracing and structured JSON logging."""
  global _IS_INITIALIZED, _TRACER

  if _IS_INITIALIZED and _TRACER is not None:
    return _TRACER

  # 1. Setup OpenTelemetry Resource & TracerProvider
  resource = Resource.create({
      "service.name": service_name,
      "service.version": "1.0.0",
      "deployment.environment": os.environ.get("ENVIRONMENT", "production"),
  })

  provider = TracerProvider(resource=resource)

  # Add in-memory exporter for introspection and testing
  provider.add_span_processor(SimpleSpanProcessor(test_span_exporter))

  # Optionally add console exporter for local debug
  if enable_console_trace:
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

  trace.set_tracer_provider(provider)
  _TRACER = trace.get_tracer(service_name)

  # 2. Setup Structured JSON Logging Handler
  resolved_project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
  json_formatter = StructuredJsonFormatter(project_id=resolved_project)

  root_logger = logging.getLogger()
  root_logger.setLevel(log_level)

  # Replace existing StreamHandlers with the structured formatter
  has_json_handler = False
  for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
      handler.setFormatter(json_formatter)
      has_json_handler = True

  if not has_json_handler:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

  _IS_INITIALIZED = True
  return _TRACER


def get_tracer(name: str = "customer-support-agent") -> Tracer:
  """Returns the active OpenTelemetry tracer."""
  global _TRACER
  if _TRACER is None:
    return setup_observability(service_name=name)
  return _TRACER


def get_logger(name: str = "swiftship.customer_support") -> logging.Logger:
  """Returns a configured logger."""
  return logging.getLogger(name)


# =====================================================================
# 4. Intent vs. Outcome Tracking
# =====================================================================

def track_intent_vs_outcome(
    session_id: str,
    user_id: str,
    customer_query: str,
    intent_category: str,
    intent_reasoning: str,
    target_node: str,
    outcome_action: str,
    nodes_visited: List[str],
    latency_ms: float,
    error: Optional[str] = None,
    response_summary: Optional[str] = None,
) -> Dict[str, Any]:
  """Tracks and logs Intent vs. Outcome comparison for auditing and evaluation."""
  logger = get_logger("swiftship.telemetry.intent_outcome")
  span = trace.get_current_span()

  # Determine classification alignment
  is_shipping_aligned = (
      intent_category == "shipping"
      and target_node in ("shipping_faq_agent", "shipping")
  )
  is_unrelated_aligned = (
      intent_category == "unrelated"
      and target_node in ("decline_unrelated_query", "unrelated")
  )
  is_aligned = (is_shipping_aligned or is_unrelated_aligned) and not error

  record: Dict[str, Any] = {
      "session_id": session_id,
      "user_id": user_id,
      "customer_query": PIIRedactor.redact(customer_query),
      "intent": {
          "category": intent_category,
          "reasoning": PIIRedactor.redact(intent_reasoning),
      },
      "outcome": {
          "target_node": target_node,
          "action": outcome_action,
          "alignment": "aligned" if is_aligned else "mismatched",
          "status": "error" if error else "success",
          "nodes_visited": nodes_visited,
          "response_summary": PIIRedactor.redact(response_summary or ""),
      },
      "performance": {
          "latency_ms": round(latency_ms, 2),
      },
  }

  if error:
    record["outcome"]["error"] = PIIRedactor.redact(error)

  # Attach attributes to active OpenTelemetry span
  if span and span.is_recording():
    span.set_attribute("intent.category", intent_category)
    span.set_attribute("intent.reasoning", PIIRedactor.redact(intent_reasoning))
    span.set_attribute("outcome.target_node", target_node)
    span.set_attribute("outcome.action", outcome_action)
    span.set_attribute("outcome.alignment", "aligned" if is_aligned else "mismatched")
    span.set_attribute("outcome.status", "error" if error else "success")
    span.set_attribute("outcome.latency_ms", round(latency_ms, 2))
    span.set_attribute("session.id", session_id)
    span.set_attribute("user.id", user_id)

    if error:
      span.set_status(Status(StatusCode.ERROR, description=error))
      span.record_exception(Exception(error))
    else:
      span.set_status(Status(StatusCode.OK))

  # Emit structured log entry
  log_level = logging.ERROR if error else logging.INFO
  logger.log(
      log_level,
      f"Intent vs Outcome: {intent_category} -> {target_node} ({'aligned' if is_aligned else 'mismatched'}) in {round(latency_ms, 1)}ms",
      extra={
          "event_type": "intent_vs_outcome",
          "structured_context": record,
      },
  )

  return record


# =====================================================================
# 5. Context Manager and Helper Utilities
# =====================================================================

@contextmanager
def trace_span(
    span_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    tracer: Optional[Tracer] = None,
):
  """Context manager to wrap code execution in an OpenTelemetry span with sanitized attributes."""
  t = tracer or get_tracer()
  sanitized_attrs = PIIRedactor.redact_data(attributes or {}) if attributes else {}

  with t.start_as_current_span(span_name, attributes=sanitized_attrs) as span:
    try:
      yield span
      span.set_status(Status(StatusCode.OK))
    except Exception as exc:
      span.set_status(Status(StatusCode.ERROR, description=str(exc)))
      span.record_exception(exc)
      raise
