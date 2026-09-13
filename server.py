"""FastAPI web server serving the Customer Support Agent and interactive Web UI."""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google.adk.runners import InMemoryRunner
from google.genai import types
from opentelemetry import trace

# Setup paths and load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
  sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

load_dotenv(os.path.join(current_dir, ".env"))
load_dotenv()

# Observability, Structured JSON Logging & OpenTelemetry
try:
  from customer_support_agent.observability import (
      PIIRedactor,
      get_logger,
      get_tracer,
      setup_observability,
      track_intent_vs_outcome,
  )
except ImportError:
  from observability import (
      PIIRedactor,
      get_logger,
      get_tracer,
      setup_observability,
      track_intent_vs_outcome,
  )

tracer = setup_observability(service_name="swiftship-customer-support")
logger = get_logger("swiftship.server")

# Secure Secret Management (GCP Secret Manager with env fallback)
try:
  from customer_support_agent.secrets_manager import initialize_app_secrets
except ImportError:
  from secrets_manager import initialize_app_secrets
initialize_app_secrets()

from customer_support_agent.agent import root_agent

# Resilience & Fault Tolerance (503 UNAVAILABLE & demand spike handling)
try:
  from customer_support_agent.resilience import (
      classify_query_heuristically,
      generate_fallback_shipping_response,
      generate_fallback_decline_response,
      is_retryable_error,
  )
except ImportError:
  from resilience import (
      classify_query_heuristically,
      generate_fallback_shipping_response,
      generate_fallback_decline_response,
      is_retryable_error,
  )

app = FastAPI(
    title="SwiftShip Customer Support Agent",
    description="ADK 2.0 Graph Workflow Agent for Shipping Customer Support with Observability & Tracing",
    version="2.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
  """Intercepts HTTP requests to inject distributed tracing and structured access logs."""
  start_time = time.perf_counter()
  method = request.method
  path = request.url.path

  with tracer.start_as_current_span(
      f"http.{method.lower()}",
      attributes={
          "http.method": method,
          "http.url": str(request.url),
          "http.target": path,
          "http.client_ip": request.client.host if request.client else "unknown",
      },
  ) as span:
    span_ctx = span.get_span_context()
    trace_id_hex = (
        format(span_ctx.trace_id, "032x") if span_ctx.is_valid else ""
    )

    try:
      response: Response = await call_next(request)
      duration_ms = (time.perf_counter() - start_time) * 1000

      if trace_id_hex:
        response.headers["X-Trace-Id"] = trace_id_hex
      span.set_attribute("http.status_code", response.status_code)

      if not path.startswith("/static"):
        logger.info(
            f"{method} {path} -> {response.status_code} ({round(duration_ms, 1)}ms)",
            extra={
                "event_type": "http_access",
                "structured_context": {
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": (
                        request.client.host if request.client else "unknown"
                    ),
                },
            },
        )
      return response
    except Exception as exc:
      duration_ms = (time.perf_counter() - start_time) * 1000
      span.set_attribute("http.status_code", 500)
      logger.error(
          f"Unhandled exception on {method} {path}: {exc}",
          exc_info=True,
          extra={
              "event_type": "http_error",
              "structured_context": {
                  "method": method,
                  "path": path,
                  "duration_ms": round(duration_ms, 2),
                  "error": str(exc),
              },
          },
      )
      raise


# Static directory for web interface
static_dir = os.path.join(current_dir, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize ADK Runner and Session Store
APP_NAME = "swiftship_customer_support"
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
sessions: dict[str, str] = {}


class ChatRequest(BaseModel):
  message: str
  session_id: Optional[str] = None


class ChatResponse(BaseModel):
  response: str
  session_id: str
  nodes_visited: list[str]
  route: Optional[str] = None
  reasoning: Optional[str] = None
  trace_id: Optional[str] = None
  latency_ms: Optional[float] = None
  intent_vs_outcome: Optional[Dict[str, Any]] = None


@app.get("/health")
async def health_check():
  """Health check endpoint for Cloud Run."""
  return {"status": "ok", "app": APP_NAME, "version": "1.0.0"}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
  """Serves the main customer support chat interface."""
  index_path = os.path.join(static_dir, "index.html")
  if os.path.exists(index_path):
    return FileResponse(index_path)
  return HTMLResponse("<h1>SwiftShip Customer Support</h1><p>UI loading...</p>")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
  """Processes a user message through the ADK 2.0 graph workflow with OpenTelemetry and intent tracking."""
  turn_start = time.perf_counter()
  user_text = request.message.strip()
  if not user_text:
    raise HTTPException(status_code=400, detail="Message cannot be empty.")

  with tracer.start_as_current_span("customer_support.chat_turn") as turn_span:
    span_ctx = turn_span.get_span_context()
    trace_id_hex = (
        format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    )

    # Redact customer query for privacy-safe logs and trace attributes
    clean_query = PIIRedactor.redact(user_text)
    turn_span.set_attribute("customer.query_redacted", clean_query)

    # Ensure or retrieve session
    session_id = request.session_id
    if not session_id or session_id not in sessions:
      session = await runner.session_service.create_session(
          app_name=APP_NAME,
          user_id="web_user",
      )
      session_id = session.id
      sessions[session_id] = session_id

    turn_span.set_attribute("session.id", session_id)
    turn_span.set_attribute("user.id", "web_user")

    logger.info(
        f"Processing chat turn for session {session_id}",
        extra={
            "event_type": "chat_turn_start",
            "structured_context": {
                "session_id": session_id,
                "user_id": "web_user",
                "query_redacted": clean_query[:120],
                "query_length": len(user_text),
            },
        },
    )

    message_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_text)],
    )

    nodes_visited = []
    response_texts = []
    route_taken = None
    reasoning_text = None

    try:
      with tracer.start_as_current_span("workflow.runner_execution") as wf_span:
        max_attempts = 3
        last_err = None
        for attempt in range(1, max_attempts + 1):
          try:
            nodes_visited.clear()
            response_texts.clear()
            route_taken = None
            reasoning_text = None

            async for event in runner.run_async(
                user_id="web_user",
                session_id=session_id,
                new_message=message_content,
            ):
              # Track graph execution nodes
              if event.node_info and event.node_info.path:
                node_name = event.node_info.path.split("/")[-1].split("@")[0]
                if node_name not in nodes_visited:
                  nodes_visited.append(node_name)

              # Track route if emitted
              if hasattr(event, "actions") and event.actions and event.actions.route:
                route_taken = event.actions.route

              # Collect agent messages
              if event.content and event.content.parts:
                for part in event.content.parts:
                  if part.text:
                    # Check if this part contains classifier JSON metadata
                    if '"category":' in part.text and '"reasoning":' in part.text:
                      try:
                        parsed = json.loads(part.text)
                        route_taken = parsed.get("category", route_taken)
                        reasoning_text = parsed.get("reasoning")
                      except Exception:
                        pass
                    else:
                      response_texts.append(part.text)

            last_err = None
            break
          except Exception as run_err:
            last_err = run_err
            if is_retryable_error(run_err) and attempt < max_attempts:
              backoff = 0.5 * (2 ** (attempt - 1))
              logger.warning(
                  f"Transient model error (attempt {attempt}/{max_attempts}): {run_err}. Retrying in {backoff:.2f}s...",
                  extra={
                      "event_type": "runner_retry",
                      "structured_context": {"attempt": attempt, "backoff": backoff, "error": str(run_err)},
                  },
              )
              await asyncio.sleep(backoff)
            else:
              raise run_err

        if last_err:
          raise last_err

        wf_span.set_attribute("workflow.nodes_visited", ",".join(nodes_visited))

      # Determine fallback route if not explicitly captured
      if not route_taken:
        if "shipping_faq_agent" in nodes_visited:
          route_taken = "shipping"
        elif "decline_unrelated_query" in nodes_visited:
          route_taken = "unrelated"
        else:
          route_taken = "shipping"

      final_response = "".join(response_texts).strip()
      if not final_response:
        final_response = (
            "Thank you for contacting SwiftShip. Your request has been processed."
        )

      duration_ms = (time.perf_counter() - turn_start) * 1000

      # Determine target node and outcome action
      target_node = (
          "shipping_faq_agent"
          if "shipping_faq_agent" in nodes_visited
          else "decline_unrelated_query"
      )
      outcome_action = (
          "answered_faq"
          if target_node == "shipping_faq_agent"
          else "declined_unrelated"
      )

      # Record Intent vs Outcome telemetry
      intent_outcome = track_intent_vs_outcome(
          session_id=session_id,
          user_id="web_user",
          customer_query=user_text,
          intent_category=route_taken,
          intent_reasoning=reasoning_text or "Classified by graph workflow",
          target_node=target_node,
          outcome_action=outcome_action,
          nodes_visited=nodes_visited,
          latency_ms=duration_ms,
          response_summary=final_response[:200],
      )

      return ChatResponse(
          response=final_response,
          session_id=session_id,
          nodes_visited=nodes_visited,
          route=route_taken,
          reasoning=reasoning_text,
          trace_id=trace_id_hex,
          latency_ms=round(duration_ms, 2),
          intent_vs_outcome=intent_outcome,
      )

    except Exception as e:
      duration_ms = (time.perf_counter() - turn_start) * 1000
      logger.warning(
          f"Workflow execution encountered model outage/transient error ({e}). Engaging graceful degradation.",
          extra={
              "event_type": "workflow_graceful_fallback",
              "structured_context": {
                  "session_id": session_id,
                  "duration_ms": round(duration_ms, 2),
                  "error": str(e),
                  "is_retryable": is_retryable_error(e),
              },
          },
      )

      # Intelligent heuristic fallback response
      fallback_intent = classify_query_heuristically(user_text)
      route_taken = fallback_intent["category"]
      reasoning_text = fallback_intent["reasoning"]

      if route_taken == "shipping":
        target_node = "shipping_faq_agent"
        outcome_action = "answered_faq_fallback"
        final_response = generate_fallback_shipping_response(user_text)
        nodes_visited = ["process_user_query", "query_classifier", "route_query", "shipping_faq_agent"]
      else:
        target_node = "decline_unrelated_query"
        outcome_action = "declined_unrelated_fallback"
        final_response = generate_fallback_decline_response()
        nodes_visited = ["process_user_query", "query_classifier", "route_query", "decline_unrelated_query"]

      intent_outcome = track_intent_vs_outcome(
          session_id=session_id,
          user_id="web_user",
          customer_query=user_text,
          intent_category=route_taken,
          intent_reasoning=reasoning_text,
          target_node=target_node,
          outcome_action=outcome_action,
          nodes_visited=nodes_visited,
          latency_ms=duration_ms,
          response_summary=final_response[:200],
      )

      return ChatResponse(
          response=final_response,
          session_id=session_id,
          nodes_visited=nodes_visited,
          route=route_taken,
          reasoning=reasoning_text,
          trace_id=trace_id_hex,
          latency_ms=round(duration_ms, 2),
          intent_vs_outcome=intent_outcome,
      )


if __name__ == "__main__":
  import uvicorn

  port = int(os.environ.get("PORT", 8080))
  uvicorn.run(app, host="0.0.0.0", port=port)
