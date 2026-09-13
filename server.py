"""FastAPI web server serving the Customer Support Agent, REST APIs, and Web UI.

Features:
  - Tool & Interface Design: Exposes tools catalog, session inspection, and user feedback endpoints.
  - Context & Memory: Multi-turn session persistence, entity tracking, and cross-turn reference resolution.
  - Orchestration & Logic: Resilient retry, OpenTelemetry distributed tracing, and zero-downtime graceful fallback.
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
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
  from customer_support_agent.tools import get_available_tools_metadata
  from customer_support_agent.memory import MemoryStore, SessionMemory
  from customer_support_agent.hitl import HITLManager
  from customer_support_agent.resilience import (
      classify_query_heuristically,
      generate_fallback_shipping_response,
      generate_fallback_decline_response,
      is_retryable_error,
  )
except ImportError:
  from observability import (
      PIIRedactor,
      get_logger,
      get_tracer,
      setup_observability,
      track_intent_vs_outcome,
  )
  from tools import get_available_tools_metadata
  from memory import MemoryStore, SessionMemory
  from hitl import HITLManager
  from resilience import (
      classify_query_heuristically,
      generate_fallback_shipping_response,
      generate_fallback_decline_response,
      is_retryable_error,
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

app = FastAPI(
    title="SwiftShip Customer Support Agent API",
    description="ADK 2.0 Graph Workflow Agent with Tool Design, Multi-Turn Memory, and Tracing",
    version="2.1.0",
)

# Enable CORS for web UI clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def opentelemetry_logging_middleware(request: Request, call_next):
  """HTTP middleware that instruments incoming requests with OpenTelemetry spans and correlation headers."""
  start_time = time.perf_counter()
  method = request.method
  path = request.url.path

  with tracer.start_as_current_span(f"http.{method.lower()}") as span:
    span_ctx = span.get_span_context()
    trace_id_hex = (
        format(span_ctx.trace_id, "032x") if span_ctx.is_valid else "n/a"
    )

    span.set_attribute("http.method", method)
    span.set_attribute("http.route", path)
    span.set_attribute("http.url", str(request.url))

    try:
      response: Response = await call_next(request)
      duration_ms = (time.perf_counter() - start_time) * 1000

      span.set_attribute("http.status_code", response.status_code)
      span.set_attribute("http.duration_ms", duration_ms)

      if trace_id_hex != "n/a":
        response.headers["X-Trace-Id"] = trace_id_hex

      if not path.startswith("/static"):
        logger.info(
            f"{method} {path} - {response.status_code} ({duration_ms:.1f}ms)",
            extra={
                "event_type": "http_request",
                "structured_context": {
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            },
        )
      return response

    except Exception as exc:
      duration_ms = (time.perf_counter() - start_time) * 1000
      span.set_attribute("http.status_code", 500)
      span.record_exception(exc)
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

# Initialize ADK Runner
APP_NAME = "swiftship_customer_support"
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
sessions: dict[str, str] = {}


# ==============================================================================
# API Request & Response Schemas (Interface Design)
# ==============================================================================

class ChatRequest(BaseModel):
  message: str = Field(description="The user's query or message to customer support.")
  session_id: Optional[str] = Field(default=None, description="Unique session identifier for multi-turn memory.")


class ChatResponse(BaseModel):
  response: str = Field(description="The agent's text response to the user.")
  session_id: str = Field(description="Session identifier.")
  nodes_visited: list[str] = Field(description="DAG nodes traversed during execution.")
  route: Optional[str] = Field(default=None, description="Routing path chosen ('shipping' or 'unrelated').")
  reasoning: Optional[str] = Field(default=None, description="Intent classification rationale.")
  trace_id: Optional[str] = Field(default=None, description="OpenTelemetry distributed trace identifier.")
  latency_ms: Optional[float] = Field(default=None, description="Execution turn latency in milliseconds.")
  tools_used: List[Dict[str, Any]] = Field(default_factory=list, description="List of tools invoked during this turn.")
  memory_context: Optional[Dict[str, Any]] = Field(default=None, description="Active multi-turn entity and session state.")
  requires_human_confirmation: bool = Field(default=False, description="Whether this interaction triggered a Human-in-the-Loop hook.")
  hitl_ticket: Optional[Dict[str, Any]] = Field(default=None, description="Human-in-the-Loop confirmation ticket details.")
  intent_vs_outcome: Optional[Dict[str, Any]] = Field(default=None, description="Intent vs Outcome audit payload.")


class FeedbackRequest(BaseModel):
  session_id: str = Field(description="Session ID associated with the interaction.")
  rating: Literal["helpful", "unhelpful"] = Field(description="Customer rating.")
  comment: Optional[str] = Field(default=None, description="Optional user comment or feedback.")


class HITLConfirmRequest(BaseModel):
  ticket_id: str = Field(description="The HITL confirmation ticket ID (e.g. HITL-2026...).")
  action: Literal["approve", "reject"] = Field(description="Supervisor approval decision.")
  reviewer: str = Field(default="supervisor", description="Name or identifier of the reviewing supervisor.")
  notes: Optional[str] = Field(default=None, description="Optional justification or instructions.")


# ==============================================================================
# REST Endpoints
# ==============================================================================

@app.get("/health")
async def health_check():
  """Health check endpoint for Cloud Run and monitoring probes."""
  return {
      "status": "ok",
      "app": APP_NAME,
      "version": "2.1.0",
      "active_sessions": MemoryStore.active_session_count(),
  }


@app.get("/api/tools")
async def list_tools():
  """Returns the catalog of callable tools registered with the Customer Support Agent."""
  return {
      "tools": get_available_tools_metadata(),
      "count": len(get_available_tools_metadata()),
  }


@app.get("/api/session/{session_id}")
async def get_session_history(session_id: str):
  """Returns multi-turn conversation history and accumulated memory context for a session."""
  memory = MemoryStore.get(session_id)
  if not memory:
    raise HTTPException(status_code=404, detail="Session not found or has expired.")
  return {
      "session_id": session_id,
      "summary": memory.get_context_summary(),
      "turns": memory.get_history_messages(),
  }


@app.delete("/api/session/{session_id}")
async def reset_session(session_id: str):
  """Clears conversation history and memory context for a session."""
  MemoryStore.clear(session_id)
  sessions.pop(session_id, None)
  return {"status": "cleared", "session_id": session_id}


@app.post("/api/feedback")
async def record_feedback(feedback: FeedbackRequest):
  """Records customer satisfaction feedback."""
  logger.info(
      f"Received feedback for session {feedback.session_id}: {feedback.rating}",
      extra={
          "event_type": "user_feedback",
          "structured_context": {
              "session_id": feedback.session_id,
              "rating": feedback.rating,
              "comment": feedback.comment,
          },
      },
  )
  return {"status": "success", "message": "Feedback recorded. Thank you!"}


@app.get("/api/hitl/pending")
async def get_pending_hitl_tickets():
  """Returns all pending Human-in-the-Loop confirmation tickets awaiting supervisor review."""
  return {
      "pending_tickets": HITLManager.list_pending_tickets(),
      "count": len(HITLManager.list_pending_tickets()),
  }


@app.post("/api/hitl/confirm")
async def confirm_hitl_action(request: HITLConfirmRequest):
  """Allows a supervisor to approve or reject a pending high-impact action."""
  if request.action == "approve":
    ticket = HITLManager.approve_ticket(request.ticket_id, approver=request.reviewer, notes=request.notes)
  else:
    ticket = HITLManager.reject_ticket(request.ticket_id, reviewer=request.reviewer, reason=request.notes or "Rejected by supervisor")

  if not ticket:
    raise HTTPException(status_code=404, detail=f"Ticket '{request.ticket_id}' not found.")

  return {
      "status": "success",
      "ticket_id": ticket.ticket_id,
      "ticket_status": ticket.status,
      "resolution_notes": ticket.resolution_notes,
  }


@app.get("/", response_class=HTMLResponse)
async def serve_index():
  """Serves the interactive web chat interface."""
  index_path = os.path.join(static_dir, "index.html")
  if os.path.exists(index_path):
    return FileResponse(index_path)
  return HTMLResponse("<h1>SwiftShip Customer Support</h1><p>UI loading...</p>")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
  """Processes a user query through the tool-augmented ADK 2.0 graph workflow with async memory & tracing."""
  turn_start = time.perf_counter()
  user_text = request.message.strip()
  if not user_text:
    raise HTTPException(status_code=400, detail="Message cannot be empty.")

  with tracer.start_as_current_span("customer_support.chat_turn") as turn_span:
    span_ctx = turn_span.get_span_context()
    trace_id_hex = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None

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

    # Make session_id and user query accessible to workflow nodes
    os.environ["CURRENT_SESSION_ID"] = session_id
    os.environ["CURRENT_USER_QUERY"] = user_text

    turn_span.set_attribute("session.id", session_id)
    turn_span.set_attribute("user.id", "web_user")

    # Update session memory with incoming query and persist in background
    memory = MemoryStore.get_or_create(session_id)
    extracted_entities = memory.add_user_turn(user_text)
    enriched_query = memory.enrich_query_with_context(user_text)
    background_tasks.add_task(MemoryStore.save_turn_async, session_id, memory.turns[-1])

    # Evaluate Human-in-the-Loop confirmation requirement
    hitl_ticket_obj = HITLManager.evaluate_hitl_requirement(session_id, user_text, "shipping", extracted_entities)
    requires_hitl = hitl_ticket_obj is not None
    hitl_ticket_data = hitl_ticket_obj.__dict__ if hitl_ticket_obj else None
    turn_span.set_attribute("hitl.required", requires_hitl)

    logger.info(
        f"Processing chat turn for session {session_id}",
        extra={
            "event_type": "chat_turn_start",
            "structured_context": {
                "session_id": session_id,
                "user_id": "web_user",
                "query_redacted": clean_query[:120],
                "query_length": len(user_text),
                "entities": extracted_entities,
            },
        },
    )

    message_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=enriched_query)],
    )

    nodes_visited = []
    response_texts = []
    tools_used = []
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
            tools_used.clear()
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

              # Detect tool executions from events if present
              if hasattr(event, "tools") and event.tools:
                for t in event.tools:
                  tools_used.append({"name": getattr(t, "name", "tool")})

              # Collect agent messages
              if event.content and event.content.parts:
                for part in event.content.parts:
                  if hasattr(part, "function_call") and part.function_call:
                    tools_used.append({
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args or {}),
                    })
                  elif part.text:
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

      # Register agent response in multi-turn memory
      memory.add_agent_turn(
          response=final_response,
          route=route_taken,
          tools_used=[t.get("name", "tool") for t in tools_used],
      )
      background_tasks.add_task(MemoryStore.save_turn_async, session_id, memory.turns[-1])
      memory_summary = memory.get_context_summary()

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
          tools_used=tools_used,
          memory_context=memory_summary,
          requires_human_confirmation=requires_hitl,
          hitl_ticket=hitl_ticket_data,
          intent_vs_outcome=intent_outcome,
      )

    except Exception as e:
      duration_ms = (time.perf_counter() - turn_start) * 1000
      logger.warning(
          f"Workflow execution encountered model outage/transient error ({e}). Engaging graceful tool degradation.",
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

      # Intelligent heuristic fallback response with tool execution
      fallback_intent = classify_query_heuristically(user_text)
      route_taken = fallback_intent["category"]
      reasoning_text = fallback_intent["reasoning"]

      if route_taken == "shipping":
        target_node = "shipping_faq_agent"
        outcome_action = "answered_faq_fallback"
        final_response, tools_used = generate_fallback_shipping_response(user_text)
        nodes_visited = ["process_user_query", "query_classifier", "route_query", "shipping_faq_agent"]
      else:
        target_node = "decline_unrelated_query"
        outcome_action = "declined_unrelated_fallback"
        final_response = generate_fallback_decline_response()
        nodes_visited = ["process_user_query", "query_classifier", "route_query", "decline_unrelated_query"]

      memory.add_agent_turn(
          response=final_response,
          route=route_taken,
          tools_used=[t.get("name", "tool") for t in tools_used],
      )
      background_tasks.add_task(MemoryStore.save_turn_async, session_id, memory.turns[-1])
      memory_summary = memory.get_context_summary()

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
          tools_used=tools_used,
          memory_context=memory_summary,
          requires_human_confirmation=requires_hitl,
          hitl_ticket=hitl_ticket_data,
          intent_vs_outcome=intent_outcome,
      )


if __name__ == "__main__":
  import uvicorn

  port = int(os.environ.get("PORT", 8080))
  uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
