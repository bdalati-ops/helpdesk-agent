import asyncio
import os
import sys
import time
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

# Ensure customer_support_agent package can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
  sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

# Load environment variables from .env
load_dotenv(os.path.join(current_dir, ".env"))
load_dotenv()

from customer_support_agent.observability import (
    PIIRedactor,
    get_logger,
    get_tracer,
    setup_observability,
    track_intent_vs_outcome,
)

tracer = setup_observability(service_name="swiftship-cli-runner")
logger = get_logger("swiftship.cli")

from customer_support_agent.agent import root_agent


async def run_query(runner: InMemoryRunner, session_id: str, query: str):
  """Runs a single query through the customer support workflow with tracing & observability."""
  clean_query = PIIRedactor.redact(query)
  start_time = time.perf_counter()

  with tracer.start_as_current_span("cli.query_turn") as span:
    span_ctx = span.get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else "n/a"
    span.set_attribute("customer.query_redacted", clean_query)
    span.set_attribute("session.id", session_id)

    logger.info(
        f"Executing CLI query turn [Trace: {trace_id}]",
        extra={
            "event_type": "cli_turn_start",
            "structured_context": {
                "session_id": session_id,
                "query_redacted": clean_query[:100],
            },
        },
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=query)],
    )

    agent_response_parts = []
    nodes_visited = []
    route_taken = None
    reasoning_text = None

    async for event in runner.run_async(
        user_id="user_1",
        session_id=session_id,
        new_message=message,
    ):
      if event.node_info and event.node_info.path:
        node_name = event.node_info.path.split("/")[-1].split("@")[0]
        if node_name not in nodes_visited:
          nodes_visited.append(node_name)

      if hasattr(event, "actions") and event.actions and event.actions.route:
        route_taken = event.actions.route

      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            if '"category":' in part.text and '"reasoning":' in part.text:
              try:
                import json
                parsed = json.loads(part.text)
                route_taken = parsed.get("category", route_taken)
                reasoning_text = parsed.get("reasoning")
              except Exception:
                pass
            else:
              agent_response_parts.append(part.text)

    duration_ms = (time.perf_counter() - start_time) * 1000
    target_node = (
        "shipping_faq_agent"
        if "shipping_faq_agent" in nodes_visited
        else "decline_unrelated_query"
    )
    outcome_action = (
        "answered_faq" if target_node == "shipping_faq_agent" else "declined_unrelated"
    )

    track_intent_vs_outcome(
        session_id=session_id,
        user_id="user_1",
        customer_query=query,
        intent_category=route_taken or "unknown",
        intent_reasoning=reasoning_text or "CLI query execution",
        target_node=target_node,
        outcome_action=outcome_action,
        nodes_visited=nodes_visited,
        latency_ms=duration_ms,
        response_summary="".join(agent_response_parts)[:150],
    )

    final_text = "".join(agent_response_parts)
    print(f"\n[Customer]: {clean_query}")
    if final_text:
      print(f"[SwiftShip Agent] (Trace: {trace_id}, {round(duration_ms, 1)}ms):\n{final_text}\n")
    else:
      print(f"[SwiftShip Agent] (Trace: {trace_id}): (No message output)\n")


async def main():
  app_name = "customer_support_app"
  runner = InMemoryRunner(agent=root_agent, app_name=app_name)
  session = await runner.session_service.create_session(
      app_name=app_name, user_id="user_1"
  )

  if len(sys.argv) > 1:
    # Single-query execution from command line arguments
    user_query = " ".join(sys.argv[1:])
    await run_query(runner, session.id, user_query)
  else:
    # Interactive REPL mode
    print("=" * 60)
    print("SwiftShip Customer Support Assistant (ADK 2.0)")
    print("Ask a shipping question or type 'exit' / 'quit' to end.")
    print("=" * 60)

    # If running non-interactively or stdin is closed, run sample queries
    if not sys.stdin.isatty():
      sample_queries = [
          "What are your standard and overnight shipping rates?",
          "How can I track package SW-987654321?",
          "How do I return a damaged item?",
          "Can you help me solve this algebra equation: 3x + 5 = 20?",
      ]
      for query in sample_queries:
        await run_query(runner, session.id, query)
      return

    while True:
      try:
        user_input = input("Customer: ").strip()
      except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

      if not user_input:
        continue
      if user_input.lower() in ("exit", "quit", "q"):
        print("Goodbye!")
        break

      await run_query(runner, session.id, user_input)


if __name__ == "__main__":
  asyncio.run(main())
