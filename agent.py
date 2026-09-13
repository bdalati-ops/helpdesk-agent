"""ADK 2.0 Customer Support Graph Workflow Agent for a Shipping Company.

Features:
  - Tool & Interface Design: Callable domain tools for tracking, rate calculation, policies, and RMA generation.
  - Context & Memory: Multi-turn session memory, entity extraction (SW-..., zips, weights), and context resolution.
  - Orchestration & Logic: Intelligent intent classification, dynamic DAG routing, and graceful degradation.
"""

import os
import sys
import time
from typing import Any, Dict, List, Literal, Optional
from google.adk import Agent, Context, Event, Workflow
from google.adk.apps import App
from google.adk.workflow import START
from pydantic import BaseModel, Field

# Ensure local modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
  sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

try:
  from customer_support_agent.observability import PIIRedactor, get_logger, get_tracer
  from customer_support_agent.tools import (
      track_package,
      calculate_shipping_rate,
      query_delivery_policies,
      create_return_request,
  )
  from customer_support_agent.memory import MemoryStore, SessionMemory
except ImportError:
  from observability import PIIRedactor, get_logger, get_tracer
  from tools import (
      track_package,
      calculate_shipping_rate,
      query_delivery_policies,
      create_return_request,
  )
  from memory import MemoryStore, SessionMemory

logger = get_logger("swiftship.agent.workflow")
tracer = get_tracer("swiftship.agent.workflow")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class QueryClassification(BaseModel):
  """Structured classification result for customer queries with intent and confidence scoring."""

  category: Literal["shipping", "unrelated"] = Field(
      description=(
          "Classification category: 'shipping' for queries concerning shipping"
          " rates, package tracking, delivery status, delivery timelines, or"
          " returns/exchanges; 'unrelated' for queries outside of shipping."
      )
  )
  sub_intent: Optional[str] = Field(
      default="general",
      description="Specific sub-intent: tracking, rates, delivery_policy, returns, general_faq, or unrelated.",
  )
  confidence: float = Field(
      default=1.0,
      description="Confidence score for this classification between 0.0 and 1.0.",
  )
  reasoning: str = Field(
      default="",
      description="Brief explanation justifying the chosen classification.",
  )


def process_user_query(node_input: str) -> Event:
  """
  Ingests the incoming user query, enriches it with multi-turn session memory,
  extracts logistics entities (tracking IDs, ZIP codes, weights), and sanitizes PII.
  """
  with tracer.start_as_current_span("workflow.process_user_query") as span:
    redacted_query = PIIRedactor.redact(node_input)
    span.set_attribute("workflow.node", "process_user_query")
    span.set_attribute("customer.query_redacted", redacted_query)

    # Resolve or create session memory context
    session_id = os.getenv("CURRENT_SESSION_ID", "default_session")
    memory = MemoryStore.get_or_create(session_id)
    extracted_entities = memory.add_user_turn(node_input)
    enriched_query = memory.enrich_query_with_context(node_input)
    context_summary = memory.get_context_summary()

    span.set_attribute("memory.turn_count", context_summary["turn_count"])
    if context_summary.get("active_tracking_number"):
      span.set_attribute("memory.active_tracking_id", context_summary["active_tracking_number"])

    logger.info(
        "Processing user query with multi-turn memory enrichment",
        extra={
            "event_type": "node_execution",
            "structured_context": {
                "node": "process_user_query",
                "session_id": session_id,
                "query_length": len(node_input),
                "query_redacted": redacted_query[:120],
                "extracted_entities": extracted_entities,
                "memory_turn_count": context_summary["turn_count"],
            },
        },
    )

    return Event(
        output=enriched_query,
        state={
            "user_query": enriched_query,
            "raw_user_query": node_input,
            "session_id": session_id,
            "entities": extracted_entities,
            "context_summary": context_summary,
            "workflow_start_time": time.time(),
        },
    )


classifier_agent = Agent(
    name="query_classifier",
    model=DEFAULT_MODEL,
    description="Classifies customer inquiries into 'shipping' or 'unrelated' with intent categorization.",
    instruction="""\
You are an intent classification specialist for SwiftShip, a commercial shipping and logistics company.

Analyze the incoming customer query:
"{user_query}"

Classify the query into exactly one of two primary categories:
1. "shipping": The query is related to shipping or logistics services, including:
   - Shipping rates, pricing quotes, weight/dimensional surcharges, and speed options.
   - Package tracking, tracking numbers, in-transit updates, delivery dates, and status checks.
   - Delivery issues, signature requirements, address changes, access point holds, or missed attempts.
   - Return policies, generating return labels, drop-off locations, return pickups, and exchange transit.
   Set 'sub_intent' to one of: 'tracking', 'rates', 'delivery_policy', 'returns', or 'general_faq'.

2. "unrelated": The query is completely unrelated to shipping or parcel delivery. Examples:
   - General knowledge, coding, weather, math problems, jokes, poems, or cooking recipes.
   - Non-shipping product questions, casual chit-chat, or unrelated technical support.
   Set 'sub_intent' to 'unrelated'.

Output your classification matching the required JSON schema with 'category', 'sub_intent', 'confidence', and 'reasoning'.
""",
    output_schema=QueryClassification,
    output_key="classification",
)


def route_query(node_input: Any):
  """Evaluates the classification output and yields the appropriate graph route."""
  with tracer.start_as_current_span("workflow.route_query") as span:
    if isinstance(node_input, QueryClassification):
      category = node_input.category
      reasoning = node_input.reasoning
      sub_intent = node_input.sub_intent or "general"
    elif isinstance(node_input, dict):
      category = node_input.get("category", "shipping")
      reasoning = node_input.get("reasoning", "")
      sub_intent = node_input.get("sub_intent", "general")
    else:
      category = getattr(node_input, "category", "shipping")
      reasoning = getattr(node_input, "reasoning", "")
      sub_intent = getattr(node_input, "sub_intent", "general")

    span.set_attribute("workflow.node", "route_query")
    span.set_attribute("intent.category", category)
    span.set_attribute("intent.sub_intent", sub_intent)
    span.set_attribute("intent.reasoning", PIIRedactor.redact(reasoning))

    logger.info(
        f"Evaluating intent route: '{category}' (sub_intent: '{sub_intent}')",
        extra={
            "event_type": "routing_decision",
            "structured_context": {
                "node": "route_query",
                "category": category,
                "sub_intent": sub_intent,
                "reasoning": PIIRedactor.redact(reasoning),
            },
        },
    )

    yield Event(route=category)


shipping_faq_agent = Agent(
    name="shipping_faq_agent",
    model=DEFAULT_MODEL,
    description=(
        "Customer support representative answering shipping inquiries and executing tools"
        " for rates, tracking, delivery policies, and return label generation."
    ),
    instruction="""\
You are a friendly, knowledgeable, and empathetic customer support representative for SwiftShip, a premier shipping and logistics company.

Customer Inquiry & Session Context:
"{user_query}"

You have access to official SwiftShip tools to retrieve live data and assist customers:
1. `track_package(tracking_number)`: Call to look up real-time delivery status, location, history, and delivery ETA for any tracking number (e.g. SW-123456789).
2. `calculate_shipping_rate(origin_zip, destination_zip, weight_lbs, service_tier)`: Call to calculate exact rates and transit times for Ground, Express, or Overnight.
3. `query_delivery_policies(topic)`: Call to look up official rules regarding signatures, Access Point holds, weekend deliveries, or oversized cargo.
4. `create_return_request(tracking_number, order_id, reason)`: Call to generate a Return Merchandise Authorization (RMA) and prepaid shipping label / QR code.

Multi-Turn Context & Memory Guidelines:
- Pay close attention to conversational context. If the customer previously mentioned a tracking number or asks "where is it?" / "when will it arrive?", resolve the active tracking number from the session context and call `track_package` without asking the customer to re-enter it.
- If the customer provided postal codes or weights earlier in the conversation, use them when calculating rates.
- If a tool returns data, explain the results clearly in a warm, professional customer service tone using bullet points.
- If details are missing to execute a tool (e.g., neither query nor memory contains a tracking number), provide helpful general information and politely invite the customer to supply the details.
""",
    tools=[
        track_package,
        calculate_shipping_rate,
        query_delivery_policies,
        create_return_request,
    ],
)


def decline_unrelated_query():
  """Politely declines to answer non-shipping queries."""
  with tracer.start_as_current_span("workflow.decline_unrelated_query") as span:
    span.set_attribute("workflow.node", "decline_unrelated_query")
    span.set_attribute("outcome.action", "declined_unrelated")

    logger.info(
        "Routing to polite decline handler for unrelated query",
        extra={
            "event_type": "decline_unrelated",
            "structured_context": {
                "node": "decline_unrelated_query",
                "action": "polite_decline",
            },
        },
    )

    decline_message = (
        "Thank you for contacting SwiftShip Customer Support! "
        "I am specialized solely in shipping and logistics services—such as "
        "calculating shipping rates, tracking packages, providing delivery "
        "updates, and assisting with returns. "
        "I am unable to answer queries outside of shipping. "
        "If you have any questions regarding shipping or package delivery, "
        "please feel free to ask!"
    )
    yield Event(message=decline_message)
    yield Event(output=decline_message)


# Graph Workflow Definition
root_agent = Workflow(
    name="customer_support_workflow",
    description=(
        "Customer support workflow with multi-turn memory, intent classification, and tool-augmented routing."
    ),
    edges=[
        ("START", process_user_query, classifier_agent, route_query),
        (
            route_query,
            {
                "shipping": shipping_faq_agent,
                "unrelated": decline_unrelated_query,
            },
        ),
    ],
)

app = App(
    name="customer-support-agent",
    root_agent=root_agent,
)
