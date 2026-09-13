"""ADK 2.0 Customer Support Graph Workflow Agent for a Shipping Company.

Features:
  - Strategic Model Routing: Differentiates models by cognitive tier (fast triage vs. domain synthesis vs. complex reasoning).
  - Human-in-the-Loop (HITL) Hooks: Guards high-impact actions (address reroutes, high-value refunds, supervisor escalations).
  - Context & Memory: Multi-turn sliding windows, progressive summarization, and entity resolution.
  - Tool Design: Dedicated tools for tracking, rate calculation, delivery policies, and RMA generation.
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
  from customer_support_agent.hitl import HITLManager, HITLConfirmationRequest
except ImportError:
  from observability import PIIRedactor, get_logger, get_tracer
  from tools import (
      track_package,
      calculate_shipping_rate,
      query_delivery_policies,
      create_return_request,
  )
  from memory import MemoryStore, SessionMemory
  from hitl import HITLManager, HITLConfirmationRequest

logger = get_logger("swiftship.agent.workflow")
tracer = get_tracer("swiftship.agent.workflow")

# ==============================================================================
# 1. Strategic Model Routing Tiers
# ==============================================================================
# Tier 1: Fast, cost-efficient, ultra-low-latency model for triage and classification
MODEL_TIER_FAST_ROUTER = os.getenv("MODEL_TIER_FAST_ROUTER", "gemini-2.0-flash")

# Tier 2: Balanced, tool-augmented model for domain synthesis, rates, and policies
MODEL_TIER_DOMAIN_SYNTHESIS = os.getenv("MODEL_TIER_DOMAIN_SYNTHESIS", "gemini-2.5-flash")

# Tier 3: High-capacity reasoning model for complex claims, disputes, and arbitration
MODEL_TIER_COMPLEX_REASONING = os.getenv("MODEL_TIER_COMPLEX_REASONING", "gemini-2.5-pro")


def get_strategic_model(tier: str = "synthesis") -> str:
  """Strategically routes to the optimal model based on task complexity."""
  tiers = {
      "fast_triage": MODEL_TIER_FAST_ROUTER,
      "synthesis": MODEL_TIER_DOMAIN_SYNTHESIS,
      "complex_reasoning": MODEL_TIER_COMPLEX_REASONING,
  }
  return tiers.get(tier, MODEL_TIER_DOMAIN_SYNTHESIS)


# ==============================================================================
# 2. Schemas & Workflow Nodes
# ==============================================================================

class QueryClassification(BaseModel):
  """Structured classification result for customer queries with intent and confidence scoring."""

  category: Literal["shipping", "complex_claim", "unrelated"] = Field(
      description=(
          "Classification category: 'shipping' for standard shipping FAQs and lookups;"
          " 'complex_claim' for damage disputes or claims exceeding $100;"
          " 'unrelated' for non-shipping queries."
      )
  )
  sub_intent: Optional[str] = Field(
      default="general",
      description="Specific sub-intent: tracking, rates, delivery_policy, returns, claims, or unrelated.",
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
  Ingests user query, enriches it with multi-turn session memory,
  evaluates Human-in-the-Loop (HITL) confirmation hooks, and sanitizes PII.
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

    # Check Human-in-the-Loop Confirmation Hook
    hitl_ticket = HITLManager.evaluate_hitl_requirement(
        session_id=session_id,
        query=node_input,
        intent="unknown",
        entities=extracted_entities,
    )
    requires_hitl = hitl_ticket is not None
    span.set_attribute("hitl.required", requires_hitl)
    if requires_hitl:
      span.set_attribute("hitl.ticket_id", hitl_ticket.ticket_id)
      span.set_attribute("hitl.action_type", hitl_ticket.action_type)

    logger.info(
        "Processed query with strategic routing and HITL evaluation",
        extra={
            "event_type": "node_execution",
            "structured_context": {
                "node": "process_user_query",
                "session_id": session_id,
                "hitl_required": requires_hitl,
                "hitl_ticket_id": hitl_ticket.ticket_id if hitl_ticket else None,
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
            "hitl_ticket": hitl_ticket.__dict__ if hitl_ticket else None,
            "requires_hitl": requires_hitl,
            "workflow_start_time": time.time(),
        },
    )


# Fast Triage Agent (Uses Tier 1 Model: gemini-2.0-flash)
classifier_agent = Agent(
    name="query_classifier",
    model=get_strategic_model("fast_triage"),
    description="Fast intent classification specialist triaging queries using Tier 1 model.",
    instruction="""\
You are an intent classification specialist for SwiftShip.

Analyze the incoming customer query:
"{user_query}"

Classify the query into one of three categories:
1. "shipping": General shipping inquiries including rates, tracking, delivery windows, Access Point holds, and standard returns.
2. "complex_claim": Formal claims, high-value lost freight disputes, or damage compensation requests.
3. "unrelated": Non-shipping topics (general chit-chat, math, weather, coding).

Set 'sub_intent' to one of: 'tracking', 'rates', 'delivery_policy', 'returns', 'claims', or 'unrelated'.
Output JSON schema matching 'category', 'sub_intent', 'confidence', and 'reasoning'.
""",
    output_schema=QueryClassification,
    output_key="classification",
)


def route_query(node_input: Any):
  """Evaluates classification and HITL state to yield the appropriate execution route."""
  with tracer.start_as_current_span("workflow.route_query") as span:
    # Check if a pending HITL confirmation ticket was registered
    session_id = os.getenv("CURRENT_SESSION_ID", "default_session")
    requires_hitl = False
    raw_query = os.getenv("CURRENT_USER_QUERY", "")
    if raw_query:
      ticket = HITLManager.evaluate_hitl_requirement(session_id, raw_query, "shipping")
      if ticket:
        requires_hitl = True

    if requires_hitl:
      span.set_attribute("workflow.node", "route_query")
      span.set_attribute("route.target", "hitl_confirmation")
      yield Event(route="hitl_confirmation")
      return

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

    yield Event(route=category)


# Domain Specialist (Uses Tier 2 Model: gemini-2.5-flash with Tools)
shipping_faq_agent = Agent(
    name="shipping_faq_agent",
    model=get_strategic_model("synthesis"),
    description="Tool-augmented customer support agent executing rates, tracking, policies, and RMA generation.",
    instruction="""\
You are a knowledgeable and empathetic customer support representative for SwiftShip.

Customer Inquiry & Session Context:
"{user_query}"

You have access to official SwiftShip tools:
1. `track_package(tracking_number)`: Look up live tracking status, location, history, and ETA.
2. `calculate_shipping_rate(origin_zip, destination_zip, weight_lbs, service_tier)`: Calculate exact rates across tiers.
3. `query_delivery_policies(topic)`: Look up rules for signatures, holds, weekend delivery, or oversized cargo.
4. `create_return_request(tracking_number, order_id, reason)`: Generate an RMA and prepaid return label / QR code.

Multi-Turn Context Guidelines:
- If the customer asks "when will it arrive?" or mentions "it", use the active tracking number from conversation context.
- Summarize tool results clearly with professional formatting.
""",
    tools=[
        track_package,
        calculate_shipping_rate,
        query_delivery_policies,
        create_return_request,
    ],
)


# Complex Claims Agent (Uses Tier 3 Model: gemini-2.5-pro for deep reasoning)
complex_claim_agent = Agent(
    name="complex_claim_agent",
    model=get_strategic_model("complex_reasoning"),
    description="Senior claims specialist utilizing Tier 3 deep reasoning model for freight disputes.",
    instruction="""\
You are a senior logistics claims specialist for SwiftShip handling freight claims, damaged parcels, and service arbitrations.

Customer Dispute:
"{user_query}"

Guidelines:
- Acknowledge customer frustration with extreme professionalism and empathy.
- Detail the formal SwiftShip claims process under the Carmack Amendment / Commercial Carriage Terms.
- State that claims for damaged or missing freight require formal evidence (photos of parcel, commercial invoice).
- Inform the customer that their case will be reviewed by human claims adjusters.
""",
)


# Human-in-the-Loop Confirmation Node
def hitl_confirmation_node():
  """Halts automatic execution and emits a supervisor confirmation hold notice."""
  with tracer.start_as_current_span("workflow.hitl_confirmation_node") as span:
    span.set_attribute("workflow.node", "hitl_confirmation_node")
    span.set_attribute("outcome.action", "held_for_human_approval")

    session_id = os.getenv("CURRENT_SESSION_ID", "default_session")
    raw_query = os.getenv("CURRENT_USER_QUERY", "")
    ticket = HITLManager.evaluate_hitl_requirement(session_id, raw_query, "shipping")
    ticket_id = ticket.ticket_id if ticket else "HITL-PENDING"
    action_type = ticket.action_type.replace("_", " ").title() if ticket else "Sensitive Action"

    msg = (
        f"⚠️ **Action Held for Supervisor Verification**\n\n"
        f"Your request involves a restricted or high-impact operation (**{action_type}**) "
        f"that requires human supervisor authorization under SwiftShip security and anti-fraud policies.\n\n"
        f"- **Confirmation Ticket**: `{ticket_id}`\n"
        f"- **Status**: `Pending Supervisor Review`\n"
        f"- **Reason**: {ticket.summary if ticket else 'Verification required'}\n\n"
        f"A human customer service supervisor has been notified in the dispatch console. "
        f"Once reviewed, you will receive an automatic status update."
    )
    yield Event(message=msg)
    yield Event(output=msg)


def decline_unrelated_query():
  """Politely declines to answer non-shipping queries."""
  with tracer.start_as_current_span("workflow.decline_unrelated_query") as span:
    span.set_attribute("workflow.node", "decline_unrelated_query")
    span.set_attribute("outcome.action", "declined_unrelated")

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
        "Customer support workflow with strategic model routing, HITL confirmation hooks, and multi-turn memory."
    ),
    edges=[
        ("START", process_user_query, classifier_agent, route_query),
        (
            route_query,
            {
                "shipping": shipping_faq_agent,
                "complex_claim": complex_claim_agent,
                "hitl_confirmation": hitl_confirmation_node,
                "unrelated": decline_unrelated_query,
            },
        ),
    ],
)

app = App(
    name="customer-support-agent",
    root_agent=root_agent,
)
