"""ADK 2.0 Customer Support Graph Workflow Agent for a Shipping Company.

This workflow serves as a customer support representative for SwiftShip, a
shipping and logistics provider. It classifies incoming user queries into
shipping-related topics (rates, tracking, delivery, returns) versus unrelated
topics. It routes shipping queries to a dedicated Shipping FAQ agent and
unrelated queries to a node that politely declines to answer.
"""

import os
import sys
import time
from typing import Literal
from google.adk import Agent, Context, Event, Workflow
from google.adk.apps import App
from google.adk.workflow import START
from pydantic import BaseModel, Field

# Ensure observability can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
  sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

try:
  from customer_support_agent.observability import PIIRedactor, get_logger, get_tracer
except ImportError:
  from observability import PIIRedactor, get_logger, get_tracer

logger = get_logger("swiftship.agent.workflow")
tracer = get_tracer("swiftship.agent.workflow")


class QueryClassification(BaseModel):
  """Structured classification result for customer queries."""

  category: Literal["shipping", "unrelated"] = Field(
      description=(
          "Classification category: 'shipping' for queries concerning shipping"
          " rates, package tracking, delivery status, delivery timelines, or"
          " returns/exchanges; 'unrelated' for queries outside of shipping."
      )
  )
  reasoning: str = Field(
      default="",
      description="Brief explanation justifying the chosen classification.",
  )


def process_user_query(node_input: str) -> Event:
  """Extracts the incoming user query, sanitizes PII for telemetry, and records it in session state."""
  with tracer.start_as_current_span("workflow.process_user_query") as span:
    redacted_query = PIIRedactor.redact(node_input)
    span.set_attribute("workflow.node", "process_user_query")
    span.set_attribute("customer.query_redacted", redacted_query)

    logger.info(
        "Processing user query in workflow node",
        extra={
            "event_type": "node_execution",
            "structured_context": {
                "node": "process_user_query",
                "query_length": len(node_input),
                "query_redacted": redacted_query[:120],
            },
        },
    )

    return Event(
        output=node_input,
        state={
            "user_query": node_input,
            "workflow_start_time": time.time(),
        },
    )


classifier_agent = Agent(
    name="query_classifier",
    model="gemini-3.6-flash",
    description="Classifies customer inquiries into 'shipping' or 'unrelated'.",
    instruction="""\
You are an intent classification specialist for SwiftShip, a commercial shipping and logistics company.

Analyze the incoming customer query:
"{user_query}"

Classify the query into exactly one of two categories:
1. "shipping": The query is related to shipping or logistics services, including:
   - Shipping rates, pricing quotes, weight/dimensional surcharges, and speed options.
   - Package tracking, tracking numbers, in-transit updates, delivery dates, and status checks.
   - Delivery issues, signature requirements, address changes, access point holds, or missed attempts.
   - Return policies, generating return labels, drop-off locations, return pickups, and exchange transit.

2. "unrelated": The query is completely unrelated to shipping or parcel delivery. Examples:
   - General knowledge, coding, weather, math problems, jokes, poems, or cooking recipes.
   - Non-shipping product questions, casual chit-chat, or unrelated technical support.

Output your classification matching the required JSON schema with 'category' and 'reasoning'.
""",
    output_schema=QueryClassification,
    output_key="classification",
)


def route_query(node_input: QueryClassification):
  """Evaluates the classification output and yields the appropriate graph route."""
  with tracer.start_as_current_span("workflow.route_query") as span:
    span.set_attribute("workflow.node", "route_query")
    span.set_attribute("intent.category", node_input.category)
    span.set_attribute("intent.reasoning", PIIRedactor.redact(node_input.reasoning))

    logger.info(
        f"Evaluating intent route: '{node_input.category}'",
        extra={
            "event_type": "routing_decision",
            "structured_context": {
                "node": "route_query",
                "category": node_input.category,
                "reasoning": PIIRedactor.redact(node_input.reasoning),
            },
        },
    )

    yield Event(route=node_input.category)


shipping_faq_agent = Agent(
    name="shipping_faq_agent",
    model="gemini-3.6-flash",
    description=(
        "Customer support representative answering shipping FAQs on rates,"
        " tracking, delivery, and returns."
    ),
    instruction="""\
You are a friendly, courteous, and knowledgeable customer support representative for SwiftShip, a premier shipping and logistics company.

Customer Query:
"{user_query}"

Provide an accurate, clear, and empathetic answer based on SwiftShip services:

1. Shipping Rates & Service Tiers:
   - Standard Ground: 3 to 5 business days, starting at $5.99.
   - Priority Express: 2 business days, starting at $12.99.
   - Overnight Express: Next business day guaranteed by 10:30 AM, starting at $24.99.
   - International Shipping: 5 to 10 business days depending on customs and destination country.
   - Rates depend on package weight, package dimensions, origin, and destination zip codes.

2. Package Tracking & Status:
   - Tracking numbers are typically 10 to 12 alphanumeric characters (e.g., SW-123456789).
   - Common statuses: "Order Manifest Received", "In Transit", "Out for Delivery", "Delivered", and "Delivery Exception".
   - Customers can view real-time status and enable SMS/email alerts on the SwiftShip tracking portal.

3. Delivery Policies:
   - Delivery operates Monday through Saturday between 8:00 AM and 8:00 PM local time.
   - Direct signature is required for packages valued over $500 or containing restricted goods.
   - If a customer is away, packages can be safely held at any local SwiftShip Access Point for up to 7 calendar days.
   - Address adjustments can be made before the package arrives at the local delivery hub.

4. Returns & Exchanges:
   - Return shipping labels can be generated via the online portal or printed at drop-off kiosks using a mobile QR code.
   - Drop-offs are accepted at all SwiftShip branches, partner lockers, and authorized retail drop boxes.
   - Scheduled courier pickup is available for return shipments.
   - Return transit typically takes 3 to 5 business days before merchant inspection and refund authorization.

Tone & Guidelines:
- Maintain a warm, helpful, and professional customer service tone.
- Use clear bullet points or short paragraphs for readability.
- If the customer does not provide specific details (like a tracking number, package weight, or postal codes), provide general guidance and politely invite them to provide the missing details so you can assist further.
""",
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
        "Customer support workflow that classifies user inquiries and routes"
        " shipping queries to a Shipping FAQ agent or politely declines"
        " unrelated queries."
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
