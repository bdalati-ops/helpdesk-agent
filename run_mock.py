"""Demonstration runner for customer_support_agent workflow using a mock LLM.

This script demonstrates both execution paths of the ADK 2.0 Graph Workflow:
1. Shipping query -> routes to shipping_faq_agent.
2. Unrelated query -> routes to decline_unrelated_query.
"""

import asyncio
import json
import os
import sys

# Ensure package importability
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from google.adk import Agent, Workflow
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START
from google.genai import types

from customer_support_agent.agent import (
    QueryClassification,
    decline_unrelated_query,
    process_user_query,
    route_query,
)


class MockLlm(BaseLlm):
  """Mock LLM delivering predefined responses for workflow testing."""

  def __init__(self, responses: list[str]):
    super().__init__(model="mock-gemini")
    self._responses = list(responses)

  async def generate_content_async(self, llm_request, stream=False):
    resp = (
        self._responses.pop(0) if self._responses else "Mock response fallback"
    )
    yield LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=resp)],
        )
    )


def create_demo_workflow(classifier_responses: list[str], faq_responses: list[str]) -> Workflow:
  """Builds a test workflow identical to the production graph with MockLlm backends."""
  mock_classifier = Agent(
      name="query_classifier",
      model=MockLlm(classifier_responses),
      instruction="Classify query",
      output_schema=QueryClassification,
      output_key="classification",
  )

  mock_faq_agent = Agent(
      name="shipping_faq_agent",
      model=MockLlm(faq_responses),
      instruction="Answer shipping FAQ",
  )

  return Workflow(
      name="customer_support_demo_workflow",
      edges=[
          (START, process_user_query, mock_classifier, route_query),
          (
              route_query,
              {
                  "shipping": mock_faq_agent,
                  "unrelated": decline_unrelated_query,
              },
          ),
      ],
  )


async def execute_demo_turn(workflow: Workflow, query: str, label: str):
  print(f"\n{'='*70}")
  print(f"Scenario: {label}")
  print(f"Customer Input: \"{query}\"")
  print(f"{'='*70}")

  runner = InMemoryRunner(agent=workflow, app_name="demo_app")
  session = await runner.session_service.create_session(
      app_name="demo_app", user_id="demo_customer"
  )

  msg = types.Content(role="user", parts=[types.Part.from_text(text=query)])

  executed_nodes = []
  async for event in runner.run_async(
      user_id="demo_customer", session_id=session.id, new_message=msg
  ):
    # Track executed nodes
    if event.node_info and event.node_info.path:
      node = event.node_info.path.split("/")[-1].split("@")[0]
      if node not in executed_nodes:
        executed_nodes.append(node)

    # Print message contents
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f"\n[{event.author or 'Agent'} Output]:\n{part.text}")

  print(f"\nExecution Path: {' -> '.join(executed_nodes)}")


async def main():
  print("ADK 2.0 Graph Workflow: Customer Support Agent Demonstration")

  # 1. Test Shipping Query
  shipping_query = "What are your standard and overnight shipping rates to New York?"
  shipping_wf = create_demo_workflow(
      classifier_responses=[
          json.dumps({
              "category": "shipping",
              "reasoning": "User is inquiring about shipping rates and delivery tiers.",
          })
      ],
      faq_responses=[
          "Hello! For deliveries to New York, SwiftShip offers:\n"
          "• Standard Ground: 3-5 business days starting at $5.99.\n"
          "• Priority Express: 2 business days starting at $12.99.\n"
          "• Overnight Express: Next business day guaranteed by 10:30 AM starting at $24.99.\n\n"
          "Please let me know if you need an exact rate quote based on package weight and dimensions!"
      ],
  )
  await execute_demo_turn(shipping_wf, shipping_query, "SHIPPING QUERY (Routes to shipping_faq_agent)")

  # 2. Test Unrelated Query
  unrelated_query = "Can you help me solve this algebra problem: 5x + 10 = 35?"
  unrelated_wf = create_demo_workflow(
      classifier_responses=[
          json.dumps({
              "category": "unrelated",
              "reasoning": "User is asking for assistance with high school math.",
          })
      ],
      faq_responses=[],
  )
  await execute_demo_turn(unrelated_wf, unrelated_query, "UNRELATED QUERY (Routes to decline_unrelated_query)")


if __name__ == "__main__":
  asyncio.run(main())
