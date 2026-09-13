import os
import sys
from pydantic import ValidationError

# Ensure parent directory is on sys.path for standalone runs
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from customer_support_agent.agent import (
    QueryClassification,
    classifier_agent,
    decline_unrelated_query,
    process_user_query,
    root_agent,
    route_query,
    shipping_faq_agent,
)


def test_query_classification_valid():
  """Tests that QueryClassification accepts valid shipping and unrelated categories."""
  shipping_res = QueryClassification(
      category="shipping", reasoning="User asked about rates."
  )
  assert shipping_res.category == "shipping"

  unrelated_res = QueryClassification(
      category="unrelated", reasoning="User asked for a chocolate cake recipe."
  )
  assert unrelated_res.category == "unrelated"


def test_query_classification_invalid():
  """Tests that QueryClassification rejects invalid categories."""
  try:
    QueryClassification(category="billing", reasoning="Invalid category.")
    assert False, "Expected ValidationError was not raised"
  except ValidationError:
    pass


def test_process_user_query():
  """Tests that process_user_query properly extracts query and sets session state."""
  event = process_user_query("Where is my package SW-999?")
  assert event.output == "Where is my package SW-999?"
  assert event.actions.state_delta["user_query"] == "Where is my package SW-999?"


def test_route_query_shipping():
  """Tests that route_query yields a 'shipping' route for shipping classification."""
  classification = QueryClassification(
      category="shipping", reasoning="Tracking query"
  )
  events = list(route_query(classification))
  assert len(events) == 1
  assert events[0].actions.route == "shipping"


def test_route_query_unrelated():
  """Tests that route_query yields an 'unrelated' route for unrelated classification."""
  classification = QueryClassification(
      category="unrelated", reasoning="Weather inquiry"
  )
  events = list(route_query(classification))
  assert len(events) == 1
  assert events[0].actions.route == "unrelated"


def test_decline_unrelated_query():
  """Tests that decline_unrelated_query produces polite customer service text."""
  events = list(decline_unrelated_query())
  assert len(events) == 2

  message_event = events[0]
  output_event = events[1]

  # Verify UI message content
  assert message_event.content is not None
  text = "".join(part.text or "" for part in message_event.content.parts)
  assert "SwiftShip Customer Support" in text
  assert "shipping and logistics services" in text

  # Verify workflow output
  assert "SwiftShip Customer Support" in output_event.output


def test_workflow_structure():
  """Validates that the Workflow graph contains the expected nodes and routing targets."""
  assert root_agent.name == "customer_support_workflow"
  assert len(root_agent.edges) == 2

  # Edge 1: ("START", process_user_query, classifier_agent, route_query)
  first_edge_chain = root_agent.edges[0]
  assert first_edge_chain[0] == "START"
  assert first_edge_chain[1] == process_user_query
  assert first_edge_chain[2] == classifier_agent
  assert first_edge_chain[3] == route_query

  # Edge 2: (route_query, {"shipping": shipping_faq_agent, "unrelated": decline_unrelated_query, ...})
  second_edge = root_agent.edges[1]
  assert second_edge[0] == route_query
  routes_dict = second_edge[1]
  assert routes_dict["shipping"] == shipping_faq_agent
  assert routes_dict["unrelated"] == decline_unrelated_query


if __name__ == "__main__":
  test_query_classification_valid()
  test_query_classification_invalid()
  test_process_user_query()
  test_route_query_shipping()
  test_route_query_unrelated()
  test_decline_unrelated_query()
  test_workflow_structure()
  print("All unit tests passed successfully!")
