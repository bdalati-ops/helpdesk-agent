"""
Unit tests for Resilience & Fault-Tolerance Module (Orchestration & Logic).
"""

import pytest
from resilience import (
    is_retryable_error,
    classify_query_heuristically,
    generate_fallback_shipping_response,
    generate_fallback_decline_response,
)


def test_is_retryable_error():
  err_503 = Exception("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}")
  assert is_retryable_error(err_503) is True

  err_429 = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for model")
  assert is_retryable_error(err_429) is True

  err_other = ValueError("Invalid input parameter value")
  assert is_retryable_error(err_other) is False


def test_classify_query_heuristically_shipping():
  res = classify_query_heuristically("Where is package SW-123456789?")
  assert res["category"] == "shipping"
  assert res["target_node"] == "shipping_faq_agent"

  res_rates = classify_query_heuristically("How much does overnight shipping cost?")
  assert res_rates["category"] == "shipping"
  assert res_rates["target_node"] == "shipping_faq_agent"


def test_classify_query_heuristically_unrelated():
  res = classify_query_heuristically("What is the recipe for brownies?")
  assert res["category"] == "unrelated"
  assert res["target_node"] == "decline_unrelated_query"

  res_code = classify_query_heuristically("Write a python script to sort numbers")
  assert res_code["category"] == "unrelated"
  assert res_code["target_node"] == "decline_unrelated_query"


def test_fallback_responses():
  ship_resp = generate_fallback_shipping_response("Can you track SW-987654321?")
  assert "SW-987654321" in ship_resp
  assert "SwiftShip" in ship_resp

  decline_resp = generate_fallback_decline_response()
  assert "SwiftShip Customer Support" in decline_resp
  assert "shipping and logistics" in decline_resp
