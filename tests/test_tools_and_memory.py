"""
Unit tests for Tool Design, Context & Memory, and Orchestration.
"""

import json
import pytest
from tools import (
    track_package,
    calculate_shipping_rate,
    query_delivery_policies,
    create_return_request,
    get_available_tools_metadata,
)
from memory import MemoryStore, SessionMemory


# ==============================================================================
# 1. Tool Design Tests
# ==============================================================================

def test_track_package_known_shipment():
  raw = track_package("SW-123456789")
  res = json.loads(raw)
  assert res["found"] is True
  assert res["tracking_number"] == "SW-123456789"
  assert "status" in res
  assert "estimated_delivery" in res
  assert len(res["recent_events"]) > 0


def test_track_package_dynamic_valid_format():
  raw = track_package("SW-998877665")
  res = json.loads(raw)
  assert res["found"] is True
  assert res["tracking_number"] == "SW-998877665"
  assert res["status"] in ["In Transit", "Arrived at Sort Facility", "Out for Delivery", "Delivered to Access Point"]


def test_calculate_shipping_rate():
  raw = calculate_shipping_rate(origin_zip="10001", destination_zip="90210", weight_lbs=5.0)
  res = json.loads(raw)
  assert "quotes" in res
  assert "ground" in res["quotes"]
  assert "express" in res["quotes"]
  assert "overnight" in res["quotes"]
  assert res["quotes"]["ground"]["cost"] > 0
  assert res["quotes"]["express"]["cost"] > res["quotes"]["ground"]["cost"]


def test_query_delivery_policies():
  raw = query_delivery_policies("signature requirements for expensive parcels")
  res = json.loads(raw)
  assert "signature" in res["policy"].lower()
  assert "$500" in res["rule"]


def test_create_return_request():
  raw = create_return_request(tracking_number="SW-123456789", reason="defective")
  res = json.loads(raw)
  assert "rma_number" in res
  assert res["rma_number"].startswith("RMA-")
  assert res["status"] == "Authorized"
  assert "prepaid_label_url" in res


def test_tools_metadata_catalog():
  metadata = get_available_tools_metadata()
  assert len(metadata) == 4
  names = [m["name"] for m in metadata]
  assert "track_package" in names
  assert "calculate_shipping_rate" in names


# ==============================================================================
# 2. Context & Memory Tests
# ==============================================================================

def test_memory_entity_extraction():
  memory = SessionMemory("test-sess-1")
  entities = memory.add_user_turn("My tracking number is SW-123456789, shipping from 10001 to 90210 with weight 8.5 lbs.")

  assert "SW-123456789" in memory.tracking_numbers
  assert memory.active_tracking_number == "SW-123456789"
  assert "10001" in memory.zip_codes
  assert "90210" in memory.zip_codes
  assert 8.5 in memory.weights


def test_cross_turn_context_enrichment():
  memory = SessionMemory("test-sess-2")
  memory.add_user_turn("Can you track my parcel SW-554433221?")
  memory.add_agent_turn("Your parcel is in transit.")

  # Follow-up turn using implicit pronoun "it"
  follow_up = "When will it arrive?"
  enriched = memory.enrich_query_with_context(follow_up)

  assert "SW-554433221" in enriched
  assert "Conversation Context" in enriched


def test_memory_store_lifecycle():
  sess_id = "mem-store-test-99"
  mem = MemoryStore.get_or_create(sess_id)
  mem.add_user_turn("Where is package SW-123?")
  assert MemoryStore.get(sess_id) is not None

  summary = mem.get_context_summary()
  assert summary["session_id"] == sess_id
  assert summary["turn_count"] == 1

  MemoryStore.clear(sess_id)
  assert MemoryStore.get(sess_id) is None
