"""
Callable Domain Tools for SwiftShip Customer Support Agent (ADK 2.0).

Provides structured, executable tools with strict schemas and docstrings for:
  1. track_package: Real-time parcel tracking and event timeline lookup
  2. calculate_shipping_rate: Ground, Express, and Overnight rate quotes with dimensional weight
  3. query_delivery_policies: Access point holds, signature rules, and delivery windows
  4. create_return_request: Return Merchandise Authorization (RMA) and prepaid label generation
"""

import datetime
import hashlib
import json
import re
from typing import Any, Dict, List, Optional


# Mock package database for demonstration and deterministic lookups
MOCK_SHIPMENTS = {
    "SW-123456789": {
        "status": "In Transit",
        "service_tier": "Priority Express",
        "origin": "New York, NY (10001)",
        "destination": "Los Angeles, CA (90001)",
        "estimated_delivery": (datetime.datetime.utcnow() + datetime.timedelta(days=1)).strftime("%Y-%m-%d 14:00 UTC"),
        "weight_lbs": 4.5,
        "signature_required": False,
        "history": [
            {"time": "2026-09-12 08:30", "location": "New York, NY", "event": "Parcel picked up by courier"},
            {"time": "2026-09-12 18:45", "location": "Newark Hub, NJ", "event": "Departed regional sort facility"},
            {"time": "2026-09-13 06:15", "location": "Chicago Air Hub, IL", "event": "In transit to destination facility"},
        ],
    },
    "SW-987654321": {
        "status": "Delivered",
        "service_tier": "Standard Ground",
        "origin": "Atlanta, GA (30301)",
        "destination": "Miami, FL (33101)",
        "estimated_delivery": "2026-09-12 11:20 UTC (Delivered)",
        "weight_lbs": 2.0,
        "signature_required": True,
        "history": [
            {"time": "2026-09-09 10:00", "location": "Atlanta, GA", "event": "Manifest received"},
            {"time": "2026-09-10 14:00", "location": "Jacksonville, FL", "event": "In transit"},
            {"time": "2026-09-12 07:45", "location": "Miami, FL", "event": "Out for delivery"},
            {"time": "2026-09-12 11:20", "location": "Miami, FL", "event": "Delivered - Signed by Front Desk"},
        ],
    },
    "SW-554433221": {
        "status": "Out for Delivery",
        "service_tier": "Overnight Express",
        "origin": "Dallas, TX (75201)",
        "destination": "Austin, TX (78701)",
        "estimated_delivery": (datetime.datetime.utcnow() + datetime.timedelta(hours=4)).strftime("%Y-%m-%d %H:%M UTC"),
        "weight_lbs": 8.2,
        "signature_required": True,
        "history": [
            {"time": "2026-09-12 21:00", "location": "Dallas Sort Center, TX", "event": "Processed at origin hub"},
            {"time": "2026-09-13 05:00", "location": "Austin Hub, TX", "event": "Arrived at local delivery depot"},
            {"time": "2026-09-13 08:15", "location": "Austin, TX", "event": "Out for delivery on courier vehicle"},
        ],
    },
}


def track_package(tracking_number: str) -> str:
  """
  Retrieves real-time tracking status, location, history, and delivery ETA for a shipment.

  Args:
    tracking_number: The SwiftShip tracking ID (e.g., 'SW-123456789' or 10-12 alphanumeric characters).

  Returns:
    JSON string with status, origin, destination, ETA, and event history.
  """
  clean_id = tracking_number.strip().upper()

  # Check if known mock shipment
  if clean_id in MOCK_SHIPMENTS:
    data = MOCK_SHIPMENTS[clean_id]
    result = {
        "tracking_number": clean_id,
        "found": True,
        "status": data["status"],
        "service_tier": data["service_tier"],
        "origin": data["origin"],
        "destination": data["destination"],
        "estimated_delivery": data["estimated_delivery"],
        "weight_lbs": data["weight_lbs"],
        "signature_required": data["signature_required"],
        "recent_events": data["history"][-3:],
    }
    return json.dumps(result)

  # Fallback for dynamic valid-format tracking numbers
  is_valid_format = bool(re.match(r"^SW-[0-9A-Z]{8,12}$", clean_id) or len(clean_id) >= 8)
  if is_valid_format:
    seed = int(hashlib.md5(clean_id.encode()).hexdigest()[:8], 16)
    statuses = ["In Transit", "Arrived at Sort Facility", "Out for Delivery", "Delivered to Access Point"]
    status = statuses[seed % len(statuses)]
    days_ahead = (seed % 3) + 1
    eta = (datetime.datetime.utcnow() + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d by 18:00 Local")

    result = {
        "tracking_number": clean_id,
        "found": True,
        "status": status,
        "service_tier": "Standard Ground" if seed % 2 == 0 else "Priority Express",
        "origin": "Regional Distribution Hub",
        "destination": "Customer Destination Address",
        "estimated_delivery": eta,
        "weight_lbs": round(1.0 + (seed % 15) * 0.5, 1),
        "signature_required": bool(seed % 3 == 0),
        "recent_events": [
            {"time": "2026-09-12 10:00", "location": "Origin Sort Hub", "event": "Parcel scanned into network"},
            {"time": "2026-09-13 04:30", "location": "In Transit Hub", "event": f"Status: {status}"},
        ],
    }
    return json.dumps(result)

  return json.dumps({
      "tracking_number": clean_id,
      "found": False,
      "error": "Invalid tracking number format. SwiftShip tracking IDs start with 'SW-' followed by 8-12 digits.",
  })


def calculate_shipping_rate(
    origin_zip: str,
    destination_zip: str,
    weight_lbs: float,
    service_tier: Optional[str] = None,
) -> str:
  """
  Calculates instant shipping rates and delivery estimates across SwiftShip service tiers.

  Args:
    origin_zip: 5-digit US origin ZIP code (e.g. '10001').
    destination_zip: 5-digit US destination ZIP code (e.g. '90210').
    weight_lbs: Package weight in pounds (must be greater than 0, max 150 lbs).
    service_tier: Optional specific tier ('ground', 'express', 'overnight'). If None, calculates all tiers.

  Returns:
    JSON string with itemized rates, base pricing, surcharges, and estimated transit days.
  """
  weight = max(0.1, float(weight_lbs))
  tier = (service_tier or "all").lower()

  # Zone distance estimate based on zip difference
  try:
    oz = int(origin_zip[:3]) if origin_zip and origin_zip[:3].isdigit() else 100
    dz = int(destination_zip[:3]) if destination_zip and destination_zip[:3].isdigit() else 900
    zone_diff = abs(oz - dz)
    distance_tier = 1.0 + min(1.5, zone_diff / 500.0)
  except Exception:
    distance_tier = 1.2

  # Calculate tier options
  tiers = {
      "ground": {
          "name": "Standard Ground",
          "base_price": 5.99,
          "per_pound": 1.25,
          "transit_days": "3-5 business days",
          "cost": round((5.99 + (weight * 1.25 * distance_tier)), 2),
      },
      "express": {
          "name": "Priority Express",
          "base_price": 12.99,
          "per_pound": 2.10,
          "transit_days": "2 business days guaranteed",
          "cost": round((12.99 + (weight * 2.10 * distance_tier)), 2),
      },
      "overnight": {
          "name": "Overnight Express",
          "base_price": 24.99,
          "per_pound": 3.75,
          "transit_days": "Next business day by 10:30 AM",
          "cost": round((24.99 + (weight * 3.75 * distance_tier)), 2),
      },
  }

  selected_quotes = {}
  if tier in tiers:
    selected_quotes[tier] = tiers[tier]
  else:
    selected_quotes = tiers

  response = {
      "origin_zip": origin_zip or "Standard Zone",
      "destination_zip": destination_zip or "Standard Zone",
      "weight_lbs": weight,
      "quotes": selected_quotes,
      "insurance_included": "$100 standard value coverage included free of charge",
      "oversized_surcharge": 25.00 if weight > 70 else 0.00,
  }
  return json.dumps(response)


def query_delivery_policies(topic: str) -> str:
  """
  Queries official SwiftShip operational and delivery policies.

  Args:
    topic: Keyword or policy subject, such as 'signature', 'hold', 'weekend', 'oversized', or 'international'.

  Returns:
    JSON string with the policy rules, requirements, and customer instructions.
  """
  t = topic.lower()
  policies = {
      "signature": {
          "policy": "Signature Requirements",
          "rule": "Direct signature is automatically required for shipments valued over $500, hazardous materials, or age-restricted goods. Senders may also add signature service for an additional $3.50 fee.",
          "options": ["Direct Signature", "Adult Signature (21+)", "Indirect Signature"],
      },
      "hold": {
          "policy": "SwiftShip Access Point Holds",
          "rule": "Packages can be safely rerouted or held at any local SwiftShip Access Point locker or authorized partner location for up to 7 calendar days at no additional charge. Valid government ID required for pickup.",
      },
      "weekend": {
          "policy": "Weekend & Saturday Delivery",
          "rule": "Saturday residential delivery is included as standard service for Ground and Priority Express at no extra charge. Sunday delivery is available in select metropolitan areas for emergency freight and critical healthcare shipments.",
          "hours": "Monday - Saturday: 8:00 AM to 8:00 PM local time",
      },
      "oversized": {
          "policy": "Oversized & Freight Limits",
          "rule": "Individual packages may weigh up to 150 lbs (68 kg) and measure up to 108 inches in length (or 165 inches length + girth). Shipments exceeding these limits must be booked through SwiftShip Freight Services.",
      },
      "returns": {
          "policy": "Returns & Refund Transit",
          "rule": "Items can be returned within 30 days of purchase using our prepaid labels. Drop-offs are accepted at 15,000+ drop boxes and authorized retail partners. Refunds typically authorize within 3-5 business days after warehouse arrival.",
      },
  }

  for k, v in policies.items():
    if k in t:
      return json.dumps(v)

  # Default comprehensive policies
  return json.dumps({
      "policy": "General Delivery Overview",
      "standard_hours": "Monday - Saturday 8:00 AM - 8:00 PM",
      "signature_threshold": "$500",
      "access_point_hold_window": "7 calendar days",
      "max_package_weight": "150 lbs",
      "return_window": "30 days from delivery",
  })


def create_return_request(
    tracking_number: Optional[str] = None,
    order_id: Optional[str] = None,
    reason: str = "general_return",
) -> str:
  """
  Creates an official Return Merchandise Authorization (RMA) and generates return label instructions.

  Args:
    tracking_number: Optional original SwiftShip tracking number (e.g., 'SW-123456789').
    order_id: Optional merchant order reference (e.g., 'ORD-55421').
    reason: Return justification (e.g., 'damaged', 'wrong_item', 'defective', 'unwanted').

  Returns:
    JSON string with RMA number, printable label URL, QR drop-off code, and return timeline.
  """
  identifier = (tracking_number or order_id or "SW-RET").upper().replace("SW-", "").replace("ORD-", "")
  rma_id = f"RMA-{identifier[:6]}-{datetime.datetime.utcnow().strftime('%M%S')}"

  return json.dumps({
      "rma_number": rma_id,
      "status": "Authorized",
      "return_window_expiry": (datetime.datetime.utcnow() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
      "prepaid_label_url": f"https://portal.swiftship.com/labels/return/{rma_id}.pdf",
      "mobile_qr_code": f"QR-SWIFTSHIP-{rma_id}",
      "dropoff_instructions": [
          "Print the attached prepaid PDF label or show the mobile QR code at any SwiftShip Access Point or partner location.",
          "Ensure the item is packaged securely with original padding if possible.",
          "Drop off within 30 days. You will receive an SMS and email notification once scanned by courier.",
      ],
      "refund_processing_time": "3 to 5 business days upon warehouse arrival",
  })


ALL_TOOLS = [
    track_package,
    calculate_shipping_rate,
    query_delivery_policies,
    create_return_request,
]


def get_available_tools_metadata() -> List[Dict[str, Any]]:
  """Returns structured documentation of all available agent tools for API reflection."""
  metadata = []
  for t in ALL_TOOLS:
    metadata.append({
        "name": t.__name__,
        "description": (t.__doc__ or "").strip().split("\n\n")[0],
        "parameters": [
            {"name": arg, "type": "string"}
            for arg in t.__code__.co_varnames[:t.__code__.co_argcount]
        ],
    })
  return metadata
