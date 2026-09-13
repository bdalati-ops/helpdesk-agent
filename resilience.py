"""
Resilience & Fault-Tolerance Module for SwiftShip Orchestration & Logic.

Provides:
  - Automatic exponential backoff & jitter for transient 503 UNAVAILABLE, 429, and network errors
  - Multi-model redundancy and configurable fallback model selection
  - Intelligent graceful degradation with tool execution when LLM APIs spike in demand
  - Circuit breaking and status tracking
"""

import asyncio
import json
import os
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
  from customer_support_agent.observability import PIIRedactor, get_logger, get_tracer
  from customer_support_agent.tools import (
      track_package,
      calculate_shipping_rate,
      query_delivery_policies,
      create_return_request,
  )
  logger = get_logger("swiftship.resilience")
  tracer = get_tracer("swiftship.resilience")
except ImportError:
  try:
    from observability import PIIRedactor, get_logger, get_tracer
    from tools import (
        track_package,
        calculate_shipping_rate,
        query_delivery_policies,
        create_return_request,
    )
    logger = get_logger("swiftship.resilience")
    tracer = get_tracer("swiftship.resilience")
  except ImportError:
    logger = None
    tracer = None


# Configurable model cascade
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODELS = [
    os.getenv("GEMINI_FALLBACK_MODEL_1", "gemini-2.0-flash"),
    os.getenv("GEMINI_FALLBACK_MODEL_2", "gemini-1.5-flash"),
    "gemini-2.5-pro",
]

# Patterns for retryable errors (503 UNAVAILABLE, high demand spikes, 429 quota)
RETRYABLE_PATTERNS = [
    r"503",
    r"unavailable",
    r"high demand",
    r"spikes in demand",
    r"temporary",
    r"try again later",
    r"429",
    r"resource_exhausted",
    r"quota",
    r"rate limit",
    r"deadline exceeded",
    r"connection reset",
    r"service unavailable",
]


def is_retryable_error(exc: Exception) -> bool:
  """Checks whether an exception represents a transient, retryable error."""
  msg = str(exc).lower()
  return any(re.search(p, msg) for p in RETRYABLE_PATTERNS)


async def execute_with_retry(
    func: Callable[[], Any],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    operation_name: str = "llm_operation",
) -> Any:
  """
  Executes an async callable with exponential backoff and jitter for retryable errors.
  """
  delay = initial_delay
  last_exception = None

  for attempt in range(1, max_retries + 1):
    try:
      return await func()
    except Exception as e:
      last_exception = e
      if not is_retryable_error(e) or attempt == max_retries:
        if logger:
          logger.error(
              f"[{operation_name}] Attempt {attempt}/{max_retries} failed permanently: {e}",
              extra={"event_type": "retry_failed", "structured_context": {"attempt": attempt, "error": str(e)}},
          )
        raise e

      jitter = random.uniform(0.1, 0.5)
      sleep_time = min(delay + jitter, max_delay)
      if logger:
        logger.warning(
            f"[{operation_name}] Attempt {attempt}/{max_retries} encountered transient error: {e}. Retrying in {sleep_time:.2f}s...",
            extra={"event_type": "retry_backoff", "structured_context": {"attempt": attempt, "sleep_time": sleep_time, "error": str(e)}},
        )
      await asyncio.sleep(sleep_time)
      delay *= backoff_factor

  raise last_exception


def classify_query_heuristically(query: str) -> Dict[str, str]:
  """
  Deterministic fallback classifier used when the LLM service is unavailable (503).
  """
  q = query.lower()

  # Tracking number regex (SW-123456789)
  has_tracking_id = bool(re.search(r"sw-[0-9a-z]{8,12}", q))

  shipping_terms = [
      "ship", "shipping", "parcel", "package", "track", "tracking",
      "rate", "rates", "cost", "quote", "delivery", "deliver", "delivered",
      "return", "returns", "refund", "ground", "express", "overnight",
      "next-day", "transit", "freight", "carrier", "customs", "label",
      "pickup", "box", "dimension", "weight", "address", "signature"
  ]

  unrelated_terms = [
      "weather", "temperature", "forecast", "code", "python", "javascript",
      "quicksort", "algorithm", "joke", "funny", "recipe", "cook", "bake",
      "sourdough", "capital of", "president", "movie", "song", "poem",
      "quantum", "astronomy", "football", "baseball", "world series",
      "system prompt", "ignore instructions"
  ]

  is_shipping = has_tracking_id or any(term in q for term in shipping_terms)
  is_unrelated = any(term in q for term in unrelated_terms)

  if is_shipping and not (is_unrelated and not has_tracking_id and "ship" not in q):
    sub_intent = "tracking" if has_tracking_id else "rates" if "rate" in q or "cost" in q else "returns" if "return" in q else "general"
    return {
        "category": "shipping",
        "sub_intent": sub_intent,
        "reasoning": f"Determined via fallback heuristic: {sub_intent} keywords matched.",
        "target_node": "shipping_faq_agent",
    }
  else:
    return {
        "category": "unrelated",
        "sub_intent": "unrelated",
        "reasoning": "Determined via fallback heuristic: non-shipping inquiry.",
        "target_node": "decline_unrelated_query",
    }


def generate_fallback_shipping_response(query: str) -> Tuple[str, List[Dict[str, Any]]]:
  """
  Generates a high-quality, domain-aligned response for shipping inquiries by
  directly invoking real tools when the LLM service is undergoing a demand spike.

  Returns:
    Tuple of (response_markdown_text, tools_executed_metadata)
  """
  q = query.lower()
  tools_executed = []

  # 1. Package Tracking
  tracking_matches = re.findall(r"sw-[0-9a-z]{8,12}", q, re.IGNORECASE)
  if tracking_matches or "track" in q or "where is" in q or "status" in q:
    pkg_id = tracking_matches[0].upper() if tracking_matches else "SW-123456789"
    try:
      tool_res = json.loads(track_package(pkg_id))
      tools_executed.append({"name": "track_package", "args": {"tracking_number": pkg_id}, "result": tool_res})

      if tool_res.get("found"):
        history_lines = ""
        if tool_res.get("recent_events"):
          history_lines = "\n**Recent Tracking History:**\n" + "\n".join(
              [f"- *{ev['time']}* ({ev['location']}): {ev['event']}" for ev in tool_res["recent_events"]]
          )

        return (
            f"Here is the latest live status for shipment **{tool_res['tracking_number']}**:\n\n"
            f"- **Current Status**: **{tool_res['status']}**\n"
            f"- **Service Tier**: {tool_res['service_tier']}\n"
            f"- **Route**: {tool_res['origin']} ➔ {tool_res['destination']}\n"
            f"- **Estimated Delivery**: **{tool_res['estimated_delivery']}**\n"
            f"- **Package Weight**: {tool_res['weight_lbs']} lbs\n"
            f"- **Signature Required**: {'Yes' if tool_res['signature_required'] else 'No'}\n"
            f"{history_lines}\n\n"
            "You can manage delivery preferences or sign up for SMS notifications at any time."
        ), tools_executed
    except Exception as e:
      if logger:
        logger.warning(f"Fallback track tool failed: {e}")

  # 2. Shipping Rates & Quotes
  if any(w in q for w in ["rate", "cost", "price", "how much", "quote", "international"]):
    zips = re.findall(r"\b[0-9]{5}\b", q)
    origin_zip = zips[0] if len(zips) >= 1 else "10001"
    dest_zip = zips[1] if len(zips) >= 2 else "90210"
    weights = re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:lbs?|pounds?|kg)\b", q)
    weight_val = float(weights[0]) if weights else 5.0

    try:
      tool_res = json.loads(calculate_shipping_rate(origin_zip, dest_zip, weight_val))
      tools_executed.append({"name": "calculate_shipping_rate", "args": {"origin_zip": origin_zip, "destination_zip": dest_zip, "weight_lbs": weight_val}, "result": tool_res})

      quote_lines = "\n".join([
          f"- **{q['name']}** ({q['transit_days']}): **${q['cost']:.2f}**"
          for q in tool_res["quotes"].values()
      ])

      return (
          f"Here are the calculated shipping rates for a **{weight_val} lb** package from **{origin_zip}** to **{dest_zip}**:\n\n"
          f"{quote_lines}\n\n"
          f"- **Value Protection**: {tool_res['insurance_included']}.\n"
          "- All shipments include real-time GPS tracking and weekend residential delivery at no additional charge."
      ), tools_executed
    except Exception as e:
      if logger:
        logger.warning(f"Fallback rate tool failed: {e}")

  # 3. Returns & Refunds
  if any(w in q for w in ["return", "refund", "exchange", "damaged", "label"]):
    try:
      tracking_id = tracking_matches[0].upper() if tracking_matches else None
      tool_res = json.loads(create_return_request(tracking_number=tracking_id, reason="customer_inquiry"))
      tools_executed.append({"name": "create_return_request", "args": {"tracking_number": tracking_id}, "result": tool_res})

      return (
          f"I have initialized a return authorization for you:\n\n"
          f"- **Return Authorization**: **{tool_res['rma_number']}**\n"
          f"- **Prepaid Shipping Label**: [Download Printable Label PDF]({tool_res['prepaid_label_url']})\n"
          f"- **Mobile Drop-off Code**: `{tool_res['mobile_qr_code']}`\n"
          f"- **Return Deadline**: {tool_res['return_window_expiry']} (30-day window)\n"
          f"- **Refund Timeline**: {tool_res['refund_processing_time']}\n\n"
          "You can present the mobile QR code at any SwiftShip branch, partner locker, or authorized retail drop-off."
      ), tools_executed
    except Exception as e:
      if logger:
        logger.warning(f"Fallback return tool failed: {e}")

  # 4. Delivery Policies
  try:
    tool_res = json.loads(query_delivery_policies(q))
    tools_executed.append({"name": "query_delivery_policies", "args": {"topic": q[:30]}, "result": tool_res})

    policy_name = tool_res.get("policy", "SwiftShip Operational Policy")
    rule_desc = tool_res.get("rule") or tool_res.get("standard_hours", "Monday - Saturday 8am - 8pm")

    return (
        f"**{policy_name}**:\n\n"
        f"{rule_desc}\n\n"
        "- **Standard Operating Hours**: Monday through Saturday, 8:00 AM – 8:00 PM local time.\n"
        "- **Complimentary Holds**: Available at all local SwiftShip Access Points for up to 7 calendar days."
    ), tools_executed
  except Exception:
    pass

  # General overview fallback
  return (
      "Thank you for contacting SwiftShip Customer Support! We provide comprehensive shipping solutions:\n\n"
      "- **Standard Ground** (3–5 business days): Starting at $5.99\n"
      "- **Priority Express** (2 business days): Starting at $12.99\n"
      "- **Overnight Express** (Next business day by 10:30 AM): Starting at $24.99\n"
      "- **Real-time Tracking**: Available for all `SW-XXXXXXXXX` tracking IDs\n"
      "- **Returns**: 30-day return policy with free prepaid QR code drop-off\n\n"
      "Please let us know your tracking number or postal codes and we will be delighted to assist!"
  ), tools_executed


def generate_fallback_decline_response() -> str:
  """Fallback decline response for unrelated queries."""
  return (
      "Thank you for contacting SwiftShip Customer Support! "
      "I am specialized solely in shipping and logistics services—such as "
      "calculating shipping rates, tracking packages, providing delivery "
      "updates, and assisting with returns. "
      "I am unable to answer queries outside of shipping. "
      "If you have any questions regarding shipping or package delivery, "
      "please feel free to ask!"
  )
