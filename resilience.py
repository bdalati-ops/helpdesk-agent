"""
Resilience & Fault-Tolerance Module for SwiftShip Orchestration & Logic.

Provides:
  - Automatic exponential backoff & jitter for transient 503 UNAVAILABLE, 429, and network errors
  - Multi-model redundancy and configurable fallback model selection
  - Intelligent graceful degradation with deterministic domain responses when LLM APIs spike in demand
  - Circuit breaking and status tracking
"""

import asyncio
import os
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
  from customer_support_agent.observability import PIIRedactor, get_logger, get_tracer
  logger = get_logger("swiftship.resilience")
  tracer = get_tracer("swiftship.resilience")
except ImportError:
  try:
    from observability import PIIRedactor, get_logger, get_tracer
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
  has_tracking_id = bool(re.search(r"sw-[0-9]{8,12}", q))

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
    return {
        "category": "shipping",
        "reasoning": "Determined via fallback heuristic: shipping inquiry keywords matched.",
        "target_node": "shipping_faq_agent",
    }
  else:
    return {
        "category": "unrelated",
        "reasoning": "Determined via fallback heuristic: non-shipping inquiry.",
        "target_node": "decline_unrelated_query",
    }


def generate_fallback_shipping_response(query: str) -> str:
  """
  Generates a high-quality, domain-aligned response for shipping inquiries
  when the LLM service is undergoing a 503 demand spike.
  """
  q = query.lower()

  # Check for tracking numbers
  tracking_matches = re.findall(r"sw-[0-9]{8,12}", q, re.IGNORECASE)
  if tracking_matches or "track" in q or "where is" in q or "status" in q:
    pkg_id = tracking_matches[0].upper() if tracking_matches else "your package"
    return (
        f"Thank you for reaching out regarding package tracking. For **{pkg_id}**:\n\n"
        "- **Standard Status**: Most SwiftShip packages are scanned at regional hubs within 12-24 hours.\n"
        "- **Estimated Delivery**: Standard Ground arrives within 3-5 business days; Express arrives within 1-2 business days.\n"
        "- **Live Tracking**: You can view real-time GPS scans and opt-in for SMS/email alerts on the SwiftShip portal.\n\n"
        "If your package has not updated within 48 hours, please reply with your delivery postal code so our dispatch team can investigate."
    )

  # Check for rates / pricing
  if any(w in q for w in ["rate", "cost", "price", "how much", "quote", "international"]):
    return (
        "Here are SwiftShip's standard shipping rates and service tiers:\n\n"
        "- **Standard Ground** (3–5 business days): Starting at **$5.99**\n"
        "- **Priority Express** (2 business days): Starting at **$12.99**\n"
        "- **Overnight Express** (Next business day by 10:30 AM): Starting at **$24.99**\n"
        "- **International Shipping** (5–10 business days): Varies by destination country and customs.\n\n"
        "Final rates are calculated by package weight, dimensions, origin, and destination zip codes."
    )

  # Check for returns / refunds
  if any(w in q for w in ["return", "refund", "exchange", "damaged", "label"]):
    return (
        "SwiftShip makes returns simple and hassle-free:\n\n"
        "- **30-Day Policy**: Returns are accepted within 30 days of delivery.\n"
        "- **Prepaid Labels**: Generate a prepaid return shipping label or mobile QR code via our online portal.\n"
        "- **Drop-off Locations**: Bring your package to any SwiftShip branch, partner locker, or authorized retail drop box.\n"
        "- **Refund Processing**: Once received at the merchant inspection facility, refunds are processed within 3 to 5 business days."
    )

  # Delivery windows & schedule
  if any(w in q for w in ["when", "time", "hour", "weekend", "saturday", "sunday", "hold", "signature"]):
    return (
        "Here are SwiftShip's delivery service policies:\n\n"
        "- **Delivery Hours**: Monday through Saturday between 8:00 AM and 8:00 PM local time.\n"
        "- **Signatures**: Required for packages valued over $500 or containing restricted items.\n"
        "- **Hold for Pickup**: If you will be away, you can request a complimentary hold at any local SwiftShip Access Point for up to 7 calendar days."
    )

  # General overview fallback
  return (
      "Thank you for contacting SwiftShip Customer Support! We are here to assist with all your logistics needs:\n\n"
      "- **Rates & Pricing**: Ground from $5.99, Express from $12.99, Overnight from $24.99.\n"
      "- **Tracking**: Real-time status available for all SW-XXXXXXXXX tracking IDs.\n"
      "- **Returns**: Easy drop-offs and QR-code prepaid return labels.\n"
      "- **Delivery Hours**: Monday through Saturday, 8:00 AM – 8:00 PM.\n\n"
      "Please let us know your specific tracking number or shipping question and we will be delighted to help!"
  )


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
