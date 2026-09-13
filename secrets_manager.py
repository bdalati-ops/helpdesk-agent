"""
Secure Secret Management Module for SwiftShip Customer Support Agent.

Integrates with Google Cloud Secret Manager to retrieve API keys and sensitive
credentials dynamically, eliminating reliance on plaintext .env files in
production and CI/CD environments.

Features:
  - Dynamic retrieval from Google Cloud Secret Manager
  - In-memory caching to avoid redundant GCP API calls
  - Graceful fallback to local environment variables for local testing
  - Automatic integration with PII redaction and structured logging
"""

import os
import sys
from typing import Optional

try:
  from observability import PIIRedactor, get_logger
  logger = get_logger("swiftship.secrets")
except ImportError:
  logger = None
  PIIRedactor = None

# In-memory secret cache
_SECRET_CACHE = {}


def get_secret(
    secret_id: str,
    project_id: Optional[str] = None,
    version: str = "latest",
    fallback_env_var: Optional[str] = None,
) -> Optional[str]:
  """
  Retrieves a secret from Google Cloud Secret Manager, with fallback
  to an environment variable.

  Args:
    secret_id: Name or ID of the secret in Secret Manager (e.g., 'gemini-api-key').
    project_id: GCP project ID. If None, auto-detected from environment.
    version: Secret version string (default: 'latest').
    fallback_env_var: Name of env var to check if Secret Manager is unreachable.

  Returns:
    The secret payload string, or None if not found.
  """
  cache_key = f"{project_id or 'default'}:{secret_id}:{version}"
  if cache_key in _SECRET_CACHE:
    return _SECRET_CACHE[cache_key]

  # Check if explicitly requested to use env var or running in local dev mode
  use_secret_manager = os.getenv("USE_SECRET_MANAGER", "auto").lower()
  resolved_project = (
      project_id
      or os.getenv("GOOGLE_CLOUD_PROJECT")
      or os.getenv("PROJECT_ID")
      or os.getenv("GCP_PROJECT")
  )

  secret_value = None

  # 1. Attempt retrieval from Google Cloud Secret Manager if available
  if use_secret_manager in ("true", "1", "yes", "auto") and resolved_project:
    try:
      from google.cloud import secretmanager
      client = secretmanager.SecretManagerServiceClient()
      name = f"projects/{resolved_project}/secrets/{secret_id}/versions/{version}"
      response = client.access_secret_version(request={"name": name})
      secret_value = response.payload.data.decode("UTF-8").strip()
      if logger:
        logger.info(
            f"Successfully retrieved secret '{secret_id}' (version {version}) from Secret Manager",
            extra={"event_type": "secret_accessed", "structured_context": {"secret_id": secret_id, "project": resolved_project}},
        )
    except Exception as e:
      if logger and use_secret_manager in ("true", "1"):
        logger.warning(
            f"Failed to access Secret Manager for '{secret_id}': {e}. Attempting fallback.",
            extra={"event_type": "secret_fallback", "structured_context": {"error": str(e)}},
        )

  # 2. Fallback to environment variable if Secret Manager did not resolve
  if not secret_value:
    env_keys_to_check = []
    if fallback_env_var:
      env_keys_to_check.append(fallback_env_var)
    env_keys_to_check.extend([secret_id.upper().replace("-", "_"), "GOOGLE_API_KEY", "GEMINI_API_KEY"])

    for env_key in env_keys_to_check:
      val = os.getenv(env_key)
      if val:
        secret_value = val.strip()
        if logger:
          logger.info(
              f"Resolved secret '{secret_id}' from environment variable '{env_key}'",
              extra={"event_type": "secret_resolved_env", "structured_context": {"env_var": env_key}},
          )
        break

  if secret_value:
    _SECRET_CACHE[cache_key] = secret_value

  return secret_value


def initialize_app_secrets(project_id: Optional[str] = None) -> None:
  """
  Initializes core application secrets on startup and injects them
  into the process environment for SDKs (e.g. google-adk, google-genai).
  """
  secret_id = os.getenv("GEMINI_API_KEY_SECRET_ID", "gemini-api-key")
  api_key = get_secret(
      secret_id=secret_id,
      project_id=project_id,
      fallback_env_var="GOOGLE_API_KEY",
  )

  if api_key:
    # Ensure both standard SDK variable names are populated
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key
    if logger:
      logger.info(
          "Core application credentials initialized successfully.",
          extra={"event_type": "secrets_initialized"},
      )
  else:
    if logger:
      logger.warning(
          "No Gemini API key resolved from Secret Manager or environment. Live LLM calls will fail.",
          extra={"event_type": "secrets_missing"},
      )


def clear_secret_cache() -> None:
  """Clears in-memory cached secrets (useful for rotation or testing)."""
  _SECRET_CACHE.clear()
