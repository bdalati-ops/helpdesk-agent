"""
Unit tests for Secret Manager integration and fallback mechanics.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from secrets_manager import get_secret, initialize_app_secrets, clear_secret_cache


def setup_function():
  clear_secret_cache()


def test_get_secret_fallback_to_env(monkeypatch):
  monkeypatch.setenv("TEST_KEY", "env_secret_value_123")
  val = get_secret("test-key", fallback_env_var="TEST_KEY")
  assert val == "env_secret_value_123"


def test_get_secret_caching(monkeypatch):
  monkeypatch.setenv("TEST_KEY", "cached_value")
  val1 = get_secret("test-key", fallback_env_var="TEST_KEY")
  monkeypatch.setenv("TEST_KEY", "changed_value")
  # Should return cached value
  val2 = get_secret("test-key", fallback_env_var="TEST_KEY")
  assert val1 == "cached_value"
  assert val2 == "cached_value"


def test_initialize_app_secrets(monkeypatch):
  monkeypatch.setenv("GOOGLE_API_KEY", "mock_gemini_key")
  initialize_app_secrets()
  assert os.environ.get("GOOGLE_API_KEY") == "mock_gemini_key"
  assert os.environ.get("GEMINI_API_KEY") == "mock_gemini_key"
