"""
Unit and integration tests for Golden Dataset Evaluation Suite.
"""

from pathlib import Path
import json
import pytest

from eval.evaluate import run_evaluation, offline_classify

DATASET_PATH = Path(__file__).resolve().parent.parent / "eval" / "golden_dataset.json"


def test_golden_dataset_structure():
  assert DATASET_PATH.exists(), f"Missing golden dataset at {DATASET_PATH}"
  with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
  assert "test_cases" in data
  assert len(data["test_cases"]) >= 20
  for tc in data["test_cases"]:
    assert "id" in tc
    assert "query" in tc
    assert tc["expected_category"] in ("shipping", "unrelated")
    assert tc["expected_node"] in ("shipping_faq_agent", "decline_unrelated_query")


def test_golden_eval_quality_gate():
  summary = run_evaluation(
      dataset_path=DATASET_PATH,
      use_live_agent=False,
      min_accuracy=0.90,
  )
  assert summary["passed_quality_gate"] is True
  assert summary["accuracy"] >= 0.90
  assert summary["f1_score"] >= 0.85
  assert summary["total_tests"] == len(summary["details"])


def test_offline_classifier_shipping_queries():
  res = offline_classify("I need to return my package SW-123456789.")
  assert res["category"] == "shipping"
  assert res["node"] == "shipping_faq_agent"


def test_offline_classifier_unrelated_queries():
  res = offline_classify("What is the recipe for chicken parmesan?")
  assert res["category"] == "unrelated"
  assert res["node"] == "decline_unrelated_query"
