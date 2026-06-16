"""Unit tests for src/eval.py.

Tests run on CPU with no ultralytics dependency — val() is never called.
We test: yaml generation, metric parsing (both code paths), artifact
structure, and provenance linking.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.eval as eval_module
from src.eval import (
    COCO_CLASSES,
    _make_coco_yaml,
    parse_metrics,
    save_eval_artifact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeValResult:
    """Stand-in for an ultralytics validator result object."""

    def __init__(self, via_results_dict: bool = True):
        if via_results_dict:
            self.results_dict = {
                "metrics/mAP50-95(B)": 0.456,
                "metrics/mAP50(B)": 0.612,
                "metrics/precision(B)": 0.731,
                "metrics/recall(B)": 0.580,
            }
            self.box = SimpleNamespace()  # no box attrs — must fall through to results_dict
        else:
            self.results_dict = {}
            self.box = SimpleNamespace(map=0.456, map50=0.612, mp=0.731, mr=0.580)


def _metrics() -> dict:
    return {"mAP50_95": 0.456, "mAP50": 0.612, "precision": 0.731, "recall": 0.580}


# ---------------------------------------------------------------------------
# _make_coco_yaml
# ---------------------------------------------------------------------------

class TestMakeCocaYaml:
    def test_path_is_absolute(self, tmp_path):
        yaml_str = _make_coco_yaml(tmp_path)
        assert str(tmp_path.resolve()) in yaml_str

    def test_val_split_present(self, tmp_path):
        yaml_str = _make_coco_yaml(tmp_path)
        assert "images/val2017" in yaml_str

    def test_nc_is_80(self, tmp_path):
        yaml_str = _make_coco_yaml(tmp_path)
        assert "nc: 80" in yaml_str

    def test_all_80_classes_present(self, tmp_path):
        yaml_str = _make_coco_yaml(tmp_path)
        for name in COCO_CLASSES:
            assert name in yaml_str, f"Class name missing from yaml: {name}"

    def test_class_count(self):
        assert len(COCO_CLASSES) == 80

    def test_first_and_last_class(self):
        assert COCO_CLASSES[0] == "person"
        assert COCO_CLASSES[79] == "toothbrush"


# ---------------------------------------------------------------------------
# parse_metrics
# ---------------------------------------------------------------------------

class TestParseMetrics:
    def test_reads_from_results_dict(self):
        m = parse_metrics(_FakeValResult(via_results_dict=True))
        assert m["mAP50_95"] == pytest.approx(0.456)
        assert m["mAP50"]    == pytest.approx(0.612)
        assert m["precision"] == pytest.approx(0.731)
        assert m["recall"]   == pytest.approx(0.580)

    def test_falls_back_to_box_attributes(self):
        m = parse_metrics(_FakeValResult(via_results_dict=False))
        assert m["mAP50_95"] == pytest.approx(0.456)
        assert m["mAP50"]    == pytest.approx(0.612)

    def test_required_keys_always_present(self):
        m = parse_metrics(_FakeValResult())
        assert set(m.keys()) == {"mAP50_95", "mAP50", "precision", "recall"}

    def test_all_values_are_floats(self):
        m = parse_metrics(_FakeValResult())
        for k, v in m.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"

    def test_completely_empty_result_returns_none_not_crash(self):
        empty = SimpleNamespace(results_dict={})
        m = parse_metrics(empty)
        for v in m.values():
            assert v is None

    def test_results_dict_takes_priority_over_box(self):
        # results_dict says 0.9, box says 0.1 — results_dict wins
        result = SimpleNamespace(
            results_dict={"metrics/mAP50-95(B)": 0.9},
            box=SimpleNamespace(map=0.1),
        )
        m = parse_metrics(result)
        assert m["mAP50_95"] == pytest.approx(0.9)

    def test_mAP50_95_gte_zero_and_lte_one(self):
        m = parse_metrics(_FakeValResult())
        assert 0.0 <= m["mAP50_95"] <= 1.0


# ---------------------------------------------------------------------------
# save_eval_artifact
# ---------------------------------------------------------------------------

class TestSaveEvalArtifact:
    def test_creates_json_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(eval_module, "RESULTS_DIR", tmp_path)
        out = save_eval_artifact(_metrics(), "test_eval")
        assert out.exists()
        assert out.suffix == ".json"

    def test_artifact_name_becomes_filename(self, tmp_path, monkeypatch):
        monkeypatch.setattr(eval_module, "RESULTS_DIR", tmp_path)
        out = save_eval_artifact(_metrics(), "day3_mAP_fp32")
        assert out.name == "day3_mAP_fp32.json"

    def test_metrics_block_present_and_correct(self, tmp_path, monkeypatch):
        monkeypatch.setattr(eval_module, "RESULTS_DIR", tmp_path)
        out = save_eval_artifact(_metrics(), "test_keys")
        with open(out) as f:
            artifact = json.load(f)
        assert "metrics" in artifact
        assert artifact["metrics"]["mAP50_95"] == pytest.approx(0.456)
        assert artifact["metrics"]["mAP50"]    == pytest.approx(0.612)

    def test_metadata_merged_at_top_level(self, tmp_path, monkeypatch):
        monkeypatch.setattr(eval_module, "RESULTS_DIR", tmp_path)
        meta = {"model": "yolov8s", "precision": "fp32", "dataset": "coco_val2017"}
        out = save_eval_artifact(_metrics(), "test_meta", metadata=meta)
        with open(out) as f:
            artifact = json.load(f)
        assert artifact["model"] == "yolov8s"
        assert artifact["precision"] == "fp32"
        assert artifact["dataset"] == "coco_val2017"

    def test_env_provenance_linked_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(eval_module, "RESULTS_DIR", tmp_path)
        env_json = tmp_path / "day1_env.json"
        env_json.write_text(json.dumps({"timestamp_utc": "2026-01-01T00:00:00+00:00"}))
        out = save_eval_artifact(_metrics(), "test_prov", env_json_path=env_json)
        with open(out) as f:
            artifact = json.load(f)
        assert artifact["env_timestamp_utc"] == "2026-01-01T00:00:00+00:00"

    def test_env_timestamp_null_when_not_provided(self, tmp_path, monkeypatch):
        monkeypatch.setattr(eval_module, "RESULTS_DIR", tmp_path)
        out = save_eval_artifact(_metrics(), "test_no_prov")
        with open(out) as f:
            artifact = json.load(f)
        assert artifact["env_timestamp_utc"] is None

    def test_missing_env_file_flagged_not_silenced(self, tmp_path, monkeypatch):
        monkeypatch.setattr(eval_module, "RESULTS_DIR", tmp_path)
        missing = tmp_path / "nonexistent.json"
        out = save_eval_artifact(_metrics(), "test_missing", env_json_path=missing)
        with open(out) as f:
            artifact = json.load(f)
        assert artifact["env_timestamp_utc"] is not None
        assert "MISSING" in str(artifact["env_timestamp_utc"])

    def test_artifact_is_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(eval_module, "RESULTS_DIR", tmp_path)
        out = save_eval_artifact(_metrics(), "test_json")
        # json.load raises if invalid
        with open(out) as f:
            json.load(f)
