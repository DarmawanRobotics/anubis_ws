#!/usr/bin/env python3

import csv
import json
import tempfile
import unittest
from pathlib import Path

from global_localization_eval import (
    EvaluationInputError,
    circular_yaw_error_deg,
    evaluate,
    load_manifest,
    load_predictions,
)


def make_episode(episode_id, scene="room_start", region="room_a"):
    return {
        "episode_id": episode_id,
        "bag_path": f"bags/{episode_id}",
        "scene": scene,
        "region": region,
        "pose_kind": "full_pose",
        "pose": {"x": 1.0, "y": 2.0, "yaw_deg": 179.0},
        "uncertainty": {"xy_m": 0.05, "yaw_deg": 1.0},
        "gt_source": {
            "type": "calibrated_fixture",
            "record_id": f"fixture:{episode_id}",
            "independent_from_localization": True,
        },
        "annotator": "tester",
        "reviewer": "reviewer",
        "unusable_intervals": [],
    }


def make_manifest(episodes):
    return {
        "schema_version": 1,
        "map_version": "map:test",
        "coordinate_frame": "map",
        "T_map_gt": [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        "time_sync": {"method": "fixture", "residual_p99_ms": 10.0},
        "episodes": episodes,
    }


def add_region_polygons(document):
    document["regions"] = [
        {
            "name": "room_a",
            "polygon": [[-5.0, -5.0], [5.0, -5.0],
                        [5.0, 5.0], [-5.0, 5.0]],
        },
        {
            "name": "room_b",
            "polygon": [[5.1, -5.0], [15.0, -5.0],
                        [15.0, 10.0], [5.1, 10.0]],
        },
    ]
    return document


class GlobalLocalizationEvalTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_manifest(self, document):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def write_predictions(self, rows):
        path = self.root / "predictions.csv"
        fields = ["episode_id", "stage", "rank", "region", "x", "y", "yaw_deg", "status"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_manifest_accepts_independent_pose_and_region_only_gt(self):
        region_only = make_episode("region")
        region_only["pose_kind"] = "region_only"
        region_only.pop("pose")
        region_only.pop("uncertainty")
        region_only["gt_source"] = {
            "type": "human_region_label",
            "record_id": "label:region",
            "independent_from_localization": False,
        }

        manifest = load_manifest(self.write_manifest(make_manifest([
            make_episode("full"), region_only
        ])))

        self.assertTrue(manifest.episodes[0].auto_gate_eligible)
        self.assertFalse(manifest.episodes[1].auto_gate_eligible)
        self.assertEqual(manifest.episodes[1].exclusion_reasons, ("region_only_gt",))

    def test_manifest_rejects_localization_as_full_pose_gt(self):
        episode = make_episode("bad")
        episode["gt_source"]["type"] = "localization"

        with self.assertRaises(EvaluationInputError):
            load_manifest(self.write_manifest(make_manifest([episode])))

    def test_uncertain_or_misaligned_gt_is_excluded_from_auto_gate(self):
        episode = make_episode("uncertain")
        episode["uncertainty"]["xy_m"] = 0.11
        document = make_manifest([episode])
        document["time_sync"]["residual_p99_ms"] = 21.0

        manifest = load_manifest(self.write_manifest(document))

        self.assertFalse(manifest.episodes[0].auto_gate_eligible)
        self.assertIn("xy_uncertainty_gt_0.10m", manifest.episodes[0].exclusion_reasons)
        self.assertIn("time_residual_gt_20ms", manifest.episodes[0].exclusion_reasons)

    def test_yaw_error_wraps_across_180_degrees(self):
        self.assertAlmostEqual(circular_yaw_error_deg(179.0, -179.0), 2.0)

    def test_polygon_regions_resolve_blank_exporter_region(self):
        manifest = load_manifest(self.write_manifest(add_region_polygons(
            make_manifest([make_episode("known")]))))
        rows = [{
            "episode_id": "known", "stage": "coarse", "rank": 1,
            "region": "", "x": 1.0, "y": 2.0,
            "yaw_deg": 179.0, "status": "",
        }, {
            "episode_id": "known", "stage": "refined", "rank": 1,
            "region": "", "x": 1.0, "y": 2.0,
            "yaw_deg": 179.0, "status": "Passed",
        }]

        predictions = load_predictions(
            self.write_predictions(rows), manifest.regions)

        self.assertEqual(predictions[0].region, "room_a")
        self.assertEqual(predictions[1].stage, "refined")
        self.assertEqual(predictions[1].region, "room_a")

    def test_polygon_regions_reject_mislabeled_and_mark_outside_candidate(self):
        manifest = load_manifest(self.write_manifest(add_region_polygons(
            make_manifest([make_episode("known")]))))
        mislabeled = [{
            "episode_id": "known", "stage": "coarse", "rank": 1,
            "region": "room_b", "x": 1.0, "y": 2.0,
            "yaw_deg": 179.0, "status": "",
        }]
        with self.assertRaises(EvaluationInputError):
            load_predictions(self.write_predictions(mislabeled), manifest.regions)

        outside = [{
            "episode_id": "known", "stage": "coarse", "rank": 1,
            "region": "", "x": 100.0, "y": 100.0,
            "yaw_deg": 179.0, "status": "",
        }]
        predictions = load_predictions(
            self.write_predictions(outside), manifest.regions)
        self.assertEqual(predictions[0].region, "__unmapped__")

    def test_evaluator_reports_recall_false_confirm_and_scene_breakdown(self):
        correct = make_episode("correct", scene="room_a_start", region="room_a")
        wrong = make_episode("wrong", scene="room_b_start", region="room_b")
        wrong["pose"] = {"x": 10.0, "y": 5.0, "yaw_deg": 0.0}
        manifest = load_manifest(self.write_manifest(make_manifest([correct, wrong])))
        rows = [
            {"episode_id": "correct", "stage": "seed", "rank": 1,
             "region": "room_a", "x": 1.0, "y": 2.0, "yaw_deg": -179.0, "status": ""},
            {"episode_id": "correct", "stage": "coarse", "rank": 2,
             "region": "room_a", "x": 1.0, "y": 2.0, "yaw_deg": -179.0, "status": ""},
            {"episode_id": "correct", "stage": "cluster", "rank": 1,
             "region": "room_a", "x": 1.0, "y": 2.0, "yaw_deg": -179.0, "status": ""},
            {"episode_id": "correct", "stage": "decision", "rank": 1,
             "region": "room_a", "x": 1.0, "y": 2.0, "yaw_deg": -179.0,
             "status": "Confirmed"},
            {"episode_id": "wrong", "stage": "decision", "rank": 1,
             "region": "room_a", "x": 1.0, "y": 2.0, "yaw_deg": 0.0,
             "status": "Confirmed"},
        ]
        predictions = load_predictions(self.write_predictions(rows))

        report = evaluate(manifest, predictions, k_values=(1, 4), yaw_tolerance_deg=3.0)

        self.assertEqual(report["metrics"]["seed_recall"]["rate"], 0.5)
        self.assertEqual(report["metrics"]["coarse_recall@1"]["rate"], 0.0)
        self.assertEqual(report["metrics"]["coarse_recall@4"]["rate"], 0.5)
        self.assertEqual(report["metrics"]["cluster_recall"]["rate"], 0.5)
        false_confirm = report["metrics"]["false_confirm_rate"]
        self.assertEqual(false_confirm["false_confirm_count"], 1)
        self.assertEqual(false_confirm["rate_per_confirmed"], 0.5)
        self.assertIn("room_a_start", report["by_scene"])
        self.assertIn("room_b_start", report["by_scene"])

    def test_stage_tolerances_refined_metrics_and_diagnostics(self):
        manifest = load_manifest(self.write_manifest(make_manifest([make_episode("ep")])))
        rows = [
            # Level 0 region center 1.2 m / 30 deg off GT: inside the coarse
            # tolerance (1.5 m / 45 deg), outside the main pose tolerance.
            {"episode_id": "ep", "stage": "coarse", "rank": 1,
             "region": "room_a", "x": 2.2, "y": 2.0, "yaw_deg": 149.0, "status": ""},
            # refined rank 1 is far off; rank 2 matches the pose but was
            # filtered, so refined_recall hits while refined_passed misses.
            {"episode_id": "ep", "stage": "refined", "rank": 1,
             "region": "room_a", "x": 5.0, "y": 2.0, "yaw_deg": 0.0,
             "status": "Rejected"},
            {"episode_id": "ep", "stage": "refined", "rank": 2,
             "region": "room_a", "x": 1.1, "y": 2.0, "yaw_deg": -179.0,
             "status": "Rejected"},
            # The same 1.2 m offset at cluster stage must NOT count: cluster
            # rows are final poses and keep the strict tolerance.
            {"episode_id": "ep", "stage": "cluster", "rank": 1,
             "region": "room_a", "x": 2.2, "y": 2.0, "yaw_deg": 149.0, "status": ""},
        ]
        predictions = load_predictions(self.write_predictions(rows))

        report = evaluate(manifest, predictions, k_values=(4,))

        metrics = report["metrics"]
        self.assertEqual(metrics["coarse_recall@4"]["rate"], 1.0)
        self.assertEqual(metrics["refined_recall"]["rate"], 1.0)
        self.assertEqual(metrics["refined_passed_recall"]["rate"], 0.0)
        self.assertEqual(metrics["cluster_recall"]["rate"], 0.0)
        self.assertEqual(report["tolerances"]["coarse_xy_m"], 1.5)
        self.assertEqual(report["tolerances"]["coarse_yaw_deg"], 45.0)

        diagnostics = report["episode_diagnostics"]["ep"]
        self.assertEqual(diagnostics["coarse"]["pose_matched_rank"], 1)
        self.assertEqual(diagnostics["refined"]["pose_matched_rank"], 2)
        self.assertEqual(diagnostics["refined"]["row_count"], 2)
        self.assertAlmostEqual(
            diagnostics["refined"]["min_xy_error_m"], 0.1, places=3)
        self.assertEqual(diagnostics["refined"]["min_xy_rank"], 2)
        self.assertEqual(diagnostics["refined"]["statuses"], ["Rejected"])
        self.assertIsNone(diagnostics["cluster"]["pose_matched_rank"])
        self.assertEqual(diagnostics["seed"]["row_count"], 0)
        self.assertIsNone(diagnostics["seed"]["pose_matched_rank"])

    def test_unknown_episode_and_duplicate_decision_are_rejected(self):
        manifest = load_manifest(self.write_manifest(make_manifest([make_episode("known")])))
        unknown = load_predictions(self.write_predictions([{
            "episode_id": "unknown", "stage": "seed", "rank": 1,
            "region": "room_a", "x": 1.0, "y": 2.0, "yaw_deg": 179.0, "status": "",
        }]))
        with self.assertRaises(EvaluationInputError):
            evaluate(manifest, unknown)

        decision_row = {
            "episode_id": "known", "stage": "decision", "rank": 1,
            "region": "", "x": "", "y": "", "yaw_deg": "", "status": "NoCandidate",
        }
        duplicate = load_predictions(self.write_predictions([decision_row, decision_row]))
        with self.assertRaises(EvaluationInputError):
            evaluate(manifest, duplicate)


if __name__ == "__main__":
    unittest.main()
