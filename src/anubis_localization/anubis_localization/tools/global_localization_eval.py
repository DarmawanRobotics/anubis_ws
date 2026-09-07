#!/usr/bin/env python3

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


AUTO_GATE_GT_SOURCES = {
    "external_measurement",
    "calibrated_fixture",
    "independent_reference",
}
ROOM_ONLY_GT_SOURCE = "human_region_label"
UNMAPPED_REGION = "__unmapped__"
PREDICTION_STAGES = {"seed", "coarse", "refined", "cluster", "decision"}


class EvaluationInputError(ValueError):
    pass


@dataclass(frozen=True)
class RegionPolygon:
    name: str
    points: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class GroundTruthEpisode:
    episode_id: str
    bag_path: str
    scene: str
    region: str
    pose_kind: str
    x: Optional[float]
    y: Optional[float]
    yaw_deg: Optional[float]
    auto_gate_eligible: bool
    exclusion_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class EvaluationManifest:
    map_version: str
    time_residual_ms: float
    regions: Tuple[RegionPolygon, ...]
    episodes: Tuple[GroundTruthEpisode, ...]


@dataclass(frozen=True)
class Prediction:
    episode_id: str
    stage: str
    rank: int
    region: str
    x: Optional[float]
    y: Optional[float]
    yaw_deg: Optional[float]
    status: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationInputError(message)


def _required_string(value: object, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must be a non-empty string")
    return value.strip()


def _finite_number(value: object, field: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _optional_float(value: str, field: str) -> Optional[float]:
    if value.strip() == "":
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise EvaluationInputError(f"{field} must be numeric or empty") from exc
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _validate_unusable_intervals(intervals: object, episode_id: str) -> None:
    _require(isinstance(intervals, list),
             f"episode {episode_id}: unusable_intervals must be a list")
    for index, interval in enumerate(intervals):
        _require(isinstance(interval, dict),
                 f"episode {episode_id}: unusable interval {index} must be an object")
        start = _finite_number(interval.get("start_s"),
                               f"episode {episode_id}.unusable_intervals[{index}].start_s")
        end = _finite_number(interval.get("end_s"),
                             f"episode {episode_id}.unusable_intervals[{index}].end_s")
        _require(0.0 <= start < end,
                 f"episode {episode_id}: unusable interval must satisfy 0 <= start < end")


def _point_on_segment(x: float, y: float,
                      first: Tuple[float, float],
                      second: Tuple[float, float]) -> bool:
    cross = ((x - first[0]) * (second[1] - first[1]) -
             (y - first[1]) * (second[0] - first[0]))
    scale = max(1.0, abs(x), abs(y), abs(first[0]), abs(first[1]),
                abs(second[0]), abs(second[1]))
    if abs(cross) > 1e-9 * scale:
        return False
    return (min(first[0], second[0]) - 1e-9 <= x <=
            max(first[0], second[0]) + 1e-9 and
            min(first[1], second[1]) - 1e-9 <= y <=
            max(first[1], second[1]) + 1e-9)


def _point_in_polygon(x: float, y: float,
                      polygon: RegionPolygon) -> bool:
    inside = False
    points = polygon.points
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        if _point_on_segment(x, y, first, second):
            return True
        if ((first[1] > y) != (second[1] > y)):
            crossing_x = (second[0] - first[0]) * (y - first[1]) / (
                second[1] - first[1]) + first[0]
            if x < crossing_x:
                inside = not inside
    return inside


def _resolve_region(x: float, y: float,
                    regions: Sequence[RegionPolygon],
                    field: str,
                    allow_unmapped: bool = False) -> str:
    matches = [region.name for region in regions
               if _point_in_polygon(x, y, region)]
    if not matches and allow_unmapped:
        return UNMAPPED_REGION
    _require(bool(matches), f"{field} is outside every configured region polygon")
    _require(len(matches) == 1,
             f"{field} is ambiguous across region polygons: {matches}")
    return matches[0]


def _load_regions(document: Dict[str, object]) -> Tuple[RegionPolygon, ...]:
    raw_regions = document.get("regions", [])
    _require(isinstance(raw_regions, list), "regions must be a list")
    regions: List[RegionPolygon] = []
    seen_names = set()
    for region_index, raw_region in enumerate(raw_regions):
        prefix = f"regions[{region_index}]"
        _require(isinstance(raw_region, dict), f"{prefix} must be an object")
        name = _required_string(raw_region.get("name"), f"{prefix}.name")
        _require(name not in seen_names, f"duplicate region name: {name}")
        seen_names.add(name)
        raw_polygon = raw_region.get("polygon")
        _require(isinstance(raw_polygon, list) and len(raw_polygon) >= 3,
                 f"{prefix}.polygon must contain at least 3 XY points")
        points: List[Tuple[float, float]] = []
        for point_index, raw_point in enumerate(raw_polygon):
            _require(isinstance(raw_point, list) and len(raw_point) == 2,
                     f"{prefix}.polygon[{point_index}] must be [x, y]")
            points.append((
                _finite_number(raw_point[0],
                               f"{prefix}.polygon[{point_index}][0]"),
                _finite_number(raw_point[1],
                               f"{prefix}.polygon[{point_index}][1]"),
            ))
        twice_area = sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(points, points[1:] + points[:1]))
        _require(abs(twice_area) > 1e-9,
                 f"{prefix}.polygon must have non-zero area")
        regions.append(RegionPolygon(name=name, points=tuple(points)))
    return tuple(regions)


def load_manifest(path: Path) -> EvaluationManifest:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"failed to read manifest {path}: {exc}") from exc

    _require(isinstance(document, dict), "manifest root must be an object")
    _require(document.get("schema_version") == 1, "schema_version must be 1")
    map_version = _required_string(document.get("map_version"), "map_version")
    _require(document.get("coordinate_frame") == "map", "coordinate_frame must be 'map'")

    transform = document.get("T_map_gt")
    _require(isinstance(transform, list) and len(transform) == 16,
             "T_map_gt must contain 16 row-major matrix values")
    for index, value in enumerate(transform):
        _finite_number(value, f"T_map_gt[{index}]")

    time_sync = document.get("time_sync")
    _require(isinstance(time_sync, dict), "time_sync must be an object")
    _required_string(time_sync.get("method"), "time_sync.method")
    time_residual_ms = _finite_number(
        time_sync.get("residual_p99_ms"), "time_sync.residual_p99_ms")
    _require(time_residual_ms >= 0.0, "time_sync.residual_p99_ms must be >= 0")
    regions = _load_regions(document)
    region_names = {region.name for region in regions}

    raw_episodes = document.get("episodes")
    _require(isinstance(raw_episodes, list) and bool(raw_episodes),
             "episodes must be a non-empty list")

    episodes: List[GroundTruthEpisode] = []
    seen_ids = set()
    for index, raw in enumerate(raw_episodes):
        _require(isinstance(raw, dict), f"episodes[{index}] must be an object")
        prefix = f"episodes[{index}]"
        episode_id = _required_string(raw.get("episode_id"), f"{prefix}.episode_id")
        _require(episode_id not in seen_ids, f"duplicate episode_id: {episode_id}")
        seen_ids.add(episode_id)

        bag_path = _required_string(raw.get("bag_path"), f"{prefix}.bag_path")
        scene = _required_string(raw.get("scene"), f"{prefix}.scene")
        region = _required_string(raw.get("region"), f"{prefix}.region")
        if regions:
            _require(region in region_names,
                     f"{prefix}.region is not defined in regions")
        _required_string(raw.get("annotator"), f"{prefix}.annotator")
        _required_string(raw.get("reviewer"), f"{prefix}.reviewer")
        _validate_unusable_intervals(raw.get("unusable_intervals", []), episode_id)

        source = raw.get("gt_source")
        _require(isinstance(source, dict), f"{prefix}.gt_source must be an object")
        source_type = _required_string(source.get("type"), f"{prefix}.gt_source.type")
        _required_string(source.get("record_id"), f"{prefix}.gt_source.record_id")
        independent = source.get("independent_from_localization")
        _require(isinstance(independent, bool),
                 f"{prefix}.gt_source.independent_from_localization must be boolean")

        pose_kind = _required_string(raw.get("pose_kind"), f"{prefix}.pose_kind")
        _require(pose_kind in {"full_pose", "region_only"},
                 f"{prefix}.pose_kind must be full_pose or region_only")

        x = y = yaw_deg = None
        exclusion_reasons: List[str] = []
        if pose_kind == "region_only":
            _require(source_type == ROOM_ONLY_GT_SOURCE,
                     f"{prefix}: region_only requires gt_source.type={ROOM_ONLY_GT_SOURCE}")
            exclusion_reasons.append("region_only_gt")
        else:
            _require(source_type in AUTO_GATE_GT_SOURCES,
                     f"{prefix}: full_pose GT source is not approved")
            _require(independent,
                     f"{prefix}: full_pose GT must be independent from localization")
            pose = raw.get("pose")
            _require(isinstance(pose, dict), f"{prefix}.pose must be an object")
            x = _finite_number(pose.get("x"), f"{prefix}.pose.x")
            y = _finite_number(pose.get("y"), f"{prefix}.pose.y")
            yaw_deg = _finite_number(pose.get("yaw_deg"), f"{prefix}.pose.yaw_deg")
            if regions:
                resolved_region = _resolve_region(
                    x, y, regions, f"{prefix}.pose")
                _require(resolved_region == region,
                         f"{prefix}.pose maps to region {resolved_region}, not {region}")

            uncertainty = raw.get("uncertainty")
            _require(isinstance(uncertainty, dict),
                     f"{prefix}.uncertainty must be an object")
            xy_uncertainty = _finite_number(
                uncertainty.get("xy_m"), f"{prefix}.uncertainty.xy_m")
            yaw_uncertainty = _finite_number(
                uncertainty.get("yaw_deg"), f"{prefix}.uncertainty.yaw_deg")
            _require(xy_uncertainty >= 0.0 and yaw_uncertainty >= 0.0,
                     f"{prefix}: uncertainty values must be >= 0")
            if xy_uncertainty > 0.10:
                exclusion_reasons.append("xy_uncertainty_gt_0.10m")
            if yaw_uncertainty > 2.0:
                exclusion_reasons.append("yaw_uncertainty_gt_2deg")
            if time_residual_ms > 20.0:
                exclusion_reasons.append("time_residual_gt_20ms")

        episodes.append(GroundTruthEpisode(
            episode_id=episode_id,
            bag_path=bag_path,
            scene=scene,
            region=region,
            pose_kind=pose_kind,
            x=x,
            y=y,
            yaw_deg=yaw_deg,
            auto_gate_eligible=pose_kind == "full_pose" and not exclusion_reasons,
            exclusion_reasons=tuple(exclusion_reasons),
        ))

    return EvaluationManifest(
        map_version=map_version,
        time_residual_ms=time_residual_ms,
        regions=regions,
        episodes=tuple(episodes),
    )


def load_predictions(
        path: Path,
        regions: Sequence[RegionPolygon] = ()) -> Tuple[Prediction, ...]:
    required_columns = {
        "episode_id", "stage", "rank", "region", "x", "y", "yaw_deg", "status"
    }
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise EvaluationInputError(f"failed to read predictions {path}: {exc}") from exc

    predictions: List[Prediction] = []
    with stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None, "prediction CSV is missing a header")
        missing = required_columns.difference(reader.fieldnames)
        _require(not missing, f"prediction CSV missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            episode_id = _required_string(row["episode_id"], f"CSV row {row_number}.episode_id")
            stage = _required_string(row["stage"], f"CSV row {row_number}.stage")
            _require(stage in PREDICTION_STAGES,
                     f"CSV row {row_number}: unknown stage {stage}")
            try:
                rank = int(row["rank"])
            except ValueError as exc:
                raise EvaluationInputError(f"CSV row {row_number}.rank must be an integer") from exc
            _require(rank > 0, f"CSV row {row_number}.rank must be > 0")
            region = row["region"].strip()
            x = _optional_float(row["x"], f"CSV row {row_number}.x")
            y = _optional_float(row["y"], f"CSV row {row_number}.y")
            yaw_deg = _optional_float(row["yaw_deg"], f"CSV row {row_number}.yaw_deg")
            status = row["status"].strip()

            candidate_required = stage != "decision" or status == "Confirmed"
            if candidate_required:
                _require(x is not None and y is not None and yaw_deg is not None,
                         f"CSV row {row_number}: candidate pose fields are required")
                if regions:
                    resolved_region = _resolve_region(
                        x, y, regions, f"CSV row {row_number} pose",
                        allow_unmapped=True)
                    if region:
                        _require(region == resolved_region,
                                 f"CSV row {row_number}.region={region} disagrees "
                                 f"with polygon-derived region={resolved_region}")
                    region = resolved_region
                else:
                    _require(bool(region),
                             f"CSV row {row_number}.region is required when "
                             "manifest regions are not configured")

            predictions.append(Prediction(
                episode_id=episode_id,
                stage=stage,
                rank=rank,
                region=region,
                x=x,
                y=y,
                yaw_deg=yaw_deg,
                status=status,
            ))
    return tuple(predictions)


def circular_yaw_error_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _room_match(gt: GroundTruthEpisode, prediction: Prediction) -> bool:
    return prediction.region == gt.region


def _pose_match(gt: GroundTruthEpisode, prediction: Prediction,
                xy_tolerance_m: float, yaw_tolerance_deg: float) -> bool:
    if prediction.x is None or prediction.y is None or prediction.yaw_deg is None:
        return False
    if gt.x is None or gt.y is None or gt.yaw_deg is None:
        return False
    xy_error = math.hypot(prediction.x - gt.x, prediction.y - gt.y)
    yaw_error = circular_yaw_error_deg(prediction.yaw_deg, gt.yaw_deg)
    return _room_match(gt, prediction) and xy_error <= xy_tolerance_m and yaw_error <= yaw_tolerance_deg


def _ratio(hits: int, total: int) -> Dict[str, Optional[float]]:
    return {
        "hits": hits,
        "total": total,
        "rate": hits / total if total else None,
    }


def _evaluate_group(episodes: Sequence[GroundTruthEpisode],
                    by_episode: Dict[str, List[Prediction]],
                    k_values: Sequence[int],
                    xy_tolerance_m: float,
                    yaw_tolerance_deg: float,
                    coarse_xy_tolerance_m: float,
                    coarse_yaw_tolerance_deg: float) -> Dict[str, object]:
    eligible = [episode for episode in episodes if episode.auto_gate_eligible]
    metrics: Dict[str, object] = {}

    def stage_rows(episode: GroundTruthEpisode, stage: str,
                   max_rank: Optional[int] = None) -> List[Prediction]:
        rows = [row for row in by_episode.get(episode.episode_id, []) if row.stage == stage]
        if max_rank is not None:
            rows = [row for row in rows if row.rank <= max_rank]
        if stage == "decision":
            rows = [row for row in rows if row.status == "Confirmed"]
        return rows

    seed_hits = sum(any(_pose_match(ep, row, xy_tolerance_m, yaw_tolerance_deg)
                        for row in stage_rows(ep, "seed")) for ep in eligible)
    metrics["seed_recall"] = _ratio(seed_hits, len(eligible))

    # The "coarse" stage carries Level 0 top regions: hypotheses on a 2 m XY
    # grid with 90 deg yaw bins. A region is a recall success when the ground
    # truth lies within the region's Level 1 expansion reach, so it needs its
    # own tolerance — the main pose tolerance (default 1.0 m) is below the
    # grid's worst-case center offset (~1.414 m) and would report 0% recall
    # regardless of algorithm quality.
    for k_value in k_values:
        coarse_hits = sum(any(_pose_match(ep, row, coarse_xy_tolerance_m,
                                          coarse_yaw_tolerance_deg)
                              for row in stage_rows(ep, "coarse", k_value)) for ep in eligible)
        metrics[f"coarse_recall@{k_value}"] = _ratio(coarse_hits, len(eligible))

    refined_hits = sum(any(_pose_match(ep, row, xy_tolerance_m, yaw_tolerance_deg)
                           for row in stage_rows(ep, "refined")) for ep in eligible)
    metrics["refined_recall"] = _ratio(refined_hits, len(eligible))
    refined_passed_hits = sum(
        any(_pose_match(ep, row, xy_tolerance_m, yaw_tolerance_deg)
            for row in stage_rows(ep, "refined") if row.status == "Passed")
        for ep in eligible)
    metrics["refined_passed_recall"] = _ratio(refined_passed_hits, len(eligible))

    cluster_hits = sum(any(_pose_match(ep, row, xy_tolerance_m, yaw_tolerance_deg)
                           for row in stage_rows(ep, "cluster")) for ep in eligible)
    metrics["cluster_recall"] = _ratio(cluster_hits, len(eligible))

    confirmed_count = 0
    false_confirm_count = 0
    for episode in eligible:
        decisions = stage_rows(episode, "decision")
        if decisions:
            confirmed_count += 1
            if not _pose_match(episode, decisions[0], xy_tolerance_m, yaw_tolerance_deg):
                false_confirm_count += 1
    metrics["false_confirm_rate"] = {
        "false_confirm_count": false_confirm_count,
        "confirmed_count": confirmed_count,
        "eligible_episode_count": len(eligible),
        "rate_per_confirmed": (
            false_confirm_count / confirmed_count if confirmed_count else None
        ),
        "rate_per_episode": (
            false_confirm_count / len(eligible) if eligible else None
        ),
    }

    room_metrics: Dict[str, object] = {}
    for stage in ("seed", "cluster", "decision"):
        room_hits = sum(any(_room_match(ep, row) for row in stage_rows(ep, stage))
                        for ep in episodes)
        room_metrics[stage] = _ratio(room_hits, len(episodes))
    for k_value in k_values:
        room_hits = sum(any(_room_match(ep, row)
                            for row in stage_rows(ep, "coarse", k_value))
                        for ep in episodes)
        room_metrics[f"coarse@{k_value}"] = _ratio(room_hits, len(episodes))
    metrics["room_recall"] = room_metrics
    return metrics


def _episode_diagnostics(episodes: Sequence[GroundTruthEpisode],
                         by_episode: Dict[str, List[Prediction]],
                         stage_tolerances: Dict[str, Tuple[float, float]]
                         ) -> Dict[str, object]:
    """Per-episode, per-stage survival report: at which rank (if any) the
    ground truth is matched, and how close the nearest row gets. This answers
    "where in the pipeline was the correct pose lost" without re-running."""
    diagnostics: Dict[str, object] = {}
    for episode in episodes:
        rows_all = by_episode.get(episode.episode_id, [])
        stage_report: Dict[str, object] = {}
        for stage in ("seed", "coarse", "refined", "cluster", "decision"):
            rows = sorted((row for row in rows_all if row.stage == stage),
                          key=lambda row: row.rank)
            xy_tolerance_m, yaw_tolerance_deg = stage_tolerances[stage]
            room_matched_rank: Optional[int] = None
            pose_matched_rank: Optional[int] = None
            min_xy_error: Optional[float] = None
            min_xy_rank: Optional[int] = None
            min_xy_yaw_error: Optional[float] = None
            for row in rows:
                if room_matched_rank is None and _room_match(episode, row):
                    room_matched_rank = row.rank
                if (episode.x is None or episode.y is None or episode.yaw_deg is None
                        or row.x is None or row.y is None or row.yaw_deg is None):
                    continue
                xy_error = math.hypot(row.x - episode.x, row.y - episode.y)
                yaw_error = circular_yaw_error_deg(row.yaw_deg, episode.yaw_deg)
                if min_xy_error is None or xy_error < min_xy_error:
                    min_xy_error = xy_error
                    min_xy_rank = row.rank
                    min_xy_yaw_error = yaw_error
                if pose_matched_rank is None and _pose_match(
                        episode, row, xy_tolerance_m, yaw_tolerance_deg):
                    pose_matched_rank = row.rank
            statuses = sorted({row.status for row in rows if row.status})
            stage_report[stage] = {
                "row_count": len(rows),
                "room_matched_rank": room_matched_rank,
                "pose_matched_rank": pose_matched_rank,
                "min_xy_error_m": (
                    None if min_xy_error is None else round(min_xy_error, 3)),
                "min_xy_rank": min_xy_rank,
                "yaw_error_at_min_xy_deg": (
                    None if min_xy_yaw_error is None
                    else round(min_xy_yaw_error, 2)),
                "statuses": statuses,
            }
        diagnostics[episode.episode_id] = stage_report
    return diagnostics


def evaluate(manifest: EvaluationManifest,
             predictions: Sequence[Prediction],
             k_values: Sequence[int] = (4, 6, 8, 12),
             xy_tolerance_m: float = 1.0,
             yaw_tolerance_deg: float = 20.0,
             coarse_xy_tolerance_m: float = 1.5,
             coarse_yaw_tolerance_deg: float = 45.0) -> Dict[str, object]:
    _require(xy_tolerance_m > 0.0 and math.isfinite(xy_tolerance_m),
             "xy_tolerance_m must be finite and > 0")
    _require(yaw_tolerance_deg > 0.0 and math.isfinite(yaw_tolerance_deg),
             "yaw_tolerance_deg must be finite and > 0")
    _require(coarse_xy_tolerance_m > 0.0 and math.isfinite(coarse_xy_tolerance_m),
             "coarse_xy_tolerance_m must be finite and > 0")
    _require(coarse_yaw_tolerance_deg > 0.0 and math.isfinite(coarse_yaw_tolerance_deg),
             "coarse_yaw_tolerance_deg must be finite and > 0")
    _require(bool(k_values) and all(isinstance(value, int) and value > 0 for value in k_values),
             "k_values must contain positive integers")
    k_values = tuple(sorted(set(k_values)))

    episode_lookup = {episode.episode_id: episode for episode in manifest.episodes}
    by_episode: Dict[str, List[Prediction]] = {}
    for prediction in predictions:
        _require(prediction.episode_id in episode_lookup,
                 f"prediction references unknown episode_id: {prediction.episode_id}")
        by_episode.setdefault(prediction.episode_id, []).append(prediction)

    for episode_id, rows in by_episode.items():
        decisions = [row for row in rows if row.stage == "decision"]
        _require(len(decisions) <= 1,
                 f"episode {episode_id} has more than one decision row")

    scene_names = sorted({episode.scene for episode in manifest.episodes})
    exclusions = [
        {"episode_id": episode.episode_id, "reasons": list(episode.exclusion_reasons)}
        for episode in manifest.episodes if not episode.auto_gate_eligible
    ]
    stage_tolerances = {
        "seed": (xy_tolerance_m, yaw_tolerance_deg),
        "coarse": (coarse_xy_tolerance_m, coarse_yaw_tolerance_deg),
        "refined": (xy_tolerance_m, yaw_tolerance_deg),
        "cluster": (xy_tolerance_m, yaw_tolerance_deg),
        "decision": (xy_tolerance_m, yaw_tolerance_deg),
    }
    return {
        "schema_version": 1,
        "map_version": manifest.map_version,
        "tolerances": {
            "xy_m": xy_tolerance_m,
            "yaw_deg": yaw_tolerance_deg,
            "coarse_xy_m": coarse_xy_tolerance_m,
            "coarse_yaw_deg": coarse_yaw_tolerance_deg,
            "k_values": list(k_values),
        },
        "counts": {
            "episodes_total": len(manifest.episodes),
            "regions_defined": len(manifest.regions),
            "auto_gate_eligible": sum(ep.auto_gate_eligible for ep in manifest.episodes),
            "excluded_from_auto_gate": len(exclusions),
        },
        "metrics": _evaluate_group(
            manifest.episodes, by_episode, k_values, xy_tolerance_m,
            yaw_tolerance_deg, coarse_xy_tolerance_m, coarse_yaw_tolerance_deg),
        "by_scene": {
            scene: _evaluate_group(
                [ep for ep in manifest.episodes if ep.scene == scene],
                by_episode, k_values, xy_tolerance_m, yaw_tolerance_deg,
                coarse_xy_tolerance_m, coarse_yaw_tolerance_deg)
            for scene in scene_names
        },
        "episode_diagnostics": _episode_diagnostics(
            manifest.episodes, by_episode, stage_tolerances),
        "auto_gate_exclusions": exclusions,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Global localization M2a offline evaluator")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--xy-tolerance-m", type=float, default=1.0)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=20.0)
    # Level 0 top regions live on a 2 m grid with 90 deg yaw bins; their
    # recall tolerance reflects the Level 1 expansion reach, not final pose
    # accuracy. Pass the configured level1_search_radius_xy (e.g. 3.0) to
    # measure the "recoverable" recall variant.
    parser.add_argument("--coarse-xy-tolerance-m", type=float, default=1.5)
    parser.add_argument("--coarse-yaw-tolerance-deg", type=float, default=45.0)
    parser.add_argument("--k", type=int, action="append", dest="k_values")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.predictions is None:
            report: Dict[str, object] = {
                "schema_version": 1,
                "map_version": manifest.map_version,
                "episodes_total": len(manifest.episodes),
                "regions_defined": len(manifest.regions),
                "auto_gate_eligible": sum(ep.auto_gate_eligible for ep in manifest.episodes),
                "auto_gate_exclusions": [
                    {"episode_id": ep.episode_id, "reasons": list(ep.exclusion_reasons)}
                    for ep in manifest.episodes if not ep.auto_gate_eligible
                ],
            }
        else:
            predictions = load_predictions(args.predictions, manifest.regions)
            report = evaluate(
                manifest,
                predictions,
                k_values=args.k_values or (4, 6, 8, 12),
                xy_tolerance_m=args.xy_tolerance_m,
                yaw_tolerance_deg=args.yaw_tolerance_deg,
                coarse_xy_tolerance_m=args.coarse_xy_tolerance_m,
                coarse_yaw_tolerance_deg=args.coarse_yaw_tolerance_deg,
            )
    except EvaluationInputError as exc:
        print(f"ERROR: {exc}")
        return 2

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
