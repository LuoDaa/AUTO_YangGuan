"""CrownFlow-style insertion-axis solver, adapted for AUTO_YangGuan.

The production contract follows CrownFlow's mature logic:
1) validate Margin and fit a robust Margin plane;
2) estimate an independent side-wall axis from area-weighted face normals;
3) combine the two only when they agree reasonably;
4) search a bounded cone around the initial axis;
5) score candidates by draft-angle undercut, Margin visibility proxy, and deviation;
6) expose REVIEW_REQUIRED when the objective is flat or evidence disagrees.

This module intentionally avoids world-Z assumptions and direct support PCA ownership.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import trimesh
from scipy.spatial import cKDTree

DEFAULT_MAX_DEVIATION_DEG = 20.0
DRAFT_TOLERANCE_DEG = 1.0
MAX_INITIAL_DISAGREEMENT_DEG = 35.0
MAX_ACCEPTABLE_UNDERCUT_RATIO = 0.15
MIN_ACCEPTABLE_MARGIN_VISIBILITY = 0.90
FLATNESS_THRESHOLD = 0.02
ABOVE_MARGIN_TOLERANCE_MM = 0.05


class AxisStatus(str, Enum):
    OK = "OK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"


@dataclass
class AxisResult:
    axis: np.ndarray
    status: AxisStatus
    report: dict


def _unit(v, name="vector") -> np.ndarray:
    v = np.asarray(v, float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"{name} is not a finite 3-vector")
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        raise ValueError(f"{name} is degenerate")
    return v / n


def angle_deg(a, b) -> float:
    a = _unit(a)
    b = _unit(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))


def _robust_plane(points: np.ndarray, iterations: int = 4, keep: float = 0.82):
    pts = np.asarray(points, float)
    mask = np.ones(len(pts), bool)
    center = pts.mean(axis=0)
    normal = np.array([0.0, 0.0, 1.0])
    for _ in range(iterations):
        sub = pts[mask]
        if len(sub) < 3:
            break
        center = sub.mean(axis=0)
        _, _, vh = np.linalg.svd(sub - center, full_matrices=False)
        normal = _unit(vh[-1], "margin plane normal")
        residual = np.abs((pts - center) @ normal)
        threshold = float(np.quantile(residual[mask], keep))
        new_mask = residual <= max(threshold, 1e-8)
        if int(new_mask.sum()) < 3 or np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask
    sub = pts[mask] if int(mask.sum()) >= 3 else pts
    center = sub.mean(axis=0)
    _, _, vh = np.linalg.svd(sub - center, full_matrices=False)
    normal = _unit(vh[-1], "margin plane normal")
    rel = sub - center
    rel = rel - np.outer(rel @ normal, normal)
    if len(rel) >= 3:
        _, _, inplane_vh = np.linalg.svd(rel, full_matrices=False)
        e1 = _unit(inplane_vh[0], "margin in-plane axis")
    else:
        e1 = None
    return center, normal, mask, e1


def validate_margin(margin_points, support_mesh: trimesh.Trimesh) -> dict:
    pts = np.asarray(margin_points, float)
    report = {
        "status": AxisStatus.FAILED.value,
        "reasons": [],
        "n_points": int(len(pts)) if pts.ndim else 0,
    }
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 8:
        report["reasons"].append("MARGIN_POINT_COUNT_INSUFFICIENT")
        return report
    if not np.isfinite(pts).all():
        report["reasons"].append("MARGIN_NONFINITE")
        return report

    step = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    median_step = float(np.median(step))
    if median_step <= 1e-9:
        report["reasons"].append("MARGIN_DEGENERATE_SAMPLING")
        return report

    center, normal, mask, e1 = _robust_plane(pts)
    support_center = np.asarray(support_mesh.vertices, float).mean(axis=0)
    if float((support_center - center) @ normal) < 0:
        normal = -normal
    residual = np.abs((pts - center) @ normal)
    span = max(float(np.linalg.norm(np.ptp(pts, axis=0))), 1e-6)
    planarity_rms = float(np.sqrt(np.mean(residual ** 2)))
    spacing_cv = float(np.std(step) / median_step)
    max_step_ratio = float(np.max(step) / median_step)

    reasons = []
    if planarity_rms > 0.25 * span:
        reasons.append("MARGIN_EXTREME_NONPLANARITY")
    if spacing_cv > 1.5 or max_step_ratio > 5.0:
        reasons.append("MARGIN_SAMPLING_IRREGULAR")

    report.update(
        {
            "status": AxisStatus.REVIEW_REQUIRED.value if reasons else AxisStatus.OK.value,
            "reasons": reasons,
            "plane_origin": center.tolist(),
            "plane_normal": normal.tolist(),
            "in_plane_e1": None if e1 is None else e1.tolist(),
            "planarity_rms_mm": planarity_rms,
            "spacing_cv": spacing_cv,
            "max_step_ratio": max_step_ratio,
            "robust_inlier_fraction": float(np.mean(mask)),
        }
    )
    return report


def estimate_wall_axis(
    support_mesh: trimesh.Trimesh,
    seed_axis: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Least-squares side-wall axis from area-weighted face normals."""
    normals = np.asarray(support_mesh.face_normals, float)
    areas = np.asarray(support_mesh.area_faces, float)
    centers = np.asarray(support_mesh.triangles_center, float)
    axis = _unit(seed_axis, "seed axis")
    history = []

    for iteration in range(6):
        projection = centers @ axis
        lo, hi = np.quantile(projection, [0.05, 0.95])
        body = (projection >= lo) & (projection <= hi)
        wall = body & (np.abs(normals @ axis) < 0.78) & (areas > 1e-12)
        if int(wall.sum()) < 18:
            wall = body & (np.abs(normals @ axis) < 0.90) & (areas > 1e-12)
        if int(wall.sum()) < 12:
            break
        n = normals[wall]
        w = areas[wall]
        covariance = (n * w[:, None]).T @ n / max(float(w.sum()), 1e-12)
        values, vectors = np.linalg.eigh(covariance)
        revised = _unit(vectors[:, int(np.argmin(values))], "wall axis")
        if float(revised @ axis) < 0:
            revised = -revised
        delta = angle_deg(axis, revised)
        axis = _unit(0.15 * axis + 0.85 * revised, "wall axis blended")
        history.append(
            {
                "iteration": iteration + 1,
                "wall_face_count": int(wall.sum()),
                "wall_area_fraction": float(areas[wall].sum() / max(float(areas.sum()), 1e-12)),
                "normal_cov_eigenvalues": [float(x) for x in values],
                "axis_change_deg": float(delta),
            }
        )
        if delta < 0.12:
            break

    return axis, {
        "method": "AREA_WEIGHTED_SIDEWALL_NORMAL_COVARIANCE",
        "iterations": history,
        "final_axis": axis.tolist(),
    }


def _orthonormal_basis(axis: np.ndarray, e1_hint: np.ndarray | None = None):
    axis = _unit(axis)
    if e1_hint is not None:
        e1 = np.asarray(e1_hint, float)
        e1 = e1 - axis * float(e1 @ axis)
        if float(np.linalg.norm(e1)) > 1e-6:
            e1 = _unit(e1)
            return e1, _unit(np.cross(axis, e1))
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(helper @ axis)) > 0.85:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(axis, helper))
    return e1, _unit(np.cross(axis, e1))


def _cone_candidates(initial: np.ndarray, max_deviation_deg: float, e1_hint=None):
    initial = _unit(initial)
    e1, e2 = _orthonormal_basis(initial, e1_hint)
    candidates = [initial]
    for angle in np.linspace(4.0, max_deviation_deg, 5):
        a = math.radians(float(angle))
        for phi in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            radial = math.cos(phi) * e1 + math.sin(phi) * e2
            candidates.append(_unit(math.cos(a) * initial + math.sin(a) * radial))
    return candidates


def _nearest_margin_height(points: np.ndarray, margin: np.ndarray, axis: np.ndarray):
    axis = _unit(axis)
    origin = np.asarray(margin, float).mean(axis=0)
    e1, e2 = _orthonormal_basis(axis)
    basis = np.stack([e1, e2, axis], axis=1)
    p = (np.asarray(points, float) - origin) @ basis
    m = (np.asarray(margin, float) - origin) @ basis
    tree = cKDTree(m[:, :2])
    _, ids = tree.query(p[:, :2], k=1)
    return p[:, 2] - m[np.asarray(ids, int), 2], p, m


def _margin_visibility_proxy(
    support_mesh: trimesh.Trimesh,
    margin: np.ndarray,
    axis: np.ndarray,
) -> float:
    """Cheap visibility proxy independent of optional ray/rtree backends.

    A Margin sample is treated as visible if the support does not have a higher
    vertex very close to the same perpendicular-axis column. This deliberately
    errs toward REVIEW_REQUIRED rather than pretending exact ray visibility.
    """
    vertices = np.asarray(support_mesh.vertices, float)
    _, pv, pm = _nearest_margin_height(vertices, margin, axis)
    tree = cKDTree(pv[:, :2])
    visible = []
    for q in pm:
        ids = tree.query_ball_point(q[:2], r=0.30)
        if not ids:
            visible.append(True)
            continue
        zmax = float(np.max(pv[np.asarray(ids, int), 2]))
        visible.append(zmax <= float(q[2]) + 0.20)
    return float(np.mean(visible)) if visible else 0.0


def _score_axis(
    support_mesh: trimesh.Trimesh,
    margin: np.ndarray,
    axis: np.ndarray,
    initial_axis: np.ndarray,
):
    axis = _unit(axis)
    centers = np.asarray(support_mesh.triangles_center, float)
    normals = np.asarray(support_mesh.face_normals, float)
    areas = np.asarray(support_mesh.area_faces, float)
    height, _, _ = _nearest_margin_height(centers, margin, axis)
    active = (height > ABOVE_MARGIN_TOLERANCE_MM) & (areas > 1e-12)
    if int(active.sum()) < 12:
        return None
    threshold = -math.sin(math.radians(DRAFT_TOLERANCE_DEG))
    dot = normals @ axis
    undercut = active & (dot < threshold)
    active_area = float(areas[active].sum())
    undercut_area = float(areas[undercut].sum())
    ratio = undercut_area / max(active_area, 1e-12)
    if undercut.any():
        severity = np.maximum(0.0, -dot[undercut])
        mean_severity = float(np.average(severity, weights=areas[undercut]))
        max_severity = float(np.max(severity))
    else:
        mean_severity = 0.0
        max_severity = 0.0
    visibility = _margin_visibility_proxy(support_mesh, margin, axis)
    deviation = angle_deg(axis, initial_axis)
    score = 3.0 * ratio + 0.7 * mean_severity + 1.0 * (1.0 - visibility) + 0.012 * deviation
    return {
        "axis": axis,
        "score": float(score),
        "undercut_area_mm2": undercut_area,
        "undercut_area_ratio": float(ratio),
        "mean_undercut_severity": mean_severity,
        "max_undercut_severity": max_severity,
        "margin_visibility_ratio": visibility,
        "deviation_deg": float(deviation),
        "active_face_count": int(active.sum()),
    }


def solve_insertion_axis(
    support_mesh: trimesh.Trimesh,
    margin_points,
    rough_axis: np.ndarray | None = None,
    *,
    max_deviation_deg: float = DEFAULT_MAX_DEVIATION_DEG,
) -> AxisResult:
    margin = np.asarray(margin_points, float)
    validation = validate_margin(margin, support_mesh)
    if validation["status"] == AxisStatus.FAILED.value:
        raise ValueError("Margin validation failed: " + ",".join(validation["reasons"]))

    plane_normal = _unit(validation["plane_normal"], "Margin plane normal")
    if rough_axis is not None:
        rough = _unit(rough_axis, "rough axis")
        if float(plane_normal @ rough) < 0:
            plane_normal = -plane_normal
    else:
        rough = plane_normal.copy()

    wall_axis, wall_report = estimate_wall_axis(support_mesh, rough)
    if float(wall_axis @ plane_normal) < 0:
        wall_axis = -wall_axis
    disagreement = angle_deg(plane_normal, wall_axis)

    if disagreement <= MAX_INITIAL_DISAGREEMENT_DEG:
        initial = _unit(0.58 * plane_normal + 0.42 * wall_axis, "combined insertion axis")
        initial_source = "MARGIN_PLANE_PLUS_SIDEWALL"
    else:
        # CrownFlow deliberately does not average independent evidence that strongly disagrees.
        initial = plane_normal
        initial_source = "MARGIN_PLANE_DOMINANT_WALL_DISAGREES"

    e1_hint = validation.get("in_plane_e1")
    candidates = _cone_candidates(initial, float(max_deviation_deg), e1_hint)
    scored = [
        item
        for item in (_score_axis(support_mesh, margin, axis, initial) for axis in candidates)
        if item is not None
    ]
    if not scored:
        raise ValueError("Insertion-axis search produced no valid candidates")
    scored.sort(key=lambda row: row["score"])
    best = scored[0]

    # Fine search around the coarse winner.
    fine_e1, fine_e2 = _orthonormal_basis(best["axis"], e1_hint)
    fine = [best]
    for angle in (1.0, 2.0, 3.0, 4.0):
        a = math.radians(angle)
        for phi in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            radial = math.cos(phi) * fine_e1 + math.sin(phi) * fine_e2
            axis = _unit(math.cos(a) * best["axis"] + math.sin(a) * radial)
            if angle_deg(axis, initial) <= float(max_deviation_deg) + 1e-6:
                row = _score_axis(support_mesh, margin, axis, initial)
                if row is not None:
                    fine.append(row)
    fine.sort(key=lambda row: row["score"])
    best = fine[0]

    scores = np.asarray([row["score"] for row in scored], float)
    objective_range = float(np.max(scores) - np.min(scores)) if len(scores) else 0.0
    objective_is_flat = bool(objective_range < FLATNESS_THRESHOLD)

    reasons = []
    if validation["status"] != AxisStatus.OK.value:
        reasons.append("MARGIN_REVIEW_REQUIRED")
    if disagreement > MAX_INITIAL_DISAGREEMENT_DEG:
        reasons.append("MARGIN_WALL_AXIS_DISAGREE")
    if best["undercut_area_ratio"] > MAX_ACCEPTABLE_UNDERCUT_RATIO:
        reasons.append("UNDERCUT_RATIO_HIGH")
    if best["margin_visibility_ratio"] < MIN_ACCEPTABLE_MARGIN_VISIBILITY:
        reasons.append("MARGIN_VISIBILITY_LOW")
    if objective_is_flat:
        reasons.append("OBJECTIVE_FLAT_REVIEW_REQUIRED")

    status = AxisStatus.OK if not reasons else AxisStatus.REVIEW_REQUIRED
    confidence = max(
        0.0,
        min(
            1.0,
            1.0
            - 1.7 * best["undercut_area_ratio"]
            - 0.7 * (1.0 - best["margin_visibility_ratio"])
            - (0.25 if objective_is_flat else 0.0),
        ),
    )

    public_candidates = []
    for row in scored[:12]:
        public_candidates.append(
            {
                "axis": row["axis"].tolist(),
                "score": row["score"],
                "undercut_area_ratio": row["undercut_area_ratio"],
                "margin_visibility_ratio": row["margin_visibility_ratio"],
                "deviation_deg": row["deviation_deg"],
            }
        )

    report = {
        "algorithm_version": "crownflow/insertion-axis/ported-v1",
        "status": status.value,
        "reasons": reasons,
        "axis": best["axis"].tolist(),
        "initial_axis": initial.tolist(),
        "initial_source": initial_source,
        "margin_plane_axis": plane_normal.tolist(),
        "sidewall_axis": wall_axis.tolist(),
        "margin_wall_disagreement_deg": float(disagreement),
        "score": best["score"],
        "undercut_area_mm2": best["undercut_area_mm2"],
        "undercut_area_ratio": best["undercut_area_ratio"],
        "mean_undercut_severity": best["mean_undercut_severity"],
        "max_undercut_severity": best["max_undercut_severity"],
        "margin_visibility_ratio": best["margin_visibility_ratio"],
        "deviation_deg": best["deviation_deg"],
        "confidence": float(confidence),
        "objective_flatness": objective_range,
        "objective_is_flat": objective_is_flat,
        "margin_validation": validation,
        "wall_axis_fit": wall_report,
        "coarse_candidate_count": int(len(scored)),
        "top_candidates": public_candidates,
    }
    return AxisResult(axis=np.asarray(best["axis"], float), status=status, report=report)
