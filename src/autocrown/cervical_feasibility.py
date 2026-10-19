"""Read-only W3 cervical feasibility audit from CrownFlow's pre-stitch contract.

Nothing here moves the accepted outer, moves the owned inner, or publishes a
final crown. It checks whether the source surfaces are even safe candidates for
joining to the same confirmed 3D Margin before any collar triangles are built.

CrownFlow read-only reference: geometry/canonical_intaglio.py cervical_feasibility.
"""
from __future__ import annotations

import numpy as np
import trimesh
from trimesh.proximity import closest_point
from shapely.geometry import LineString

from .frame import PatientFrame, axial_height_from_margin
from .support_owned_intaglio import _boundary_loops, _triangle_clip_above


def _polyline_distance(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Point-to-3D-segment distances, not closest polyline VERTEX distances."""
    p = np.asarray(points, float)
    m = np.asarray(curve, float)
    if len(m) > 3 and np.linalg.norm(m[-1] - m[0]) < 1e-9:
        m = m[:-1]
    if len(m) < 3 or len(p) == 0:
        raise ValueError("INVALID_3D_MARGIN_OR_CERVICAL_LOOP")
    a, b = m, np.roll(m, -1, axis=0)
    edges = b - a
    d2 = np.einsum("ij,ij->i", edges, edges)
    if np.any(d2 < 1e-16):
        raise ValueError("DEGENERATE_MARGIN_EDGE")
    values = []
    for begin in range(0, len(p), 256):
        q = p[begin:begin + 256]
        t = np.clip(np.einsum("ijk,jk->ij", q[:, None] - a, edges)
                    / d2[None, :], 0.0, 1.0)
        foot = a[None, :, :] + t[:, :, None] * edges[None, :, :]
        values.append(np.sqrt(np.min(np.sum((q[:, None] - foot) ** 2, axis=2), axis=1)))
    return np.concatenate(values)


def _simple_xy_ring(points: np.ndarray, frame: PatientFrame) -> bool:
    xy = np.asarray(frame.to_local(points), float)[:, :2]
    if len(xy) < 8 or not np.isfinite(xy).all():
        return False
    if np.linalg.norm(xy[-1] - xy[0]) < 1e-10:
        xy = xy[:-1]
    edges = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    if np.any(edges < 1e-8):
        return False
    # A zero-area or self-crossing projected ring is not a valid canonical
    # parameter domain for the 1:1 cervical collar.
    shape = LineString(np.vstack([xy, xy[0]]))
    return bool(shape.is_simple and shape.length > 0.1)


def _separation_stats(source: trimesh.Trimesh, points: np.ndarray) -> dict:
    _, distance, _ = closest_point(source, points)
    d = np.asarray(distance, float)
    if not np.isfinite(d).all():
        raise ValueError("NONFINITE_SOURCE_SEPARATION")
    return {
        "min": float(np.min(d)),
        "p01": float(np.quantile(d, .01)),
        "p05": float(np.quantile(d, .05)),
        "median": float(np.median(d)),
        "fraction_below_0_03": float(np.mean(d < .03)),
        "fraction_below_0_05": float(np.mean(d < .05)),
        "fraction_below_0_10": float(np.mean(d < .10)),
    }


def audit_pre_stitch_feasibility(
    accepted_outer: trimesh.Trimesh,
    owned_inner: trimesh.Trimesh,
    margin: np.ndarray,
    frame: PatientFrame,
    *,
    transition_mm: float = 0.35,
) -> tuple[trimesh.Trimesh | None, dict]:
    """Validate a prospective outer/inner pair without constructing a seam.

    Returns a debug-only clipped outer iff it has exactly one simple cervical
    loop.  A PASS_ENGINEERING is NOT manufacturing, gap, or insertion approval.
    """
    report = {
        "status": "REVIEW_REQUIRED",
        "method": "CROWNFLOW_PRESPLICE_SOURCE_FEASIBILITY_V1",
        "outer_frozen": True,
        "margin_is_sole_border_owner": True,
        "clinical_final_ready": False,
    }
    if not (np.isfinite(transition_mm) and 0 < transition_mm <= 2.0):
        report["reason"] = "INVALID_TRANSITION_MM"
        return None, report
    original = np.asarray(accepted_outer.vertices, float)
    try:
        h = axial_height_from_margin(original, margin, frame)
        outer_clip, clipping = _triangle_clip_above(
            accepted_outer, h, level=float(transition_mm)
        )
        report["outer_clip"] = clipping
        outer_loops, outer_diag = _boundary_loops(outer_clip)
        inner_loops, inner_diag = _boundary_loops(owned_inner)
        report["outer_boundary"] = outer_diag
        report["inner_boundary"] = inner_diag
        if len(outer_loops) != 1 or len(inner_loops) != 1:
            report["reason"] = "CERVICAL_LOOP_COUNT_OR_TOPOLOGY_INVALID"
            return None, report
        outer_pts = np.asarray(outer_clip.vertices, float)[outer_loops[0]]
        inner_pts = np.asarray(owned_inner.vertices, float)[inner_loops[0]]
        if not _simple_xy_ring(outer_pts, frame) or not _simple_xy_ring(inner_pts, frame):
            report["reason"] = "CERVICAL_LOOP_NOT_SIMPLE_IN_PATIENT_FRAME"
            return None, report
        outer_dist = _polyline_distance(outer_pts, margin)
        inner_dist = _polyline_distance(inner_pts, margin)
        report["ring_to_margin_mm"] = {
            "outer_median": float(np.median(outer_dist)),
            "outer_p95": float(np.quantile(outer_dist, .95)),
            "inner_median": float(np.median(inner_dist)),
            "inner_p95": float(np.quantile(inner_dist, .95)),
        }
        source_o = _separation_stats(owned_inner, outer_pts)
        source_i = _separation_stats(outer_clip, inner_pts)
        report["outer_ring_to_inner_surface_mm"] = source_o
        report["inner_ring_to_outer_surface_mm"] = source_i
        collapsed = any(
            stats["p05"] < .02 and stats["fraction_below_0_03"] > .20
            for stats in (source_o, source_i)
        )
        far_margin = bool(
            report["ring_to_margin_mm"]["outer_p95"] > 3.5
            or report["ring_to_margin_mm"]["inner_p95"] > 3.5
        )
        if collapsed:
            report["reason"] = "CERVICAL_SOURCE_COLLAPSE"
            return outer_clip, report
        if far_margin:
            report["reason"] = "CERVICAL_RING_TOO_FAR_FROM_MARGIN"
            return outer_clip, report
        if not outer_clip.is_winding_consistent or not owned_inner.is_winding_consistent:
            report["reason"] = "CERVICAL_SOURCE_WINDING_INCONSISTENT"
            return outer_clip, report
        report["status"] = "PASS_ENGINEERING"
        report["reason"] = "SOURCE_GEOMETRY_FEASIBLE_NO_COLLAR_GENERATED"
        return outer_clip, report
    except (ValueError, RuntimeError, IndexError, FloatingPointError) as exc:
        report["reason"] = type(exc).__name__
        report["error"] = str(exc)[:500]
        return None, report
