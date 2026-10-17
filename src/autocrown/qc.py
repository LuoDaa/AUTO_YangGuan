from __future__ import annotations

import numpy as np
import trimesh

from .config import QualityConfig
from .meshops import face_flip_count


def displacement_stats(reference: trimesh.Trimesh, current: trimesh.Trimesh) -> dict:
    if len(reference.vertices) != len(current.vertices):
        return {"status": "NOT_ASSESSED", "reason": "VERTEX_COUNT_CHANGED"}
    delta = np.linalg.norm(
        np.asarray(current.vertices, float) - np.asarray(reference.vertices, float), axis=1
    )
    return {
        "status": "MEASURED",
        "median_mm": float(np.median(delta)),
        "p95_mm": float(np.quantile(delta, 0.95)),
        "max_mm": float(np.max(delta)),
        "moved_fraction_gt_0_05": float(np.mean(delta > 0.05)),
        "moved_fraction_gt_0_20": float(np.mean(delta > 0.20)),
    }


def topology_report(mesh: trimesh.Trimesh) -> dict:
    # `is_watertight` is useful for a crown-only library but is not treated as a
    # universal hard gate because some development templates may still be open.
    e = np.sort(np.asarray(mesh.edges_sorted, int), axis=1)
    if len(e):
        _unique, counts = np.unique(e, axis=0, return_counts=True)
        nonmanifold = int(np.sum(counts > 2))
        boundary = int(np.sum(counts == 1))
    else:
        nonmanifold = boundary = 0
    areas = np.asarray(mesh.area_faces, float)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "nonmanifold_edges": nonmanifold,
        "boundary_edges": boundary,
        "degenerate_faces": int(np.sum(~np.isfinite(areas) | (areas < 1e-10))),
    }


def evaluate_outer(reference: trimesh.Trimesh, current: trimesh.Trimesh,
                   config: QualityConfig | None = None) -> dict:
    cfg = config or QualityConfig()
    disp = displacement_stats(reference, current)
    flips = face_flip_count(reference, current)
    topo = topology_report(current)
    reasons = []
    if topo["nonmanifold_edges"]:
        reasons.append("NONMANIFOLD")
    if topo["degenerate_faces"]:
        reasons.append("DEGENERATE_FACES")
    if flips is not None and flips > cfg.hard_face_flip_limit:
        reasons.append("FACE_FLIPS")
    if disp.get("status") == "MEASURED":
        if disp["max_mm"] > cfg.max_total_displacement_mm:
            reasons.append("MAX_DISPLACEMENT_EXCEEDED")
        elif disp["p95_mm"] > cfg.review_p95_displacement_mm:
            reasons.append("P95_DISPLACEMENT_REVIEW")
    return {
        "status": "PASS_ENGINEERING" if not reasons else "REVIEW_REQUIRED",
        "reasons": reasons,
        "displacement": disp,
        "face_flips": flips,
        "topology": topo,
        "self_intersection": {
            "status": "NOT_ASSESSED",
            "reason": "MVP_HAS_NO_FCL_OR_ROBUST_TRIANGLE_SELF_INTERSECTION_BACKEND",
        },
    }
