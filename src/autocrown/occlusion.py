from __future__ import annotations

import numpy as np
import trimesh

from .config import OcclusalConfig
from .frame import PatientFrame, axial_height_from_margin
from .meshops import (
    cap_vectors,
    closest_points,
    diffuse_displacement,
    oriented_face_normals_toward,
)


def adapt_occlusion(mesh: trimesh.Trimesh, antagonist: trimesh.Trimesh,
                    frame: PatientFrame, margin: np.ndarray, *, cervical_lock_mm: float,
                    config: OcclusalConfig | None = None):
    """Only relieve too-close / penetrating occlusal regions.

    The MVP does not pull a low cusp upward toward the antagonist. It only moves
    crown vertices that violate the requested clearance. This is intentionally
    asymmetric: under-contour is preferred over destructive over-contour.
    """
    cfg = config or OcclusalConfig()
    current = mesh.copy()
    vertices = np.asarray(current.vertices, float)
    height = axial_height_from_margin(vertices, margin, frame)
    unlocked = height > float(cervical_lock_mm)
    if int(unlocked.sum()) < 8:
        return current, {"status": "NOT_ASSESSED", "reason": "NO_OCCLUSAL_DOMAIN"}

    hvals = height[unlocked]
    threshold = float(np.quantile(hvals, cfg.top_quantile))
    domain = unlocked & (height >= threshold)
    ids = np.flatnonzero(domain)
    if len(ids) < 8:
        return current, {"status": "NOT_ASSESSED", "reason": "OCCLUSAL_DOMAIN_TOO_SMALL"}

    crown_center = np.median(vertices, axis=0)
    cp, dist, fids = closest_points(antagonist, vertices[ids])
    normals = oriented_face_normals_toward(antagonist, fids, cp, crown_center)
    signed = np.einsum("ij,ij->i", vertices[ids] - cp, normals)
    finite = np.isfinite(signed) & np.isfinite(cp).all(axis=1)
    if int(finite.sum()) < 8:
        return current, {"status": "NOT_ASSESSED", "reason": "NO_FINITE_OCCLUSAL_QUERY"}

    target = float(cfg.target_gap_mm)
    tol = float(cfg.tolerance_mm)
    violating = finite & (signed < target - tol)
    local_ids = np.flatnonzero(violating)
    if not len(local_ids):
        return current, {
            "status": "NO_CORRECTION_NEEDED",
            "method": "OCCLUSAL_PROJECTION_BACK_ONLY_V1",
            "target_gap_mm": target,
            "domain_vertices": int(len(ids)),
            "signed_gap_mm": {
                "min": float(np.min(signed[finite])),
                "median": float(np.median(signed[finite])),
                "p95": float(np.quantile(signed[finite], 0.95)),
            },
        }

    seed_ids = ids[local_ids]
    desired = cp[local_ids] + normals[local_ids] * target
    raw_delta = desired - vertices[seed_ids]
    capped = cap_vectors(raw_delta, cfg.max_relief_mm)

    seed_disp = np.zeros_like(vertices)
    seed_disp[seed_ids] = capped
    seed_mask = np.zeros(len(vertices), bool)
    seed_mask[seed_ids] = True
    locked = height <= float(cervical_lock_mm)
    editable = domain & ~locked
    disp = diffuse_displacement(
        current, seed_disp, seed_mask, editable, locked,
        iterations=cfg.smooth_iterations, alpha=cfg.smooth_alpha,
    )
    moved = np.linalg.norm(disp, axis=1) > 1e-8
    new_vertices = vertices + disp
    new_vertices[locked] = vertices[locked]
    current.vertices = new_vertices

    return current, {
        "status": "APPLIED",
        "method": "OCCLUSAL_PROJECTION_BACK_ONLY_V1",
        "target_gap_mm": target,
        "domain_vertices": int(len(ids)),
        "violating_seed_vertices": int(len(seed_ids)),
        "moved_vertices": int(moved.sum()),
        "max_displacement_mm": float(np.linalg.norm(disp, axis=1).max(initial=0.0)),
        "signed_gap_before_mm": {
            "min": float(np.min(signed[finite])),
            "median": float(np.median(signed[finite])),
            "p95": float(np.quantile(signed[finite], 0.95)),
        },
    }
