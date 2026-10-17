from __future__ import annotations

import numpy as np
import trimesh

from .config import ProximalConfig
from .frame import PatientFrame, axial_height_from_margin
from .meshops import (
    cap_vectors,
    closest_points,
    diffuse_displacement,
    oriented_face_normals_toward,
)


def _domain(mesh: trimesh.Trimesh, frame: PatientFrame, margin: np.ndarray,
            side: str, lock_mm: float, quantile: float, mesial_sign: float = 1.0):
    local = frame.to_local(mesh.vertices)
    height = axial_height_from_margin(mesh.vertices, margin, frame)
    unlocked = height > float(lock_mm)
    x = local[:, 0] * float(mesial_sign)
    if not np.any(unlocked):
        return np.zeros(len(mesh.vertices), bool), height
    values = x[unlocked]
    if side == "mesial":
        threshold = float(np.quantile(values, quantile))
        mask = unlocked & (x >= threshold)
    elif side == "distal":
        threshold = float(np.quantile(values, 1.0 - quantile))
        mask = unlocked & (x <= threshold)
    else:
        raise ValueError("side must be mesial or distal")
    return mask, height


def adapt_proximal(mesh: trimesh.Trimesh, neighbor: trimesh.Trimesh,
                   frame: PatientFrame, margin: np.ndarray, side: str,
                   *, cervical_lock_mm: float,
                   config: ProximalConfig | None = None,
                   mesial_sign: float = 1.0):
    """Small local growth + deterministic projection-back against one neighbour.

    ``frame.x`` is a right-handed arch tangent, not assumed to point mesially on
    every quadrant. ``mesial_sign`` maps that tangent to the FDI mesial direction
    without ever flipping the buccal axis of the patient frame.
    """
    cfg = config or ProximalConfig()
    current = mesh.copy()
    domain, height = _domain(
        current, frame, margin, side, cervical_lock_mm, cfg.domain_quantile,
        mesial_sign=mesial_sign,
    )
    ids = np.flatnonzero(domain)
    if len(ids) < 8:
        return current, {"status": "NOT_ASSESSED", "reason": "PROXIMAL_DOMAIN_TOO_SMALL", "side": side}

    vertices = np.asarray(current.vertices, float)
    crown_center = np.median(vertices, axis=0)
    cp, dist, fids = closest_points(neighbor, vertices[ids])
    normals = oriented_face_normals_toward(neighbor, fids, cp, crown_center)
    signed = np.einsum("ij,ij->i", vertices[ids] - cp, normals)
    finite = np.isfinite(signed) & np.isfinite(cp).all(axis=1)
    if int(finite.sum()) < 8:
        return current, {"status": "NOT_ASSESSED", "reason": "NO_FINITE_PROXIMAL_QUERY", "side": side}

    target = float(cfg.target_gap_mm)
    tol = float(cfg.tolerance_mm)
    desired = cp + normals * target
    raw_delta = desired - vertices[ids]
    too_close = finite & (signed < target - tol)
    too_far = finite & (signed > target + tol)

    far_local = np.flatnonzero(too_far)
    growth_local = np.zeros(len(ids), bool)
    if len(far_local):
        order = far_local[np.argsort(signed[far_local])]
        take = max(int(cfg.min_contact_vertices), int(np.ceil(len(ids) * cfg.contact_fraction)))
        take = min(take, len(order))
        growth_local[order[:take]] = True

    relief_local = too_close
    seed_local = growth_local | relief_local
    seed_ids = ids[seed_local]
    if not len(seed_ids):
        report = {
            "status": "NO_CORRECTION_NEEDED",
            "side": side,
            "mesial_sign": float(mesial_sign),
            "target_gap_mm": target,
            "domain_vertices": int(len(ids)),
            "signed_gap_mm": {
                "min": float(np.min(signed[finite])),
                "median": float(np.median(signed[finite])),
                "p95": float(np.quantile(signed[finite], 0.95)),
            },
        }
        return current, report

    seed_disp = np.zeros_like(vertices)
    local_delta = raw_delta[seed_local]
    growth_seed = growth_local[seed_local]
    relief_seed = relief_local[seed_local]
    capped = local_delta.copy()
    if np.any(growth_seed):
        capped[growth_seed] = cap_vectors(capped[growth_seed], cfg.max_growth_mm)
    if np.any(relief_seed):
        capped[relief_seed] = cap_vectors(capped[relief_seed], cfg.max_relief_mm)
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
        "side": side,
        "mesial_sign": float(mesial_sign),
        "method": "COMPACT_GROWTH_PLUS_PROJECTION_BACK_V2",
        "target_gap_mm": target,
        "domain_vertices": int(len(ids)),
        "growth_seed_vertices": int(growth_local.sum()),
        "relief_seed_vertices": int(relief_local.sum()),
        "moved_vertices": int(moved.sum()),
        "max_displacement_mm": float(np.linalg.norm(disp, axis=1).max(initial=0.0)),
        "signed_gap_before_mm": {
            "min": float(np.min(signed[finite])),
            "median": float(np.median(signed[finite])),
            "p95": float(np.quantile(signed[finite], 0.95)),
        },
    }
