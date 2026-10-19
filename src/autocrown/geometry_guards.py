"""Non-destructive, stage-local anatomy/topology acceptance for the 1.0 outer.

CrownFlow's mature principle is to preserve an already-valid anatomical body:
a contact or occlusal solver may *propose* coordinates, but downstream code only
receives them after an independent read-only mesh gate.  This module does not
smooth, Boolean, or repair any input mesh, and never changes topology.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .frame import PatientFrame, axial_height_from_margin


def _face_vectors(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(vertices, float)[np.asarray(faces, np.int64)]
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def _audit_geometry(before: trimesh.Trimesh, proposed: trimesh.Trimesh,
                    margin: np.ndarray, frame: PatientFrame, *,
                    cervical_lock_mm: float, max_stage_move_mm: float) -> dict:
    a = np.asarray(before.vertices, float)
    b = np.asarray(proposed.vertices, float)
    fa = np.asarray(before.faces, np.int64)
    fb = np.asarray(proposed.faces, np.int64)
    reasons = []
    if a.shape != b.shape or fa.shape != fb.shape or not np.array_equal(fa, fb):
        return {"status": "REVIEW_REQUIRED", "reasons": ["TOPOLOGY_CHANGED"]}
    if not np.isfinite(b).all():
        return {"status": "REVIEW_REQUIRED", "reasons": ["NONFINITE_VERTICES"]}
    if len(a) == 0 or len(fa) == 0:
        return {"status": "REVIEW_REQUIRED", "reasons": ["EMPTY_MESH"]}
    delta = np.linalg.norm(b - a, axis=1)
    max_move = float(delta.max(initial=0.0))
    if max_move > float(max_stage_move_mm) + 1e-6:
        reasons.append("STAGE_DISPLACEMENT_LIMIT")
    cervical = axial_height_from_margin(a, margin, frame) <= float(cervical_lock_mm)
    cervical_move = float(delta[cervical].max(initial=0.0))
    if cervical_move > 1e-6:
        reasons.append("CONFIRMED_CERVICAL_BAND_MOVED")

    original_face = _face_vectors(a, fa)
    current_face = _face_vectors(b, fa)
    old_norm = np.linalg.norm(original_face, axis=1)
    new_norm = np.linalg.norm(current_face, axis=1)
    valid_old = old_norm > 2e-8
    degenerated = valid_old & (new_norm < np.maximum(2e-8, old_norm * 0.07))
    new_degenerate = int(np.sum(degenerated))
    if new_degenerate:
        reasons.append("NEW_DEGENERATE_FACES")
    dot = np.einsum("ij,ij->i", original_face, current_face)
    flipped = valid_old & (dot <= -1e-11)
    face_flips = int(np.sum(flipped))
    if face_flips:
        reasons.append("FACE_NORMAL_REVERSED")

    return {
        "status": "PASS_ENGINEERING" if not reasons else "REVIEW_REQUIRED",
        "reasons": reasons,
        "max_displacement_mm": max_move,
        "p95_displacement_mm": float(np.quantile(delta, 0.95)),
        "cervical_locked_vertices": int(cervical.sum()),
        "max_cervical_displacement_mm": cervical_move,
        "new_degenerate_faces": new_degenerate,
        "face_normal_reversals": face_flips,
        "same_faces_and_vertex_indices": True,
    }


def accept_anatomy_stage(
    before: trimesh.Trimesh,
    proposed: trimesh.Trimesh,
    margin: np.ndarray,
    frame: PatientFrame,
    *,
    stage: str,
    cervical_lock_mm: float,
    max_stage_move_mm: float,
) -> tuple[trimesh.Trimesh, dict]:
    """Accept, damp, or reject a proposed deformation; never mutate its source.

    A rejected candidate returns the untouched previous stage instead of
    contaminating later contact/occlusion, and is *never* reported as successful.
    Accepted backoffs preserve a strict subset of the solver's displacement.
    """
    a = np.asarray(before.vertices, float)
    b = np.asarray(proposed.vertices, float)
    full = _audit_geometry(
        before, proposed, margin, frame,
        cervical_lock_mm=cervical_lock_mm, max_stage_move_mm=max_stage_move_mm,
    )
    if full["status"] == "PASS_ENGINEERING":
        return proposed, {
            "status": "ACCEPTED",
            "stage": stage,
            "method": "FROZEN_CERVICAL_MORPHOLOGY_GATE_V1",
            "accepted_fraction": 1.0,
            "full_candidate": full,
        }
    if a.shape != b.shape or not np.isfinite(b).all():
        return before.copy(), {
            "status": "REJECTED",
            "stage": stage,
            "method": "FROZEN_CERVICAL_MORPHOLOGY_GATE_V1",
            "reason": "UNSAFE_CANDIDATE",
            "full_candidate": full,
        }
    # Geometry-only line search, not a second unconstrained morph solver.
    for fraction in (0.5, 0.25, 0.125, 0.0625):
        retry = before.copy()
        retry.vertices = a + float(fraction) * (b - a)
        check = _audit_geometry(
            before, retry, margin, frame,
            cervical_lock_mm=cervical_lock_mm,
            max_stage_move_mm=max_stage_move_mm,
        )
        if check["status"] == "PASS_ENGINEERING":
            return retry, {
                "status": "ACCEPTED_BACKOFF",
                "stage": stage,
                "method": "FROZEN_CERVICAL_MORPHOLOGY_GATE_V1",
                "accepted_fraction": fraction,
                "full_candidate": full,
                "accepted_geometry": check,
            }
    return before.copy(), {
        "status": "REJECTED",
        "stage": stage,
        "method": "FROZEN_CERVICAL_MORPHOLOGY_GATE_V1",
        "reason": "NO_SAFE_DEFORMATION_FRACTION",
        "full_candidate": full,
    }
