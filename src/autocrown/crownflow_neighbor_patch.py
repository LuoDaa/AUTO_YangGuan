"""CrownFlow V5-inspired coronal neighbour extraction for standalone cases.

Provenance: CrownFlow backend/app/api/abutment_fast.py, AUTO neighbour V5.
Unlike a closest-component-only ROI, this extracts a connected coronal contact
core and an adjacent reference halo on the ORIGINAL target-jaw mesh. No jaw
vertices are moved and failure never substitutes a guessed neighbouring tooth.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .frame import PatientFrame


def _face_adjacency(mesh: trimesh.Trimesh) -> list[list[int]]:
    adjacency = [[] for _ in range(len(mesh.faces))]
    for a, b in np.asarray(mesh.face_adjacency, dtype=int):
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    return adjacency


def _connected(adjacency: list[list[int]], seed: int, allowed: np.ndarray) -> np.ndarray:
    if seed < 0 or seed >= len(allowed) or not allowed[seed]:
        return np.zeros(0, np.int64)
    visited = {int(seed)}
    front = [int(seed)]
    while front:
        i = front.pop()
        for j in adjacency[i]:
            if allowed[j] and j not in visited:
                visited.add(j)
                front.append(j)
    return np.asarray(sorted(visited), np.int64)


def _halo(adjacency: list[list[int]], core: np.ndarray, allowed: np.ndarray,
          rings: int = 12) -> np.ndarray:
    seen = set(int(x) for x in core)
    front = set(seen)
    for _ in range(int(rings)):
        next_ring = {j for i in front for j in adjacency[i] if allowed[j] and j not in seen}
        if not next_ring:
            break
        seen.update(next_ring)
        front = next_ring
    return np.asarray(sorted(seen), np.int64)


def extract_neighbor_patch_v5(
    jaw: trimesh.Trimesh, frame: PatientFrame, margin: np.ndarray,
    side: str, *, mesial_sign: float, patch_radius_mm: float = 3.0,
) -> tuple[trimesh.Trimesh | None, dict]:
    """Extract a patient-frame side-owned contiguous proximal reference surface.

    The sign convention is explicit: frame.x is right-handed arch tangent,
    while FDI determines which sign of x is anatomical mesial. Read-only
    extraction; ambiguous faces or tiny areas remain REVIEW_REQUIRED.
    """
    if side not in ("mesial", "distal"):
        raise ValueError("side must be mesial or distal")
    if len(jaw.faces) < 8 or len(margin) < 8:
        return None, {"status": "REVIEW_REQUIRED", "reason": "INVALID_JAW_OR_MARGIN"}
    centers = np.asarray(jaw.triangles_center, float)
    areas = np.asarray(jaw.area_faces, float)
    normals = np.asarray(jaw.face_normals, float)
    local = np.asarray(frame.to_local(centers), float)
    mlocal = np.asarray(frame.to_local(np.asarray(margin, float)), float)
    sign = float(mesial_sign) * (1.0 if side == "mesial" else -1.0)
    side_x = sign * local[:, 0]
    half_x = max(float(np.ptp(mlocal[:, 0])) * 0.5, 0.8)
    half_y = max(float(np.ptp(mlocal[:, 1])) * 0.5, 0.8)
    z_hi = float(np.quantile(mlocal[:, 2], 0.90))
    base = (
        (side_x > half_x + 0.25)
        & (side_x < half_x + 10.0)
        & (np.abs(local[:, 1]) < half_y + 5.5)
        & (areas > 1e-8)
    )
    strict = base & (local[:, 2] > z_hi + 0.5) & (local[:, 2] < z_hi + 6.5)
    relaxed = base & (local[:, 2] > z_hi + 0.15) & (local[:, 2] < z_hi + 8.0)
    presence = (
        (side_x > half_x + 0.15)
        & (side_x < half_x + 11.0)
        & (np.abs(local[:, 1]) < half_y + 6.75)
        & (local[:, 2] > z_hi - 1.25)
        & (local[:, 2] < z_hi + 9.5)
        & (areas > 1e-8)
    )
    candidates = np.flatnonzero(strict)
    mode = "CORONAL_STRICT"
    anatomy = strict
    if len(candidates) < 3:
        candidates, anatomy, mode = np.flatnonzero(relaxed), relaxed, "CORONAL_RELAXED"
    if len(candidates) < 3 and np.count_nonzero(presence) >= 3:
        candidates, anatomy, mode = np.flatnonzero(presence), presence, "PRESENCE_HEIGHT_FALLBACK"
    if not len(candidates):
        return None, {
            "status": "REVIEW_REQUIRED",
            "reason": "NO_CORONAL_NEIGHBOR_EVIDENCE_NOT_PROVEN_ABSENT",
            "side": side, "strict_faces": int(strict.sum()),
            "relaxed_faces": int(relaxed.sum()), "presence_faces": int(presence.sum()),
        }

    near_x = float(np.quantile(side_x[candidates], 0.05))
    bounded = anatomy & (side_x <= near_x + 3.0)
    ids = np.flatnonzero(bounded)
    if len(ids) >= 3:
        candidates = ids
    normal_ref = -sign * np.asarray(frame.x, float)
    facing = normals @ normal_ref
    z_target = z_hi + 3.0
    score = (
        1.5 * np.maximum(side_x[candidates] - near_x, 0.0)
        + 0.11 * np.abs(local[candidates, 1])
        + 0.07 * np.abs(local[candidates, 2] - z_target)
        - 0.20 * np.maximum(facing[candidates], 0.0)
    )
    seed = int(candidates[int(np.argmin(score))])
    distance = np.linalg.norm(centers - centers[seed], axis=1)
    adj = _face_adjacency(jaw)
    core = np.zeros(0, np.int64)
    for scale in (1.0, 1.35, 1.75, 2.25, 3.0):
        mask = bounded & (distance <= float(patch_radius_mm) * scale)
        core = _connected(adj, seed, mask)
        if len(core) >= 8:
            break

    fallback_used = False
    if len(core) < 8:
        # Under a sparse/tilted scan, preserve side ownership while opening the
        # coronal height window. Never cross to the opposite side of the margin.
        broader = presence & (side_x <= near_x + 4.75)
        core = _connected(adj, seed, broader & (distance <= float(patch_radius_mm) * 4.0))
        fallback_used = True
    if len(core) < 8:
        return None, {
            "status": "REVIEW_REQUIRED", "reason": "NO_CONNECTED_CORONAL_CORE",
            "side": side, "candidate_faces": int(len(candidates)),
            "core_faces": int(len(core)), "search_mode": mode,
        }

    inner_floor = max(0.35, half_x - 0.75, near_x - 2.25)
    halo_allowed = (
        (side_x > inner_floor) & (side_x < near_x + 5.5)
        & (np.abs(local[:, 1]) < half_y + 7.0)
        & (local[:, 2] > z_hi - 0.8) & (local[:, 2] < z_hi + 8.75)
        & (areas > 1e-8)
    )
    radial = _connected(
        adj, seed, halo_allowed & (distance <= max(float(patch_radius_mm) + 3, 6.0))
    )
    grown = _halo(adj, core, halo_allowed, rings=14 if fallback_used else 12)
    selected = np.unique(np.r_[core, radial, grown])
    if len(selected) < 8:
        return None, {"status": "REVIEW_REQUIRED", "reason": "REFERENCE_HALO_TOO_SMALL", "side": side}
    region = jaw.submesh([selected], append=True, repair=False)
    region.remove_unreferenced_vertices()
    if len(region.faces) < 8 or float(region.area) < 0.12:
        return None, {"status": "REVIEW_REQUIRED", "reason": "REFERENCE_PATCH_TOO_SMALL", "side": side}
    if len(region.split(only_watertight=False)) != 1:
        return None, {"status": "REVIEW_REQUIRED", "reason": "REFERENCE_PATCH_DISCONNECTED", "side": side}
    return region, {
        "status": "READY", "side": side,
        "method": "CROWNFLOW_CORONAL_CORE_CONNECTED_HALO_V5_STANDALONE",
        "search_mode": mode, "fallback_used": fallback_used,
        "candidate_faces": int(len(candidates)), "core_faces": int(len(core)),
        "reference_faces": int(len(region.faces)),
        "reference_area_mm2": float(region.area),
        "mesial_sign": float(mesial_sign),
        "nearest_side_coordinate_mm": near_x,
        "coronal_margin_reference_z_mm": z_hi,
        "reference_normal_alignment_to_frame": float(np.median(np.abs(facing[core]))),
        "human_reviewed": False,
    }
