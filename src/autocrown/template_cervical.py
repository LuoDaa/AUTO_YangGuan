"""Crown-only surface extraction from an STL-only registered natural-tooth library.

Unlike CrownFlow's labeled EOFF templates, these source STLs do not contain a
confirmed cervical vertex-label loop. A centroid-based triangle discard can
leave ragged spikes or multiple root fragments. This routine instead cuts
*triangles* at a verified crown-root level and insists on one clinical open rim.

A plane is only used to clean the unlabelled *template prior*: the patient
Margin is still the non-planar confirmed/observed curve and is never replaced.
"""
from __future__ import annotations

import numpy as np
import trimesh


def _boundary_loops(mesh: trimesh.Trimesh) -> list[list[int]] | None:
    edges = np.asarray(mesh.edges_sorted, dtype=np.int64)
    if not len(edges):
        return None
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        return None
    boundary = unique[counts == 1]
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if not adjacency or any(len(v) != 2 for v in adjacency.values()):
        return None
    loops = []
    visited: set[int] = set()
    for seed in adjacency:
        if seed in visited:
            continue
        ring, prev, cur = [], None, seed
        while cur not in visited:
            visited.add(cur)
            ring.append(cur)
            a, b = adjacency[cur]
            nxt = a if a != prev else b
            prev, cur = cur, nxt
        if cur != seed or len(ring) < 8:
            return None
        loops.append(ring)
    return loops


def extract_clean_crown_surface(
    template: trimesh.Trimesh,
    axis: np.ndarray,
    center: np.ndarray,
    proposed_cut: float,
) -> tuple[trimesh.Trimesh | None, dict]:
    """Attempt smallest supracervical change yielding a complete, single rim.

    The caller may retain its previous mesh only as an explicitly REVIEW_REQUIRED
    debug candidate when no valid topology is proven; it must never treat an
    unverified open surface as a production-ready crown.
    """
    axis = np.asarray(axis, float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    center = np.asarray(center, float)
    original_area = float(template.area)
    attempts = []
    for lift in (0.0, 0.15, 0.30, 0.50, 0.75, 1.00):
        origin = center + axis * (float(proposed_cut) + lift)
        try:
            sliced = trimesh.intersections.slice_mesh_plane(
                template, plane_normal=axis, plane_origin=origin, cap=False
            )
            if sliced is None or not len(sliced.faces):
                attempts.append({"cut_lift_mm": lift, "reason": "NO_FACES"})
                continue
            sliced.merge_vertices()
            sliced.remove_unreferenced_vertices()
            components = list(sliced.split(only_watertight=False))
            if not components:
                attempts.append({"cut_lift_mm": lift, "reason": "NO_COMPONENT"})
                continue
            components.sort(key=lambda x: float(x.area), reverse=True)
            candidate = components[0]
            retained = float(candidate.area) / max(float(sliced.area), 1e-8)
            loops = _boundary_loops(candidate)
            sane = bool(
                retained >= 0.97
                and loops is not None and len(loops) == 1
                and len(loops[0]) >= 24
                and len(candidate.faces) >= max(100, int(0.14 * len(template.faces)))
                and candidate.is_winding_consistent
            )
            row = {
                "cut_lift_mm": float(lift),
                "components": int(len(components)),
                "selected_area_fraction": retained,
                "boundary_loops": None if loops is None else len(loops),
                "boundary_vertices": None if not loops else len(loops[0]),
                "candidate_faces": int(len(candidate.faces)),
                "winding_consistent": bool(candidate.is_winding_consistent),
                "status": "PASS_ENGINEERING" if sane else "REVIEW_REQUIRED",
            }
            attempts.append(row)
            if sane:
                pts = np.asarray(candidate.vertices, float)[np.asarray(loops[0], int)]
                residual = np.abs((pts - origin) @ axis)
                return candidate, {
                    "status": "PASS_ENGINEERING",
                    "method": "TRIANGLE_LEVEL_CROWN_ROOT_SPLIT_WITH_SINGLE_RIM_V1",
                    "cut_lift_mm": float(lift),
                    "source_faces": int(len(template.faces)),
                    "selected_faces": int(len(candidate.faces)),
                    "boundary_vertices": int(len(loops[0])),
                    "boundary_axial_residual_max_mm": float(residual.max(initial=0.0)),
                    "crown_source_area_fraction": float(candidate.area) / max(original_area, 1e-8),
                    "attempts": attempts,
                }
        except Exception as exc:
            attempts.append({"cut_lift_mm": float(lift), "reason": str(exc)[:160]})
    return None, {
        "status": "REVIEW_REQUIRED",
        "method": "TRIANGLE_LEVEL_CROWN_ROOT_SPLIT_WITH_SINGLE_RIM_V1",
        "reason": "NO_PROVEN_SINGLE_CERVICAL_RIM_IN_UNLABELLED_STL",
        "attempts": attempts,
    }
