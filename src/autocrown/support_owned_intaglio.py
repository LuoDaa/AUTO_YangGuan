"""Read-only support-owned clinical intaglio candidate extraction.

CrownFlow reference:
  backend/app/geometry/canonical_intaglio.py
  backend/app/geometry/crown_outer/collar_body_outer.py

The cutter is an engineering construction solid. To promote its surface to an
inner candidate, prove which supramarginal connected component covers the
preparation. The confirmed 3D Margin owns the boundary, but must NEVER decide
which component is clinical merely by nearest distance.

This is deliberately NOT a completed crown: no cervical collar is built, no
Boolean final surface is released, and no clinical insertion/fit claim is made.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh
from trimesh.proximity import closest_point

from .frame import PatientFrame, axial_height_from_margin


def _triangle_clip_above(
    source: trimesh.Trimesh,
    scalar: np.ndarray,
    *,
    level: float,
) -> tuple[trimesh.Trimesh, dict]:
    """Clip full triangles at h >= level with conforming edge intersections.

    Shared source edges must share the same new vertex. Dropping vertices or
    clipping by triangle center produces cracks, false loops and clinical gaps.
    """
    vertices = np.asarray(source.vertices, float)
    faces = np.asarray(source.faces, np.int64)
    h = np.asarray(scalar, float)
    if len(h) != len(vertices) or not np.isfinite(h).all():
        raise ValueError("INVALID_MARGIN_SCALAR")
    additions = [tuple(p) for p in vertices]
    kept = h >= float(level)
    edge_cache: dict[tuple[int, int], int] = {}
    out = []

    def edge(a: int, b: int) -> int:
        if abs(h[a] - level) < 1e-11:
            return a
        if abs(h[b] - level) < 1e-11:
            return b
        key = (min(a, b), max(a, b))
        if key in edge_cache:
            return edge_cache[key]
        denominator = h[b] - h[a]
        if abs(denominator) < 1e-14:
            raise ValueError("SCALAR_EDGE_PARALLEL_WITH_OPPOSITE_CLASSIFICATION")
        t = float(np.clip((float(level) - h[a]) / denominator, 0.0, 1.0))
        q = vertices[a] + t * (vertices[b] - vertices[a])
        idx = len(additions)
        additions.append(tuple(q))
        edge_cache[key] = idx
        return idx

    for tri in faces:
        a, b, c = map(int, tri)
        corners = (a, b, c)
        flags = (bool(kept[a]), bool(kept[b]), bool(kept[c]))
        count = sum(flags)
        if count == 0:
            continue
        if count == 3:
            out.append((a, b, c))
            continue
        if count == 1:
            index = flags.index(True)
            i, j, k = (corners[(index + offset) % 3] for offset in range(3))
            out.append((i, edge(i, j), edge(k, i)))
        else:
            index = flags.index(False)
            i, j, k = (corners[(index + offset) % 3] for offset in range(3))
            p, q = edge(i, j), edge(k, i)
            out.extend(((j, k, q), (j, q, p)))

    if not out:
        raise ValueError("NO_TRIANGLES_ABOVE_MARGIN")
    new_vertices = np.asarray(additions, float)
    out_faces = np.asarray(out, np.int64)
    distinct = np.all(np.diff(np.sort(out_faces, axis=1), axis=1) > 0, axis=1)
    out_faces = out_faces[distinct]
    if not len(out_faces):
        raise ValueError("NO_NONDEGENERATE_TRIANGLES_ABOVE_MARGIN")
    mesh = trimesh.Trimesh(vertices=new_vertices, faces=out_faces, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh, {
        "input_faces": int(len(faces)),
        "clipped_faces": int(len(mesh.faces)),
        "inserted_vertices": int(len(edge_cache)),
    }


def _boundary_loops(mesh: trimesh.Trimesh) -> tuple[list[np.ndarray], dict]:
    """Require exact degree-two boundary graph; never silently repair a T junction."""
    unique, count = np.unique(np.asarray(mesh.edges_sorted, np.int64), axis=0, return_counts=True)
    if np.any(count > 2):
        return [], {"status": "NONMANIFOLD_SOURCE", "edges_over_two": int(np.sum(count > 2))}
    edges = unique[count == 1]
    graph: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        graph[int(a)].append(int(b))
        graph[int(b)].append(int(a))
    if not graph:
        return [], {"status": "NO_OPEN_BOUNDARY", "boundary_edges": 0}
    if any(len(neigh) != 2 for neigh in graph.values()):
        return [], {
            "status": "BRANCHED_OR_OPEN_BOUNDARY",
            "boundary_edges": int(len(edges)),
            "non_degree_two": int(sum(len(neigh) != 2 for neigh in graph.values())),
        }
    loops: list[np.ndarray] = []
    visited = set()
    for start in sorted(graph):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev, cur = -1, start
        for _ in range(len(graph) + 1):
            a, b = graph[cur]
            nxt = a if a != prev else b
            if nxt == start:
                break
            if nxt in visited:
                return [], {"status": "BOUNDARY_CHAIN_REVISIT", "boundary_edges": int(len(edges))}
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        else:
            return [], {"status": "BOUNDARY_CHAIN_UNCLOSED", "boundary_edges": int(len(edges))}
        if len(loop) < 8:
            return [], {"status": "BOUNDARY_RING_TOO_SHORT", "size": len(loop)}
        loops.append(np.asarray(loop, np.int64))
    return loops, {"status": "OK", "boundary_edges": int(len(edges)), "loop_count": len(loops)}


def _support_regions(support: trimesh.Trimesh, margin: np.ndarray,
                     frame: PatientFrame, limit=384) -> tuple[dict, dict]:
    # Face centers are needed: on height-field support meshes, the original
    # vertices may occupy only the top/bottom while true axial walls lie between.
    points = np.vstack([
        np.asarray(support.vertices, float),
        np.asarray(support.triangles_center, float),
    ])
    height = axial_height_from_margin(points, margin, frame)
    positive = np.isfinite(height) & (height >= 0.05)
    points, height = points[positive], height[positive]
    if len(points) < 12:
        return {}, {"status": "SUPPORT_INSUFFICIENT_ABOVE_MARGIN", "samples": int(len(points))}
    # Exclude the flat artificial support closure below the clinical Margin.
    h_lo, h_hi = np.quantile(height, (0.02, 0.98))
    span = max(float(h_hi - h_lo), 1e-9)
    t = np.clip((height - h_lo) / span, 0.0, 1.0)
    regions = {}
    for name, mask in (
        ("cervical", t <= 0.28),
        ("axial", (t > 0.28) & (t < 0.74)),
        ("top", t >= 0.74),
    ):
        selected = points[mask]
        if len(selected) > limit:
            selected = selected[np.linspace(0, len(selected)-1, limit, dtype=int)]
        regions[name] = selected
    return regions, {
        "status": "OK",
        "sample_count": int(len(points)),
        "axial_height_span_mm": float(span),
        "region_samples": {k: int(len(v)) for k, v in regions.items()},
    }


def _coverage(mesh: trimesh.Trimesh, points: np.ndarray,
              threshold_mm: float) -> dict:
    if not len(points):
        return {"samples": 0, "coverage": None, "median_mm": None, "p95_mm": None}
    _, dist, _ = closest_point(mesh, points)
    dist = np.asarray(dist, float)
    finite = np.isfinite(dist)
    if not finite.any():
        return {"samples": len(points), "coverage": 0.0, "median_mm": None, "p95_mm": None}
    return {
        "samples": int(len(points)),
        "coverage": float(np.sum(finite & (dist <= threshold_mm)) / len(points)),
        "median_mm": float(np.median(dist[finite])),
        "p95_mm": float(np.quantile(dist[finite], .95)),
    }


def extract_support_owned_candidate(
    cutter: trimesh.Trimesh,
    support: trimesh.Trimesh,
    margin: np.ndarray,
    frame: PatientFrame,
    *,
    transition_mm: float = 0.35,
    cement_gap_mm: float = 0.06,
) -> tuple[trimesh.Trimesh | None, dict]:
    """Extract a reviewable inner candidate, or fail closed with per-part proof.

    The coverage window follows CrownFlow's conservative component ownership
    discriminator (0.45–0.90 mm); it is NOT a cement-gap quality tolerance.
    """
    report = {
        "status": "REVIEW_REQUIRED",
        "method": "CROWNFLOW_SUPPORT_OWNED_INTAGLIO_COMPONENT_V1",
        "owner": "EXTRACTED_SUPPORT_COMPONENT",
        "margin_role": "BOUNDARY_ONLY_NOT_COMPONENT_OWNER",
        "clinical_final_ready": False,
        "transition_mm": float(transition_mm),
        "components": [],
    }
    if not (0 < transition_mm <= 2.0 and 0 < cement_gap_mm <= 0.5):
        report["reason"] = "INVALID_EXTRACTION_PARAMETERS"
        return None, report
    regions, support_diag = _support_regions(support, margin, frame)
    report["support"] = support_diag
    if support_diag["status"] != "OK" or any(len(p) < 8 for p in regions.values()):
        report["reason"] = "INSUFFICIENT_SUPPORT_CERVICAL_AXIAL_OR_TOP_EVIDENCE"
        return None, report
    scalar = axial_height_from_margin(cutter.vertices, margin, frame)
    try:
        clipped, clipping = _triangle_clip_above(cutter, scalar, level=float(transition_mm))
    except ValueError as exc:
        report["reason"] = str(exc)
        return None, report
    report["clipping"] = clipping
    components = [
        part for part in clipped.split(only_watertight=False)
        if len(part.faces) >= 8
    ]
    report["component_count"] = len(components)
    if not components:
        report["reason"] = "NO_CLIPPED_COMPONENTS"
        return None, report
    total_area = float(sum(float(part.area) for part in components))
    nominal_gap = max(float(cement_gap_mm), 0.06)
    ownership_mm = float(np.clip(max(0.45, nominal_gap * 4 + 0.15), 0.45, 0.90))
    report["ownership_window_mm"] = ownership_mm
    candidates = []
    for index, part in enumerate(components):
        loops, boundary = _boundary_loops(part)
        row = {
            "component": index, "faces": int(len(part.faces)),
            "area_mm2": float(part.area),
            "area_fraction": float(part.area) / max(total_area, 1e-9),
            "boundary": boundary,
            "eligible": False, "hard_valid": False,
        }
        if boundary["status"] != "OK" or len(loops) != 1:
            row["reason"] = "NOT_SINGLE_SIMPLE_CERVICAL_RING"
            candidates.append(row)
            continue
        span = float(np.ptp(np.asarray(part.vertices) @ np.asarray(frame.z)))
        row["axial_span_fraction"] = span / max(
            float(support_diag["axial_height_span_mm"]), 1e-8
        )
        cover = {name: _coverage(part, pts, ownership_mm)
                 for name, pts in regions.items()}
        row["support_coverage"] = cover
        axial_cov = cover["axial"]["coverage"] or 0.0
        top_cov = cover["top"]["coverage"] or 0.0
        all_coverage = sum(
            cover[k]["coverage"] * cover[k]["samples"] for k in regions
        ) / max(sum(cover[k]["samples"] for k in regions), 1)
        row["total_coverage"] = all_coverage
        row["eligible"] = True
        row["hard_valid"] = bool(
            all_coverage >= .40 and axial_cov >= .35 and top_cov >= .20
            and row["axial_span_fraction"] >= .45
            and row["area_fraction"] >= .20
        )
        row["rank"] = [
            int(row["hard_valid"]),
            min(cover[k]["coverage"] for k in regions),
            all_coverage,
            min(row["axial_span_fraction"], 1.5),
            row["area_fraction"],
        ]
        candidates.append(row)
    report["components"] = candidates
    validated = [(tuple(c["rank"]), idx) for idx, c in enumerate(candidates)
                 if c.get("hard_valid")]
    if not validated:
        report["reason"] = "NO_SUPPORT_OWNED_SINGLE_RING_COMPONENT"
        return None, report
    validated.sort(reverse=True)
    best = validated[0][1]
    selected = components[best].copy()
    selected.invert()  # intaglio normal points into the empty cavity
    report.update({
        "status": "INNER_CANDIDATE_ENGINEERING_ONLY",
        "selected_component": best,
        "selected_area_fraction": candidates[best]["area_fraction"],
        "selected_hard_valid": True,
        "clinical_intaglio_proven": False,
        "clinical_final_ready": False,
    })
    # Multi-component ambiguity is not silently resolved by a weak area prior.
    if len(validated) > 1:
        report["status"] = "REVIEW_REQUIRED"
        report["reason"] = "MULTIPLE_SUPPORT_OWNED_COMPONENTS"
        return None, report
    return selected, report
