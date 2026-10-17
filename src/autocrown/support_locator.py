from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import trimesh
from scipy.spatial import cKDTree


class SupportLocatorError(ValueError):
    pass


def _unit(value, name="vector"):
    value = np.asarray(value, float)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or norm < 1e-10:
        raise SupportLocatorError(f"{name} is degenerate")
    return value / norm


def _sample_vertices(mesh: trimesh.Trimesh, maximum: int = 24000):
    vertices = np.asarray(mesh.vertices, float)
    if len(vertices) <= maximum:
        return vertices
    step = max(1, len(vertices) // maximum)
    return vertices[::step][:maximum]


def _distance_to_mesh_vertices(mesh: trimesh.Trimesh, points: np.ndarray):
    tree = cKDTree(_sample_vertices(mesh))
    return np.asarray(tree.query(np.asarray(points, float), k=1)[0], float)


def _main_jaw_score(mesh: trimesh.Trimesh):
    """Prefer the arch-sized component without depending on tessellation density."""
    bounds = np.asarray(mesh.bounds, float)
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    sorted_extent = np.sort(extent)
    footprint = float(sorted_extent[-1] * sorted_extent[-2])
    area = max(float(mesh.area), 1e-6)
    span = float(np.linalg.norm(extent))
    # Working arch has large surface area and a broad 2-D footprint.  This is
    # deliberately independent of face count because scanner tessellation varies.
    return math.log1p(area) + 0.9 * math.log1p(footprint) + 0.25 * math.log1p(span)


def _solid_range(values: np.ndarray):
    values = np.asarray(values, float)
    if len(values) < 8:
        return float(values.min()), float(values.max())
    counts, edges = np.histogram(values, bins=24)
    occupied = np.flatnonzero(counts >= max(6, 0.15 * counts.max()))
    if len(occupied) < 3:
        return float(np.quantile(values, 0.02)), float(np.quantile(values, 0.98))
    return float(edges[occupied[0]]), float(edges[occupied[-1] + 1])


def _occlusal_reference(jaw: trimesh.Trimesh, opponent: trimesh.Trimesh | None):
    if opponent is None:
        return None
    delta = np.asarray(opponent.centroid, float) - np.asarray(jaw.centroid, float)
    try:
        return _unit(delta, "occlusal reference")
    except SupportLocatorError:
        return None


def _long_pca_axis(mesh: trimesh.Trimesh):
    vertices = np.asarray(mesh.vertices, float)
    values, vectors = np.linalg.eigh(np.cov((vertices - vertices.mean(axis=0)).T))
    return _unit(vectors[:, int(np.argmax(values))], "PCA long axis")


def _normal_covariance_axis(mesh: trimesh.Trimesh, reference: np.ndarray):
    """Fit support axis from side-wall normals, using a rough occlusal reference.

    For a tapered/cylindrical preparation, valid wall normals are approximately
    perpendicular to insertion.  The minimum-eigenvalue direction of their
    area-weighted normal covariance is therefore the least-squares insertion axis.
    """
    normals = np.asarray(mesh.face_normals, float)
    areas = np.asarray(mesh.area_faces, float)
    centers = np.asarray(mesh.triangles_center, float)
    reference = _unit(reference, "wall reference")
    axis = reference.copy()
    report = {}

    for iteration in range(5):
        projection = centers @ axis
        lo, hi = np.quantile(projection, [0.04, 0.96])
        body = (projection >= lo) & (projection <= hi)
        # Exclude obvious top/bottom caps.  The relaxed limit allows tapered walls.
        wall = body & (np.abs(normals @ axis) < 0.76) & (areas > 1e-10)
        if int(wall.sum()) < 18:
            wall = body & (np.abs(normals @ axis) < 0.88) & (areas > 1e-10)
        if int(wall.sum()) < 12:
            break
        n = normals[wall]
        w = areas[wall]
        cov = (n * w[:, None]).T @ n / max(float(w.sum()), 1e-12)
        values, vectors = np.linalg.eigh(cov)
        revised = _unit(vectors[:, int(np.argmin(values))], "normal covariance axis")
        if float(revised @ reference) < 0:
            revised = -revised
        # Do not allow a noisy wall fit to jump to an orthogonal radial direction.
        alignment = abs(float(revised @ reference))
        if alignment < math.cos(math.radians(55.0)):
            report["rejected_iteration"] = iteration
            report["rejected_alignment"] = alignment
            break
        blended = _unit(0.20 * reference + 0.80 * revised, "blended wall axis")
        change = math.degrees(math.acos(float(np.clip(abs(blended @ axis), -1.0, 1.0))))
        axis = blended
        report = {
            "iteration": iteration + 1,
            "wall_face_count": int(wall.sum()),
            "wall_area_fraction": float(areas[wall].sum() / max(float(areas.sum()), 1e-12)),
            "normal_cov_eigenvalues": [float(x) for x in values],
            "axis_change_deg": float(change),
            "axis_alignment_to_reference": float(abs(axis @ reference)),
        }
        if change < 0.12:
            break
    return axis, report


def _candidate_axis(mesh: trimesh.Trimesh, jaw: trimesh.Trimesh, opponent: trimesh.Trimesh | None):
    pca = _long_pca_axis(mesh)
    occlusal = _occlusal_reference(jaw, opponent)
    if occlusal is not None:
        if float(pca @ occlusal) < 0:
            pca = -pca
        # In a bite-related scan the jaw-to-opponent direction is a stronger
        # rough reference than support PCA, while PCA still contributes tilt.
        alignment = abs(float(pca @ occlusal))
        if alignment >= math.cos(math.radians(50.0)):
            reference = _unit(0.70 * occlusal + 0.30 * pca, "rough support reference")
            ref_source = "OPPONENT_PLUS_PCA"
        else:
            reference = occlusal
            ref_source = "OPPONENT_DOMINANT_PCA_DISAGREES"
    else:
        reference = pca
        ref_source = "PCA_NO_OPPONENT"

    axis, wall = _normal_covariance_axis(mesh, reference)
    if occlusal is not None and float(axis @ occlusal) < 0:
        axis = -axis
    return axis, {
        "reference_source": ref_source,
        "pca_axis": pca.tolist(),
        "occlusal_reference": None if occlusal is None else occlusal.tolist(),
        "wall_fit": wall,
        "axis_alignment_to_occlusal": None if occlusal is None else float(axis @ occlusal),
    }


def _endpoint_orientation(component, jaw, opponent, axis):
    vertices = np.asarray(component.vertices, float)
    center = vertices.mean(axis=0)
    axis = _unit(axis)
    projection = (vertices - center) @ axis
    lo, hi = _solid_range(projection)
    length = max(hi - lo, 1e-6)
    cap = max(0.45, 0.10 * length)
    bottom = vertices[(projection >= lo) & (projection <= lo + cap)]
    top = vertices[(projection >= hi - cap) & (projection <= hi)]
    if len(bottom) < 4 or len(top) < 4:
        qlo, qhi = np.quantile(projection, [0.08, 0.92])
        bottom = vertices[projection <= qlo]
        top = vertices[projection >= qhi]
    bcenter = bottom.mean(axis=0)
    tcenter = top.mean(axis=0)
    bd = float(np.median(_distance_to_mesh_vertices(jaw, bottom)))
    td = float(np.median(_distance_to_mesh_vertices(jaw, top)))

    method = "JAW_ENDPOINT_PROXIMITY"
    if bd > td:
        axis = -axis
        bcenter, tcenter = tcenter, bcenter
        bd, td = td, bd

    occlusal = _occlusal_reference(jaw, opponent)
    if occlusal is not None and float(axis @ occlusal) < 0:
        # Opponent direction is an orientation constraint.  Only the line was
        # estimated above; reversing it does not alter axis fitting.
        axis = -axis
        bcenter, tcenter = tcenter, bcenter
        method = "OPPONENT_ORIENTED"
    return axis, bcenter, tcenter, {
        "method": method,
        "base_to_jaw_median_mm": bd,
        "top_to_jaw_median_mm": td,
        "robust_length_mm": float(length),
        "axis_alignment_to_opponent": None if occlusal is None else float(axis @ occlusal),
    }


def _cross_section_metrics(mesh: trimesh.Trimesh, axis: np.ndarray):
    vertices = np.asarray(mesh.vertices, float)
    center = vertices.mean(axis=0)
    axis = _unit(axis)
    projection = (vertices - center) @ axis
    lo, hi = _solid_range(projection)
    span = max(hi - lo, 1e-6)
    radial = np.linalg.norm(vertices - center - np.outer(projection, axis), axis=1)
    body = (projection >= lo) & (projection <= hi)
    radius = float(np.median(radial[body])) if body.any() else float(np.median(radial))
    spread = float(np.std(radial[body])) if body.any() else float(np.std(radial))
    return radius, spread, span


def _rotational_symmetry(mesh: trimesh.Trimesh, axis: np.ndarray):
    points = _sample_vertices(mesh, 900)
    center = points.mean(axis=0)
    tree = cKDTree(points)
    errors = []
    for degrees in (90.0, 180.0):
        angle = math.radians(degrees)
        rel = points - center
        rotated = (
            center
            + rel * math.cos(angle)
            + np.cross(axis, rel) * math.sin(angle)
            + np.outer(rel @ axis, axis) * (1.0 - math.cos(angle))
        )
        errors.append(float(np.median(tree.query(rotated, k=1)[0])))
    return float(np.mean(errors))


@dataclass
class SupportLocalization:
    jaw: trimesh.Trimesh
    support: trimesh.Trimesh
    rough_axis: np.ndarray
    base_center: np.ndarray
    top_center: np.ndarray
    report: dict


def localize_support(target_arch: trimesh.Trimesh, opponent: trimesh.Trimesh | None = None):
    parts = list(target_arch.split(only_watertight=False))
    if len(parts) < 2:
        raise SupportLocatorError("working jaw has no disconnected support component")

    main_index = int(np.argmax([_main_jaw_score(part) for part in parts]))
    jaw = parts[main_index]
    jaw_extent = np.asarray(jaw.bounds[1] - jaw.bounds[0], float)
    jaw_scale = max(float(np.linalg.norm(jaw_extent)), 1.0)
    rows = []

    for index, component in enumerate(parts):
        if index == main_index:
            continue
        if len(component.faces) < 80 or len(component.vertices) < 45:
            continue
        extent = np.asarray(component.bounds[1] - component.bounds[0], float)
        if float(np.linalg.norm(extent)) > 0.65 * jaw_scale:
            continue
        try:
            axis, axis_report = _candidate_axis(component, jaw, opponent)
            axis, base, top, orientation = _endpoint_orientation(component, jaw, opponent, axis)
        except Exception:
            continue
        radius, spread, length = _cross_section_metrics(component, axis)
        samples = _sample_vertices(component, 650)
        distance = _distance_to_mesh_vertices(jaw, samples)
        dmin = float(distance.min())
        dmed = float(np.median(distance))
        symmetry = _rotational_symmetry(component, axis)
        radius_cv = spread / max(radius, 1e-6)
        occl_align = axis_report.get("axis_alignment_to_occlusal")

        score = 0.0
        if 0.8 <= radius <= 6.0:
            score += 1.0
        if 2.0 <= length <= 18.0:
            score += 1.0
        score += 1.1 * math.exp(-min(radius_cv, 3.0))
        score += 1.2 * math.exp(-min(symmetry / max(radius, 0.5), 5.0))
        score += 1.8 * math.exp(-dmin / 0.45)
        score += 0.8 * math.exp(-dmed / 2.5)
        wall_fraction = float((axis_report.get("wall_fit") or {}).get("wall_area_fraction", 0.0))
        score += 1.4 * min(wall_fraction / 0.45, 1.3)
        if occl_align is not None:
            score += 1.1 * max(0.0, float(occl_align))

        rows.append({
            "index": int(index),
            "mesh": component,
            "score": float(score),
            "axis": axis,
            "base_center": base,
            "top_center": top,
            "surface_area_mm2": float(component.area),
            "faces": int(len(component.faces)),
            "vertices": int(len(component.vertices)),
            "radius_mm": float(radius),
            "radius_cv": float(radius_cv),
            "robust_length_mm": float(length),
            "jaw_distance_min_mm": dmin,
            "jaw_distance_median_mm": dmed,
            "rotational_symmetry_proxy_mm": symmetry,
            "axis_fit": axis_report,
            "orientation": orientation,
        })

    if not rows:
        raise SupportLocatorError("no plausible disconnected restorative support found")
    rows.sort(key=lambda row: row["score"], reverse=True)
    first = rows[0]
    second = rows[1] if len(rows) > 1 else None
    gap = None if second is None else float(first["score"] - second["score"])
    confidence = "high" if second is None or gap >= 0.75 else "review"
    public = [
        {k: v for k, v in row.items() if k not in {"mesh", "axis", "base_center", "top_center"}}
        for row in rows
    ]
    report = {
        "method": "ARCH_AREA_SUPPORT_RANK_WALL_NORMAL_V3",
        "status": "READY" if confidence == "high" else "REVIEW_REQUIRED",
        "confidence": confidence,
        "component_count": int(len(parts)),
        "main_jaw_component_index": main_index,
        "selected_component_index": int(first["index"]),
        "selected_score": float(first["score"]),
        "score_gap_to_second": gap,
        "rough_axis": np.asarray(first["axis"], float).tolist(),
        "base_center": np.asarray(first["base_center"], float).tolist(),
        "top_center": np.asarray(first["top_center"], float).tolist(),
        "candidates": public,
    }
    return SupportLocalization(
        jaw=jaw.copy(),
        support=first["mesh"].copy(),
        rough_axis=np.asarray(first["axis"], float),
        base_center=np.asarray(first["base_center"], float),
        top_center=np.asarray(first["top_center"], float),
        report=report,
    )
