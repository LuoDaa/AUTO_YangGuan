from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


class LocalizationError(ValueError):
    pass


def _unit(v, name="vector") -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if v.shape != (3,) or not np.isfinite(v).all() or n < 1e-10:
        raise LocalizationError(f"{name} is degenerate")
    return v / n


def _sample_vertices(mesh: trimesh.Trimesh, max_points: int = 1000) -> np.ndarray:
    v = np.asarray(mesh.vertices, float)
    if len(v) <= max_points:
        return v
    step = max(1, len(v) // max_points)
    return v[::step][:max_points]


def _surface_distance(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    # Localization needs a robust proximity cue, not exact manufacturing distance.
    # Vertex KD-tree keeps this independent of optional rtree/ray backends.
    tree = cKDTree(_sample_vertices(mesh, 30000))
    return np.asarray(tree.query(np.asarray(points, float), k=1)[0], float)


def _solid_range(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, float)
    if len(values) < 8:
        return float(values.min()), float(values.max())
    hist, edges = np.histogram(values, bins=24)
    occupied = np.flatnonzero(hist >= max(6, 0.18 * hist.max()))
    if len(occupied) < 3:
        return float(np.quantile(values, 0.02)), float(np.quantile(values, 0.98))
    return float(edges[occupied[0]]), float(edges[occupied[-1] + 1])


def _fit_wall_axis(mesh: trimesh.Trimesh, initial: np.ndarray) -> tuple[np.ndarray, dict]:
    vertices = np.asarray(mesh.vertices, float)
    normals = np.asarray(mesh.vertex_normals, float)
    axis = _unit(initial, "initial axis")
    converged = False

    for _ in range(18):
        wall = np.abs(normals @ axis) < 0.42
        if int(wall.sum()) < max(24, int(0.05 * len(vertices))):
            break
        points = vertices[wall]
        h = (points - points.mean(axis=0)) @ axis
        lo, hi = np.quantile(h, [0.05, 0.95])
        points = points[(h >= lo) & (h <= hi)]
        if len(points) < 24:
            break
        h = (points - points.mean(axis=0)) @ axis
        edges = np.linspace(float(h.min()), float(h.max()), 8)
        centers = []
        for i in range(len(edges) - 1):
            mask = (h >= edges[i]) & (h < edges[i + 1])
            if int(mask.sum()) >= 8:
                centers.append(points[mask].mean(axis=0))
        if len(centers) < 3:
            break
        _, _, vh = np.linalg.svd(np.asarray(centers) - np.mean(centers, axis=0), full_matrices=False)
        revised = _unit(vh[0], "wall axis")
        if float(revised @ axis) < 0:
            revised = -revised
        delta = math.degrees(math.acos(float(np.clip(revised @ axis, -1.0, 1.0))))
        axis = revised
        if delta < 0.15:
            converged = True
            break

    center = vertices.mean(axis=0)
    proj = (vertices - center) @ axis
    radius = np.linalg.norm(vertices - center - np.outer(proj, axis), axis=1)
    wall = np.abs(normals @ axis) < 0.42
    wr = radius[wall]
    return axis, {
        "wall_vertex_ratio": float(wall.mean()) if len(wall) else 0.0,
        "radius_mm": float(np.median(wr)) if len(wr) else float(np.median(radius)),
        "radius_std_mm": float(np.std(wr)) if len(wr) else float(np.std(radius)),
        "converged": bool(converged),
    }


def _best_component_axis(mesh: trimesh.Trimesh) -> tuple[np.ndarray, dict]:
    v = np.asarray(mesh.vertices, float)
    _, vecs = np.linalg.eigh(np.cov(v.T))
    candidates = []
    for i in range(3):
        try:
            axis, report = _fit_wall_axis(mesh, vecs[:, i])
        except LocalizationError:
            continue
        radius = max(report["radius_mm"], 1e-6)
        quality = 1.8 * report["wall_vertex_ratio"] - 0.8 * min(report["radius_std_mm"] / radius, 2.0)
        candidates.append((quality, axis, report))
    if not candidates:
        raise LocalizationError("cannot estimate abutment wall axis")
    candidates.sort(key=lambda x: x[0], reverse=True)
    quality, axis, report = candidates[0]
    return axis, {**report, "axis_fit_quality": float(quality)}


def _orient_axis(
    component: trimesh.Trimesh,
    jaw: trimesh.Trimesh,
    opponent: trimesh.Trimesh | None,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    v = np.asarray(component.vertices, float)
    center = v.mean(axis=0)
    axis = _unit(axis)
    h = (v - center) @ axis
    lo, hi = _solid_range(h)
    span = max(hi - lo, 1e-6)
    cap = max(0.55, span * 0.10)
    low = v[(h >= lo) & (h <= lo + cap)]
    high = v[(h >= hi - cap) & (h <= hi)]
    if len(low) < 4 or len(high) < 4:
        qlo, qhi = np.quantile(h, [0.08, 0.92])
        low = v[h <= qlo]
        high = v[h >= qhi]
    low_center = low.mean(axis=0)
    high_center = high.mean(axis=0)
    low_d = float(np.median(_surface_distance(jaw, low)))
    high_d = float(np.median(_surface_distance(jaw, high)))

    method = "JAW_ENDPOINT_PROXIMITY"
    if low_d <= high_d:
        base_center, top_center = low_center, high_center
    else:
        axis = -axis
        base_center, top_center = high_center, low_center
        low_d, high_d = high_d, low_d

    occlusal_alignment = None
    if opponent is not None:
        occlusal = _unit(np.asarray(opponent.centroid, float) - np.asarray(jaw.centroid, float), "occlusal direction")
        if float(axis @ occlusal) < 0 and abs(high_d - low_d) < 0.20:
            axis = -axis
            base_center, top_center = top_center, base_center
            method = "OPPONENT_TIE_BREAK"
        occlusal_alignment = float(axis @ occlusal)

    return axis, base_center, top_center, {
        "method": method,
        "base_to_jaw_median_mm": float(low_d),
        "top_to_jaw_median_mm": float(high_d),
        "axis_alignment_to_opponent": occlusal_alignment,
        "robust_length_mm": float(span),
    }


def _rotational_symmetry_proxy(mesh: trimesh.Trimesh, axis: np.ndarray, center: np.ndarray) -> float:
    pts = _sample_vertices(mesh, 700)
    tree = cKDTree(pts)
    errors = []
    for degrees in (60.0, 120.0, 180.0):
        a = math.radians(degrees)
        rel = pts - center
        rotated = (
            center
            + rel * math.cos(a)
            + np.cross(axis, rel) * math.sin(a)
            + np.outer(rel @ axis, axis) * (1.0 - math.cos(a))
        )
        errors.append(float(np.median(tree.query(rotated, k=1)[0])))
    return float(np.mean(errors))


@dataclass
class AbutmentLocalization:
    jaw: trimesh.Trimesh
    support: trimesh.Trimesh
    rough_axis: np.ndarray
    base_center: np.ndarray
    top_center: np.ndarray
    report: dict


def localize_abutment(
    target_arch: trimesh.Trimesh,
    opponent: trimesh.Trimesh | None = None,
) -> AbutmentLocalization:
    parts = sorted(list(target_arch.split(only_watertight=False)), key=lambda m: len(m.faces), reverse=True)
    if len(parts) < 2:
        raise LocalizationError(
            "working jaw has no disconnected support component; manual/ROI support extraction is required"
        )
    jaw = parts[0]
    main_faces = max(len(jaw.faces), 1)
    rows = []

    for index, component in enumerate(parts[1:10], start=1):
        if len(component.faces) < 120 or len(component.faces) > 0.72 * main_faces:
            continue
        v = np.asarray(component.vertices, float)
        if len(v) < 60 or not np.isfinite(v).all():
            continue
        try:
            raw_axis, axis_report = _best_component_axis(component)
            axis, base, top, orient = _orient_axis(component, jaw, opponent, raw_axis)
        except Exception:
            continue

        center = v.mean(axis=0)
        h = (v - center) @ axis
        lo, hi = _solid_range(h)
        span = max(hi - lo, 1e-6)
        radius = max(float(axis_report["radius_mm"]), 1e-6)
        radius_cv = float(axis_report["radius_std_mm"]) / radius
        samples = _sample_vertices(component, 600)
        distance = _surface_distance(jaw, samples)
        dmin = float(np.min(distance))
        dmed = float(np.median(distance))
        symmetry = _rotational_symmetry_proxy(component, axis, center)

        # Soft ranking, not a hard implant identity classifier. Restorative
        # abutments may be tapered/asymmetric; multiple independent cues rank them.
        score = 0.0
        score += 2.0 * min(axis_report["wall_vertex_ratio"] / 0.35, 1.5)
        score += 1.4 * math.exp(-min(radius_cv, 2.0))
        score += 1.2 * math.exp(-min(symmetry / max(radius, 0.5), 4.0))
        score += 1.7 * math.exp(-dmin / 0.35)
        score += 0.8 * math.exp(-dmed / 2.0)
        if 1.0 <= radius <= 5.5:
            score += 0.8
        if 2.5 <= span <= 18.0:
            score += 0.8
        if orient.get("axis_alignment_to_opponent") is not None:
            score += 0.5 * max(0.0, float(orient["axis_alignment_to_opponent"]))

        rows.append({
            "index": index,
            "mesh": component,
            "score": float(score),
            "axis": axis,
            "base_center": base,
            "top_center": top,
            "faces": int(len(component.faces)),
            "vertices": int(len(component.vertices)),
            "radius_mm": float(radius),
            "radius_cv": float(radius_cv),
            "robust_length_mm": float(span),
            "jaw_distance_min_mm": dmin,
            "jaw_distance_median_mm": dmed,
            "rotational_symmetry_proxy_mm": symmetry,
            "axis_fit": axis_report,
            "orientation": orient,
        })

    if not rows:
        raise LocalizationError("no plausible disconnected abutment/support component found")
    rows.sort(key=lambda r: r["score"], reverse=True)
    selected = rows[0]
    second_score = rows[1]["score"] if len(rows) > 1 else None
    confidence = "high" if second_score is None or selected["score"] >= second_score + 0.9 else "review"

    public_candidates = [
        {k: v for k, v in r.items() if k not in {"mesh", "axis", "base_center", "top_center"}}
        for r in rows
    ]
    report = {
        "method": "COMPONENT_RANK_WALL_AXIS_JAW_PROXIMITY_V2",
        "status": "READY" if confidence == "high" else "REVIEW_REQUIRED",
        "confidence": confidence,
        "component_count": int(len(parts)),
        "selected_component_index": int(selected["index"]),
        "selected_score": float(selected["score"]),
        "rough_axis": np.asarray(selected["axis"], float).tolist(),
        "base_center": np.asarray(selected["base_center"], float).tolist(),
        "top_center": np.asarray(selected["top_center"], float).tolist(),
        "candidates": public_candidates,
    }
    return AbutmentLocalization(
        jaw=jaw,
        support=selected["mesh"].copy(),
        rough_axis=np.asarray(selected["axis"], float),
        base_center=np.asarray(selected["base_center"], float),
        top_center=np.asarray(selected["top_center"], float),
        report=report,
    )


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _unit(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(helper @ axis)) > 0.85:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(axis, helper))
    e2 = _unit(np.cross(axis, e1))
    return e1, e2


def _vertex_sharpness(mesh: trimesh.Trimesh) -> np.ndarray:
    out = np.zeros(len(mesh.vertices), float)
    edges = np.asarray(mesh.face_adjacency_edges, int)
    if not len(edges):
        return out
    angles = np.asarray(mesh.face_adjacency_angles, float)
    np.maximum.at(out, edges[:, 0], angles)
    np.maximum.at(out, edges[:, 1], angles)
    return out


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lo, hi = np.quantile(x[finite], [0.05, 0.95])
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def detect_finish_line(
    support: trimesh.Trimesh,
    rough_axis: np.ndarray,
    *,
    sectors: int = 180,
) -> tuple[np.ndarray, dict]:
    """Detect a deterministic engineering finish-line candidate on support."""
    v = np.asarray(support.vertices, float)
    center = v.mean(axis=0)
    axis = _unit(rough_axis)
    e1, e2 = _basis(axis)
    rel = v - center
    z = rel @ axis
    x = rel @ e1
    y = rel @ e2
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    normals = np.asarray(support.vertex_normals, float)
    normal_axis = np.abs(normals @ axis)
    sharp = _vertex_sharpness(support)

    zmin, zmax = np.quantile(z, [0.01, 0.99])
    height = max(float(zmax - zmin), 1e-6)
    bins = 90
    edges = np.linspace(float(zmin), float(zmax), bins + 1)
    mids = (edges[:-1] + edges[1:]) * 0.5
    rp = np.full(bins, np.nan)
    sp = np.full(bins, np.nan)
    npf = np.full(bins, np.nan)
    for i in range(bins):
        mask = (z >= edges[i]) & (z < edges[i + 1])
        if int(mask.sum()) < 5:
            continue
        rp[i] = np.quantile(r[mask], 0.90)
        sp[i] = np.quantile(sharp[mask], 0.90)
        npf[i] = np.quantile(normal_axis[mask], 0.75)

    def fill(a):
        a = a.copy()
        good = np.isfinite(a)
        if good.sum() < 2:
            return np.nan_to_num(a)
        ids = np.arange(len(a))
        a[~good] = np.interp(ids[~good], ids[good], a[good])
        return a

    rp = gaussian_filter1d(fill(rp), 2.0)
    sp = gaussian_filter1d(fill(sp), 1.4)
    npf = gaussian_filter1d(fill(npf), 1.4)
    dr = np.gradient(rp, mids)
    d2r = np.gradient(dr, mids)
    score = (
        1.30 * _normalize01(np.abs(d2r))
        + 1.00 * _normalize01(sp)
        + 0.45 * _normalize01(npf)
        + 0.35 * _normalize01(rp)
    )
    zn = (mids - zmin) / height
    allowed = (zn >= 0.04) & (zn <= 0.62)
    restricted = np.where(allowed, score, -1e9)
    best = int(np.argmax(restricted))
    shoulder_center = float(mids[best])
    width = float(np.clip(height * 0.14, 0.55, 1.50))
    low, high = shoulder_center - width / 2.0, shoulder_center + width / 2.0

    band = (z >= low) & (z <= high)
    ids = np.flatnonzero(band)
    if len(ids) < max(30, sectors // 2):
        raise LocalizationError("finish-line shoulder band has insufficient support samples")
    outer_threshold = float(np.quantile(r[ids], 0.30))
    ids = np.flatnonzero(band & (r >= outer_threshold))
    if len(ids) < 30:
        raise LocalizationError("finish-line outer shoulder samples are insufficient")

    dtheta = 2.0 * np.pi / sectors
    chosen = []
    chosen_scores = []
    for i in range(sectors):
        angle = -np.pi + (i + 0.5) * dtheta
        diff = np.abs(np.angle(np.exp(1j * (theta[ids] - angle))))
        local = ids[diff <= dtheta * 2.2]
        if len(local) < 2:
            local = ids[np.argsort(diff)[: min(8, len(ids))]]
        if not len(local):
            continue
        rr = r[local]
        ss = np.degrees(sharp[local])
        nn = normal_axis[local]
        zz = z[local]
        rr_score = _normalize01(rr)
        sharp_score = np.clip(ss / 45.0, 0.0, 1.0)
        z_score = np.exp(-0.5 * ((zz - shoulder_center) / max(width * 0.55, 1e-4)) ** 2)
        total = 0.50 * rr_score + 0.30 * sharp_score + 0.12 * nn + 0.08 * z_score
        j = int(np.argmax(total))
        chosen.append(v[int(local[j])])
        chosen_scores.append(float(total[j]))

    if len(chosen) < int(0.80 * sectors):
        raise LocalizationError("finish-line angular coverage is incomplete")

    chosen = np.asarray(chosen, float)
    local = np.column_stack(((chosen - center) @ e1, (chosen - center) @ e2, (chosen - center) @ axis))
    angles = np.unwrap(np.arctan2(local[:, 1], local[:, 0]))
    order = np.argsort(angles)
    local = local[order]
    angles = angles[order]
    angles0 = angles - angles[0]
    period = 2.0 * np.pi
    angles_ext = np.r_[angles0, period]
    r0 = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
    z0 = local[:, 2]
    r_ext = np.r_[r0, r0[0]]
    z_ext = np.r_[z0, z0[0]]
    targets = np.linspace(0.0, period, sectors, endpoint=False)
    rr = np.interp(targets, angles_ext, r_ext)
    zz = np.interp(targets, angles_ext, z_ext)
    rr = gaussian_filter1d(rr, 1.3, mode="wrap")
    zz = gaussian_filter1d(zz, 1.8, mode="wrap")
    curve = (
        center[None, :]
        + rr[:, None] * np.cos(targets)[:, None] * e1[None, :]
        + rr[:, None] * np.sin(targets)[:, None] * e2[None, :]
        + zz[:, None] * axis[None, :]
    )
    tree = cKDTree(v)
    distance, nearest = tree.query(curve, k=1)
    margin = v[np.asarray(nearest, int)]
    keep = np.ones(len(margin), bool)
    keep[1:] = np.linalg.norm(np.diff(margin, axis=0), axis=1) > 1e-6
    margin = margin[keep]
    if len(margin) < 64:
        raise LocalizationError("finish-line projection collapsed to too few unique samples")

    report = {
        "method": "SHOULDER_PROFILE_ANGULAR_LOOP_V2",
        "status": "ENGINEERING_AUTO_REVIEW",
        "shoulder_center_axial_mm": shoulder_center,
        "shoulder_band_mm": [float(low), float(high)],
        "support_height_mm": float(height),
        "point_count": int(len(margin)),
        "projection_vertex_distance_mm": {
            "median": float(np.median(distance)),
            "p95": float(np.quantile(distance, 0.95)),
            "max": float(np.max(distance)),
        },
        "candidate_score_median": float(np.median(chosen_scores)),
    }
    return margin, report


def _robust_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, float)
    mask = np.ones(len(pts), bool)
    for _ in range(3):
        sub = pts[mask]
        center = sub.mean(axis=0)
        _, _, vh = np.linalg.svd(sub - center, full_matrices=False)
        normal = _unit(vh[-1])
        residual = np.abs((pts - center) @ normal)
        threshold = max(float(np.quantile(residual[mask], 0.82)), 1e-6)
        new = residual <= threshold
        if new.sum() < 8 or np.array_equal(new, mask):
            break
        mask = new
    sub = pts[mask]
    center = sub.mean(axis=0)
    _, _, vh = np.linalg.svd(sub - center, full_matrices=False)
    return center, _unit(vh[-1])


def _wall_normal_axis(support: trimesh.Trimesh, reference: np.ndarray, margin_center: np.ndarray) -> tuple[np.ndarray, dict]:
    axis = _unit(reference)
    centers = np.asarray(support.triangles_center, float)
    normals = np.asarray(support.face_normals, float)
    areas = np.asarray(support.area_faces, float)
    selected_count = 0
    quality = 0.0
    for _ in range(4):
        h = (centers - margin_center) @ axis
        wall = (h > 0.05) & (np.abs(normals @ axis) < 0.72) & (areas > 1e-10)
        selected_count = int(wall.sum())
        if selected_count < 20:
            break
        n = normals[wall]
        w = areas[wall]
        cov = (n * w[:, None]).T @ n / max(float(w.sum()), 1e-12)
        vals, vecs = np.linalg.eigh(cov)
        revised = _unit(vecs[:, int(np.argmin(vals))])
        if float(revised @ axis) < 0:
            revised = -revised
        quality = float(1.0 - vals.min() / max(vals.max(), 1e-12))
        axis = _unit(0.35 * axis + 0.65 * revised)
    return axis, {"wall_face_count": selected_count, "wall_axis_quality": quality}


def _cone_candidates(axis: np.ndarray, max_deg: float = 20.0) -> list[np.ndarray]:
    axis = _unit(axis)
    e1, e2 = _basis(axis)
    out = [axis]
    for deg in (4.0, 8.0, 12.0, 16.0, max_deg):
        r = math.radians(deg)
        for k in range(20):
            phi = 2.0 * math.pi * k / 20.0
            tangent = math.cos(phi) * e1 + math.sin(phi) * e2
            out.append(_unit(math.cos(r) * axis + math.sin(r) * tangent))
    return out


def solve_insertion_axis(
    support: trimesh.Trimesh,
    margin: np.ndarray,
    rough_axis: np.ndarray,
) -> tuple[np.ndarray, dict]:
    margin = np.asarray(margin, float)
    center, plane_axis = _robust_plane(margin)
    rough = _unit(rough_axis)
    if float(plane_axis @ rough) < 0:
        plane_axis = -plane_axis
    wall_axis, wall_report = _wall_normal_axis(support, plane_axis, center)
    if float(wall_axis @ plane_axis) < 0:
        wall_axis = -wall_axis
    disagreement = math.degrees(math.acos(float(np.clip(plane_axis @ wall_axis, -1.0, 1.0))))
    initial = _unit(0.58 * plane_axis + 0.42 * wall_axis) if disagreement <= 35.0 else plane_axis

    centers = np.asarray(support.triangles_center, float)
    normals = np.asarray(support.face_normals, float)
    areas = np.asarray(support.area_faces, float)
    draft_tol = math.sin(math.radians(1.5))

    rows = []
    for candidate in _cone_candidates(initial, 20.0):
        h = (centers - center) @ candidate
        region = (h > 0.05) & (areas > 1e-10)
        if int(region.sum()) < 20:
            continue
        ndot = normals[region] @ candidate
        aa = areas[region]
        under = ndot < -draft_tol
        ratio = float(aa[under].sum() / max(float(aa.sum()), 1e-12))
        severity = float(np.average(np.clip(-ndot, 0.0, 1.0), weights=aa))
        dev = math.degrees(math.acos(float(np.clip(candidate @ initial, -1.0, 1.0))))
        objective = 1.00 * ratio + 0.22 * severity + 0.14 * (dev / 20.0)
        rows.append((objective, candidate, ratio, severity, dev))

    if not rows:
        raise LocalizationError("insertion-axis search produced no valid candidate")
    rows.sort(key=lambda row: row[0])
    _, axis, undercut_ratio, severity, deviation = rows[0]
    scores = np.asarray([r[0] for r in rows], float)
    spread = float(np.quantile(scores, 0.75) - scores.min()) if len(scores) > 3 else 0.0
    status = "OK"
    reasons = []
    if disagreement > 35.0:
        status = "REVIEW_REQUIRED"
        reasons.append("MARGIN_WALL_AXIS_DISAGREEMENT")
    if undercut_ratio > 0.18:
        status = "REVIEW_REQUIRED"
        reasons.append("HIGH_UNDERCUT_RATIO")
    if spread < 0.01:
        status = "REVIEW_REQUIRED"
        reasons.append("OBJECTIVE_FLAT_AXIS_WEAKLY_CONSTRAINED")
    confidence = float(np.clip(
        0.95
        - max(0.0, disagreement - 12.0) / 70.0
        - max(0.0, undercut_ratio - 0.08)
        - (0.18 if spread < 0.01 else 0.0),
        0.05,
        0.98,
    ))
    report = {
        "method": "MARGIN_PLANE_PLUS_WALL_NORMAL_CONE_SEARCH_V2",
        "status": status,
        "reasons": reasons,
        "axis": axis.tolist(),
        "initial_axis": initial.tolist(),
        "margin_plane_axis": plane_axis.tolist(),
        "wall_axis": wall_axis.tolist(),
        "margin_wall_disagreement_deg": float(disagreement),
        "undercut_area_ratio_proxy": float(undercut_ratio),
        "draft_severity_proxy": float(severity),
        "deviation_from_initial_deg": float(deviation),
        "objective_spread_p75_minus_best": spread,
        "confidence": confidence,
        **wall_report,
    }
    return axis, report


def build_automatic_patient_frame(
    jaw: trimesh.Trimesh,
    margin: np.ndarray,
    insertion_axis: np.ndarray,
):
    from .frame import PatientFrame

    origin = np.asarray(margin, float).mean(axis=0)
    z = _unit(insertion_axis)
    vertices = np.asarray(jaw.vertices, float)
    arch_center = np.median(vertices, axis=0)
    radial = origin - arch_center
    radial -= z * float(radial @ z)
    if float(np.linalg.norm(radial)) < 0.5:
        projected = vertices - origin
        projected -= (projected @ z)[:, None] * z
        radial = -projected[int(np.argmax(np.linalg.norm(projected, axis=1)))]
    y = _unit(radial, "automatic buccal")
    x = _unit(np.cross(y, z), "automatic mesiodistal")
    y = _unit(np.cross(z, x), "automatic buccal orthogonalized")
    if float(y @ radial) < 0:
        x = -x
        y = -y
    frame = PatientFrame(origin=origin, x=x, y=y, z=z)
    return frame, {
        "method": "WORKING_ARCH_RADIAL_PLUS_INSERTION_AXIS_V2",
        "origin": origin.tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
        "z": z.tolist(),
        "arch_center": arch_center.tolist(),
        "radial_length_mm": float(np.linalg.norm(radial)),
        "handedness": float(np.linalg.det(frame.matrix)),
    }


def localize_case_geometry(
    target_arch: trimesh.Trimesh,
    opponent: trimesh.Trimesh,
):
    located = localize_abutment(target_arch, opponent)
    margin0, margin0_report = detect_finish_line(located.support, located.rough_axis)
    axis, axis_report = solve_insertion_axis(located.support, margin0, located.rough_axis)
    margin, margin_report = detect_finish_line(located.support, axis)
    axis2, axis_report2 = solve_insertion_axis(located.support, margin, axis)
    frame, frame_report = build_automatic_patient_frame(located.jaw, margin, axis2)
    status = "READY"
    if located.report["status"] != "READY" or axis_report2["status"] != "OK":
        status = "REVIEW_REQUIRED"
    report = {
        "status": status,
        "abutment": located.report,
        "margin_initial": margin0_report,
        "axis_initial": axis_report,
        "margin": margin_report,
        "axis": axis_report2,
        "frame": frame_report,
        "margin_points": np.asarray(margin, float).tolist(),
    }
    return located.jaw, located.support, margin, frame, report
