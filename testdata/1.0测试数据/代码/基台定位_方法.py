# -*- coding: utf-8 -*-
"""复核三例独立小件的几何定位；库件比对仅作研究证据，不作临床身份判定。"""

import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
LIB = ROOT.parent / "数据" / "1629-王煜-STL" / "04_种植基台与组件"
NEG = [
    ROOT.parent / "数据" / "1629-王煜-STL" / "01_主要病例模型" / "1629-王煜 1.stl",
    ROOT.parent / "数据" / "1629-王煜-STL" / "03_牙冠解剖参考" / "UNN19.stl",
]
FINAL_ABUTMENT_CONTROL = (
    ROOT.parent / "数据" / "1629-王煜-STL" / "01_主要病例模型" / "1629-王煜 0.stl"
)


def unit(v):
    return v / np.linalg.norm(v)


def sample(vertices, count=600):
    rng = np.random.default_rng(20261012)
    return vertices[rng.choice(len(vertices), min(len(vertices), count), replace=False)]


def surface_distance(mesh, points, batch=32):
    return np.concatenate([
        trimesh.proximity.closest_point(mesh, points[i:i + batch])[1]
        for i in range(0, len(points), batch)
    ])


def pca_basis(points):
    _, vectors = np.linalg.eigh(np.cov(points.T))
    basis = vectors[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def rigid_match(source, target):
    """24 个 PCA 朝向的无缩放、无镜像点云 ICP；输出双向残差最小者。"""
    src, dst = sample(source.vertices), sample(target.vertices)
    src_basis, dst_basis = pca_basis(src), pca_basis(dst)
    dst_tree = cKDTree(dst)
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            orient = np.eye(3)[:, perm] @ np.diag(signs)
            rotation = dst_basis @ orient @ src_basis.T
            if np.linalg.det(rotation) < 0.9:
                continue
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = dst.mean(0) - rotation @ src.mean(0)
            moved = trimesh.transform_points(src, matrix)
            previous = np.inf
            for _ in range(35):
                distance, index = dst_tree.query(moved)
                step, moved, _ = trimesh.registration.procrustes(
                    moved, dst[index], scale=False, reflection=False
                )
                matrix = step @ matrix
                cost = float(np.median(distance))
                if abs(previous - cost) < 1e-5:
                    break
                previous = cost
            forward = dst_tree.query(moved)[0]
            reverse = cKDTree(moved).query(dst)[0]
            score = float((np.median(forward) + np.median(reverse)) / 2)
            if best is None or score < best[0]:
                best = (score, matrix)
    matrix = best[1]
    singular = np.linalg.svd(matrix[:3, :3])[1]
    assert np.allclose(singular, 1.0, atol=1e-6)
    assert np.linalg.det(matrix[:3, :3]) > 0.999999
    src_eval, dst_eval = sample(source.vertices, 240), sample(target.vertices, 240)
    forward = surface_distance(target, trimesh.transform_points(src_eval, matrix))
    reverse = surface_distance(source, trimesh.transform_points(dst_eval, np.linalg.inv(matrix)))
    return matrix, {
        "forward_p50_mm": round(float(np.median(forward)), 4),
        "forward_p90_mm": round(float(np.quantile(forward, 0.9)), 4),
        "reverse_p50_mm": round(float(np.median(reverse)), 4),
        "reverse_p90_mm": round(float(np.quantile(reverse, 0.9)), 4),
        "sample_vertices_each": 240,
        "rigid_no_scale_no_reflection": True,
    }


def fit_axis(mesh):
    vertices, normals = mesh.vertices.astype(float), mesh.vertex_normals.astype(float)
    values, vectors = np.linalg.eigh(np.cov(vertices.T))
    axis = unit(vectors[:, np.argmax(values)])
    center = vertices.mean(0)
    for _ in range(25):
        wall = np.abs(normals @ axis) < 0.35
        if wall.sum() < 50:
            break
        points = vertices[wall]
        projection = (points - points.mean(0)) @ axis
        lo, hi = np.quantile(projection, [0.05, 0.95])
        points = points[(projection > lo) & (projection < hi)]
        projection = (points - points.mean(0)) @ axis
        edges = np.linspace(projection.min(), projection.max(), 7)
        centers = [points[(projection >= edges[i]) & (projection < edges[i + 1])].mean(0)
                   for i in range(6)
                   if ((projection >= edges[i]) & (projection < edges[i + 1])).sum() > 20]
        if len(centers) < 3:
            break
        _, _, vectors = np.linalg.svd(np.asarray(centers) - np.mean(centers, axis=0), full_matrices=False)
        axis = unit(vectors[0])
    displacement = vertices - center
    projection = displacement @ axis
    radius = np.linalg.norm(displacement - np.outer(projection, axis), axis=1)
    wall = np.abs(normals @ axis) < 0.35
    return axis, center, float(np.median(radius[wall])), float(radius[wall].std()), float(wall.mean())


def radius_profile(mesh, axis, center, lo, hi, bins=10):
    displacement = mesh.vertices - center
    heights = displacement @ axis
    radius = np.linalg.norm(displacement - np.outer(heights, axis), axis=1)
    wall = np.abs(mesh.vertex_normals @ axis) < 0.35
    edges = np.linspace(lo, hi, bins + 1)
    return np.asarray([
        np.median(radius[wall & (heights >= edges[i]) & (heights < edges[i + 1])])
        if np.sum(wall & (heights >= edges[i]) & (heights < edges[i + 1])) >= 4 else np.nan
        for i in range(bins)
    ])


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    return float(np.interp(quantile, np.cumsum(weights[order]) / weights.sum(), values[order]))


def symmetry(mesh, axis, center):
    vertices = sample(mesh.vertices, 240)
    values = []
    for degrees in (30, 60, 90, 120, 180, 240):
        transform = trimesh.transformations.rotation_matrix(np.radians(degrees), axis, center)
        rotated = mesh.copy()
        rotated.apply_transform(transform)
        d1 = surface_distance(mesh, trimesh.transform_points(vertices, transform))
        d2 = surface_distance(rotated, vertices)
        values.append(float(np.median(np.r_[d1, d2])))
    return round(float(np.mean(values)), 3)


def control_patch(jaw, piece, occlusal):
    vertices = jaw.vertices
    far = np.linalg.norm(vertices - piece.centroid, axis=1) > 12.0
    tip = vertices[np.where(far)[0][np.argmax(vertices[far] @ occlusal)]]
    nearby = np.linalg.norm(vertices - tip, axis=1) < 3.5
    faces = np.where(np.all(nearby[jaw.faces], axis=1))[0]
    patch = jaw.submesh([faces], append=True)
    if len(patch.faces) < 50:
        raise ValueError("同颌对照局部面片过小")
    return patch


def analyze(case_dir, references):
    upper = trimesh.load(case_dir / "upperjaw_upper.stl", process=True)
    lower = trimesh.load(case_dir / "world_result.stl", process=True)
    components = sorted(lower.split(only_watertight=False), key=lambda mesh: -len(mesh.faces))
    if len(components) < 2:
        raise ValueError(f"{case_dir.name}: 无独立小件")
    jaw, piece = components[:2]
    axis, center, radius, radius_std, wall_ratio = fit_axis(piece)
    heights = (piece.vertices - center) @ axis
    surface_heights = (piece.triangles_center - center) @ axis
    bottom = weighted_quantile(surface_heights, piece.area_faces, 0.01)
    top = weighted_quantile(surface_heights, piece.area_faces, 0.99)
    low_cap = piece.vertices[heights < bottom + 0.8]
    high_cap = piece.vertices[heights > top - 0.8]
    low_distance = float(np.median(surface_distance(jaw, sample(low_cap, 80))))
    high_distance = float(np.median(surface_distance(jaw, sample(high_cap, 80))))
    if high_distance < low_distance:
        axis = -axis
        heights = -heights
        surface_heights = -surface_heights
        bottom, top = -top, -bottom
        low_distance, high_distance = high_distance, low_distance
    bottom_cap = piece.vertices[(heights >= bottom) & (heights < bottom + 0.8)]
    top_cap = piece.vertices[(heights > top - 0.5) & (heights <= top)]
    occlusal = unit(upper.centroid - jaw.centroid)
    patch = control_patch(jaw, piece, occlusal)
    patch_axis, patch_center, _, _, _ = fit_axis(patch)
    matches = []
    for path, kind in references:
        ref = trimesh.load(path, process=True)
        matrix, metrics = rigid_match(piece, ref)
        ref_axis = matrix[:3, :3] @ axis
        ref_center = trimesh.transform_points(center[None], matrix)[0]
        a = radius_profile(piece, axis, center, bottom, top)
        b = radius_profile(ref, ref_axis, ref_center, bottom, top)
        valid = np.isfinite(a) & np.isfinite(b)
        metrics["radius_profile_median_delta_mm"] = (
            round(float(np.median(np.abs(a[valid] - b[valid]))), 4) if valid.sum() >= 6 else None
        )
        metrics["radius_profile_valid_bins"] = int(valid.sum())
        matches.append({"name": path.name, "kind": kind, **metrics})
    matches.sort(key=lambda row: (row["forward_p50_mm"] + row["reverse_p50_mm"]) / 2)
    same_jaw = surface_distance(jaw, sample(piece.vertices, 240))
    return {
        "case": case_dir.name,
        "classification": "implant_locator_candidate_not_confirmed_restorative_abutment",
        "piece_faces": len(piece.faces),
        "component_faces": [len(component.faces) for component in components],
        "piece_to_jaw_min_mm": round(float(same_jaw.min()), 4),
        "piece_to_jaw_p50_mm": round(float(np.median(same_jaw)), 4),
        "piece_to_jaw_sample_vertices": 240,
        "axis_unit_direction": [round(float(x), 5) for x in axis],
        "axis_angle_to_opposing_jaw_deg": round(float(np.degrees(np.arccos(np.clip(abs(axis @ occlusal), 0, 1)))), 2),
        "axis_end_proximity_mm": {"base_median": round(low_distance, 3), "opposite_median": round(high_distance, 3)},
        "bottom_center_model_frame": [round(float(x), 3) for x in bottom_cap.mean(0)],
        "top_center_model_frame": [round(float(x), 3) for x in top_cap.mean(0)],
        "centroid_model_frame": [round(float(x), 3) for x in piece.centroid],
        "radius_mm": round(radius, 3),
        "radius_std_mm": round(radius_std, 3),
        "cylinder_wall_point_ratio": round(wall_ratio, 3),
        "full_axis_extent_mm": round(float(heights.max() - heights.min()), 2),
        "area_1_99_axis_span_mm": round(float(top - bottom), 2),
        "physical_body_length_mm": None,
        "rotational_symmetry_chamfer_mm": {
            "piece_mean": symmetry(piece, axis, center),
            "same_jaw_patch_mean": symmetry(patch, patch_axis, patch_center),
            "same_jaw_patch_faces": len(patch.faces),
            "same_jaw_patch_is_verified_natural_tooth": False,
        },
        "rigid_reference_matches": matches,
    }


def main():
    references = [(path, "implant_library_component") for path in sorted(LIB.glob("*.stl"))]
    references.append((FINAL_ABUTMENT_CONTROL, "labeled_final_abutment_design_control"))
    references.extend((path, "crown_or_anatomy_negative") for path in NEG)
    missing = [str(path) for path, _ in references if not path.exists()]
    if missing:
        raise FileNotFoundError("阴性或库件对照缺失: " + ", ".join(missing))
    results = {}
    for case_dir in sorted(ROOT.glob("基台测试*")):
        if not case_dir.is_dir():
            continue
        print("分析", case_dir.name, flush=True)
        results[case_dir.name] = analyze(case_dir, references)
        print("  前三匹配", [(r["name"], r["forward_p50_mm"], r["reverse_p50_mm"])
                         for r in results[case_dir.name]["rigid_reference_matches"][:3]], flush=True)
    if len(results) != 3:
        raise ValueError(f"应有 3 例，实际 {len(results)} 例")
    if any(len(row["rigid_reference_matches"]) != len(references) for row in results.values()):
        raise ValueError("有库件或阴性对照未执行")
    (ROOT / "基台位置_计算结果.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("已写入基台位置_计算结果.json")


if __name__ == "__main__":
    main()
