from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d

from .meshops import load_mesh
from .template_cervical import extract_clean_crown_surface


def _unit(value: np.ndarray, name: str = "vector") -> np.ndarray:
    value = np.asarray(value, float)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or norm < 1e-10:
        raise ValueError(f"{name} is degenerate")
    return value / norm


def _template_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "templates"
        / "32牙位基础解剖模板库"
        / "32位牙列-分"
    )


def _template_path(fdi: str) -> Path:
    path = _template_dir() / f"FDI_{str(fdi)}.stl"
    if not path.exists():
        raise FileNotFoundError(f"template not found for FDI {fdi}: {path}")
    return path


def _jaw_quadrants(fdi: str) -> tuple[tuple[int, int], tuple[int, int]]:
    quadrant = int(str(fdi)[0])
    if quadrant in {1, 2}:
        return (1, 2), (3, 4)
    return (3, 4), (1, 2)


@lru_cache(maxsize=2)
def _cached_jaw_centers(jaw_code: str) -> tuple[tuple[float, float, float], ...]:
    quadrants = (1, 2) if jaw_code == "upper" else (3, 4)
    centers: list[tuple[float, float, float]] = []
    for q in quadrants:
        for pos in range(1, 9):
            path = _template_dir() / f"FDI_{q}{pos}.stl"
            if not path.exists():
                continue
            try:
                center = np.asarray(load_mesh(path).centroid, float)
            except Exception:
                continue
            if center.shape == (3,) and np.isfinite(center).all():
                centers.append(tuple(float(x) for x in center))
    return tuple(centers)


def _jaw_centers(quadrants: tuple[int, int]) -> np.ndarray:
    jaw_code = "upper" if set(quadrants) == {1, 2} else "lower"
    values = _cached_jaw_centers(jaw_code)
    if len(values) < 6:
        raise ValueError(f"template library {jaw_code} jaw has insufficient registered teeth")
    return np.asarray(values, float)


def library_occlusal_reference(fdi: str) -> tuple[np.ndarray, dict]:
    """Return an occlusally oriented normal from the registered 32-tooth library.

    Individual Dental Merlin teeth retain one shared library coordinate system.
    Their jaw centroids therefore give us a much safer occlusal reference than
    assuming that the largest PCA variance of one complete molar is its long axis.
    The plane normal is sign-oriented toward the opposite jaw.
    """
    target_quadrants, opposite_quadrants = _jaw_quadrants(str(fdi))
    target = _jaw_centers(target_quadrants)
    opposite = _jaw_centers(opposite_quadrants)

    target_center = target.mean(axis=0)
    centered = target - target_center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    normal = _unit(vh[-1], "library jaw plane normal")
    toward_opponent = _unit(opposite.mean(axis=0) - target_center, "library opponent direction")
    if float(normal @ toward_opponent) < 0.0:
        normal = -normal

    residual = np.abs(centered @ normal)
    alignment = float(normal @ toward_opponent)
    return normal, {
        "method": "REGISTERED_LIBRARY_JAW_PLANE_TO_OPPONENT_V1",
        "target_tooth_count": int(len(target)),
        "opponent_tooth_count": int(len(opposite)),
        "plane_residual_median_mm": float(np.median(residual)),
        "plane_residual_p95_mm": float(np.quantile(residual, 0.95)),
        "normal_alignment_to_opponent": alignment,
        "singular_values": [float(x) for x in singular],
    }


def _template_buccal_hint(fdi: str, z_axis: np.ndarray) -> tuple[np.ndarray, dict]:
    target_quadrants, _ = _jaw_quadrants(str(fdi))
    centers = _jaw_centers(target_quadrants)
    target = np.asarray(load_mesh(_template_path(str(fdi))).centroid, float)
    arch_center = centers.mean(axis=0)
    radial = target - arch_center
    radial -= z_axis * float(radial @ z_axis)
    length = float(np.linalg.norm(radial))
    if length < 0.5:
        raise ValueError("template library buccal radial direction is degenerate")
    return _unit(radial, "template buccal"), {
        "method": "REGISTERED_LIBRARY_ARCH_RADIAL_V1",
        "radial_length_mm": length,
        "arch_center": arch_center.tolist(),
        "target_center": target.tolist(),
    }


def _choose_tooth_axis(vertices: np.ndarray, fdi: str) -> tuple[np.ndarray, dict]:
    center = vertices.mean(axis=0)
    covariance = np.cov((vertices - center).T)
    values, vectors = np.linalg.eigh(covariance)
    reference, reference_report = library_occlusal_reference(str(fdi))

    alignments = np.abs(vectors.T @ reference)
    selected = int(np.argmax(alignments))
    selected_alignment = float(alignments[selected])
    pca_axis = _unit(vectors[:, selected], "template PCA axis")
    if float(pca_axis @ reference) < 0.0:
        pca_axis = -pca_axis

    # A complete multi-root molar can have larger mesiodistal/root-spread variance
    # than crown-root variance. PCA is therefore only allowed to refine the library
    # occlusal reference when one eigenvector actually agrees with that reference.
    if selected_alignment >= 0.55:
        axis = pca_axis
        source = "LIBRARY_OCCLUSAL_GUIDED_PCA"
    else:
        axis = reference
        source = "LIBRARY_OCCLUSAL_NORMAL_DIRECT_FALLBACK"

    return _unit(axis, "template crown-root axis"), {
        "source": source,
        "selected_pca_index": selected,
        "selected_alignment_abs": selected_alignment,
        "pca_eigenvalues": [float(x) for x in values],
        "pca_alignments_abs": [float(x) for x in alignments],
        "library_reference": reference.tolist(),
        "library_reference_report": reference_report,
    }


def _cervical_cut(
    template: trimesh.Trimesh,
    vertices: np.ndarray,
    center: np.ndarray,
    z_axis: np.ndarray,
) -> tuple[float, np.ndarray, dict]:
    axial = (vertices - center) @ z_axis
    helper = np.array([1.0, 0.0, 0.0])
    helper -= z_axis * float(helper @ z_axis)
    if float(np.linalg.norm(helper)) < 1e-6:
        helper = np.array([0.0, 1.0, 0.0])
        helper -= z_axis * float(helper @ z_axis)
    e1 = _unit(helper, "template radial basis 1")
    e2 = _unit(np.cross(z_axis, e1), "template radial basis 2")
    rel = vertices - center
    radius = np.sqrt((rel @ e1) ** 2 + (rel @ e2) ** 2)

    qlo, qhi = np.quantile(axial, [0.02, 0.98])
    span = max(float(qhi - qlo), 1e-6)
    edges = np.linspace(float(qlo), float(qhi), 56)
    mids = (edges[:-1] + edges[1:]) * 0.5
    profile = np.full(len(mids), np.nan)
    counts = np.zeros(len(mids), int)
    for i in range(len(mids)):
        mask = (axial >= edges[i]) & (axial < edges[i + 1])
        counts[i] = int(mask.sum())
        if counts[i] >= 8:
            profile[i] = np.quantile(radius[mask], 0.78)

    good = np.isfinite(profile)
    if int(good.sum()) >= 10:
        ids = np.arange(len(profile))
        profile[~good] = np.interp(ids[~good], ids[good], profile[good])
        smooth = gaussian_filter1d(profile, 1.35)
        zn = (mids - qlo) / span
        # Root apices occupy the low end and the occlusal table the high end.
        # Restrict the search to the cervical/root-trunk transition.
        allowed = (zn >= 0.24) & (zn <= 0.68)
        neck_id = int(np.argmin(np.where(allowed, smooth, np.inf)))
        crown_cut = float(mids[neck_id])
        cut_mode = "RADIAL_NECK_PROFILE_OCCLUSAL_REFERENCED"
    else:
        crown_cut = float(np.quantile(axial, 0.44))
        cut_mode = "QUANTILE_FALLBACK"
        smooth = np.nan_to_num(profile)
        neck_id = -1

    face_axial = axial[np.asarray(template.faces, int)].mean(axis=1)
    return crown_cut, face_axial, {
        "cut_mode": cut_mode,
        "cut_value_axis_mm": crown_cut,
        "axial_robust_span_mm": span,
        "profile_valid_bins": int(good.sum()),
        "profile_neck_bin": int(neck_id),
    }


def canonicalize_full_tooth(
    template: trimesh.Trimesh,
    fdi: str,
) -> tuple[trimesh.Trimesh, dict]:
    """Convert a registered complete tooth into a crown-only canonical template.

    The key invariant is that +Z is *not* inferred from the largest PCA variance.
    It is anchored to the registered library jaw/opponent geometry, with PCA used
    only as a local refinement when aligned to that anatomical reference.
    """
    vertices = np.asarray(template.vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 100:
        raise ValueError("template mesh is invalid")
    if not np.isfinite(vertices).all():
        raise ValueError("template mesh contains NaN/Inf")

    center = vertices.mean(axis=0)
    z_axis, axis_report = _choose_tooth_axis(vertices, str(fdi))
    crown_cut, face_axial, cut_report = _cervical_cut(
        template, vertices, center, z_axis
    )

    keep = face_axial >= crown_cut
    crown = template.submesh([np.flatnonzero(keep)], append=True, repair=False)
    if crown is None or len(crown.faces) < max(100, int(0.15 * len(template.faces))):
        axial = (vertices - center) @ z_axis
        crown_cut = float(np.quantile(axial, 0.40))
        face_axial = axial[np.asarray(template.faces, int)].mean(axis=1)
        crown = template.submesh(
            [np.flatnonzero(face_axial >= crown_cut)], append=True, repair=False
        )
        cut_report["cut_mode"] += "_SAFE_FALLBACK"
        cut_report["cut_value_axis_mm"] = crown_cut
    if crown is None or len(crown.faces) < 100:
        raise ValueError("template crown extraction produced insufficient geometry")

    # STL-only Dental Merlin inputs have no reliable labeled cervical seam.
    # Perform an exact triangle cut and prove one complete open rim before
    # downstream placement. A fallback remains visible for engineering review,
    # but can never be reported as clinically validated geometry.
    cleaned, cervical_cut = extract_clean_crown_surface(
        template, z_axis, center, crown_cut
    )
    if cleaned is not None:
        crown = cleaned

    cv = np.asarray(crown.vertices, float)
    crown_center = cv.mean(axis=0)
    centered = cv - crown_center
    y_axis, buccal_report = _template_buccal_hint(str(fdi), z_axis)
    x_axis = _unit(np.cross(y_axis, z_axis), "template arch tangent")
    y_axis = _unit(np.cross(z_axis, x_axis), "template buccal orthogonalized")

    local = np.column_stack(
        (centered @ x_axis, centered @ y_axis, centered @ z_axis)
    )
    z0 = float(np.quantile(local[:, 2], 0.02))
    local[:, 2] -= z0

    # Use the cervical band as the canonical X/Y origin. Placement uses the same
    # anatomical band, so this prevents a correctly oriented but laterally biased
    # complete-tooth centroid from moving the crown away from the preparation.
    z_cerv = float(np.quantile(local[:, 2], 0.12))
    cervical = local[:, 2] <= z_cerv
    if int(cervical.sum()) < 8:
        cervical = local[:, 2] <= np.quantile(local[:, 2], 0.20)
    cervical_center_xy = np.median(local[cervical, :2], axis=0)
    local[:, 0] -= float(cervical_center_xy[0])
    local[:, 1] -= float(cervical_center_xy[1])

    width_x = float(np.quantile(local[:, 0], 0.98) - np.quantile(local[:, 0], 0.02))
    width_y = float(np.quantile(local[:, 1], 0.98) - np.quantile(local[:, 1], 0.02))
    height = float(np.quantile(local[:, 2], 0.98) - np.quantile(local[:, 2], 0.02))
    width = max(width_x, width_y, 1e-6)
    body = local[:, 2] >= np.quantile(local[:, 2], 0.30)
    body_center_xy = np.median(local[body, :2], axis=0) if int(body.sum()) >= 8 else np.median(local[:, :2], axis=0)
    offset = float(np.linalg.norm(body_center_xy))
    offset_ratio = offset / width
    height_width_ratio = height / width

    if offset_ratio > 0.65:
        raise ValueError(
            f"template canonicalization lateral offset is implausible: ratio={offset_ratio:.3f}"
        )
    if not (0.18 <= height_width_ratio <= 3.20):
        raise ValueError(
            "template canonicalization crown aspect is implausible: "
            f"height/width={height_width_ratio:.3f}"
        )

    crown.vertices = local
    return crown, {
        "mode": "FULL_TOOTH_TO_CANONICAL_CROWN_V4_LIBRARY_OCCLUSAL",
        "clinical_cervical_valid": bool(cleaned is not None),
        "cervical_cut": cervical_cut,
        "source_faces": int(len(template.faces)),
        "crown_faces": int(len(crown.faces)),
        "axis": axis_report,
        "cut": cut_report,
        "buccal": buccal_report,
        "canonical_axes": "+X arch-tangent, +Y buccal/outward, +Z occlusal/opponent",
        "z_axis_world": z_axis.tolist(),
        "x_axis_world": x_axis.tolist(),
        "y_axis_world": y_axis.tolist(),
        "cervical_center_xy_before_recentering_mm": [float(x) for x in cervical_center_xy],
        "cervical_to_body_offset_mm": offset,
        "cervical_to_body_offset_ratio": float(offset_ratio),
        "crown_width_mm": {"x": width_x, "y": width_y},
        "crown_height_mm": height,
        "crown_height_width_ratio": float(height_width_ratio),
    }
