from __future__ import annotations

import numpy as np
import trimesh

from .crownflow_abutment_margin import extract_abutment_margin
from .crownflow_insertion import AxisStatus, solve_insertion_axis
from .crownflow_patient_frame import build_patient_frame
from .localization import LocalizationError
from .support_locator import SupportLocatorError, localize_support


def _orient_toward_opponent(
    axis: np.ndarray,
    opponent: trimesh.Trimesh,
    jaw: trimesh.Trimesh,
) -> tuple[np.ndarray, dict]:
    axis = np.asarray(axis, float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    occlusal = np.asarray(opponent.centroid, float) - np.asarray(jaw.centroid, float)
    norm = float(np.linalg.norm(occlusal))
    report = {"orientation_flipped_toward_opponent": False, "axis_alignment_to_opponent": None}
    if norm <= 1e-10:
        return axis, report
    occlusal /= norm
    if float(axis @ occlusal) < 0.0:
        axis = -axis
        report["orientation_flipped_toward_opponent"] = True
    report["axis_alignment_to_opponent"] = float(axis @ occlusal)
    return axis, report


def _margin_once(
    support: trimesh.Trimesh,
    axis: np.ndarray,
    *,
    stage: str,
) -> tuple[np.ndarray, dict]:
    try:
        points, report = extract_abutment_margin(
            support,
            axis,
            origin=np.asarray(support.centroid, float),
            strict_qc=False,
        )
    except Exception as exc:
        raise LocalizationError(f"{stage} CrownFlow shoulder extraction failed: {exc}") from exc
    margin = np.asarray(points, float)
    if margin.ndim != 2 or margin.shape[1] != 3 or len(margin) < 32 or not np.isfinite(margin).all():
        raise LocalizationError(f"{stage} CrownFlow shoulder extraction returned invalid Margin")
    return margin, {**report, "stage": stage, "ported_from_crownflow": True}


def _axis_once(
    support: trimesh.Trimesh,
    margin: np.ndarray,
    seed: np.ndarray,
    *,
    stage: str,
) -> tuple[np.ndarray, dict]:
    try:
        result = solve_insertion_axis(support, margin, seed)
    except Exception as exc:
        raise LocalizationError(f"{stage} CrownFlow insertion solve failed: {exc}") from exc
    report = dict(result.report)
    report.update({"stage": stage, "ported_from_crownflow": True})
    if result.status == AxisStatus.FAILED:
        raise LocalizationError(f"{stage} CrownFlow insertion solver returned FAILED")
    return np.asarray(result.axis, float), report


def localize_case_geometry(
    target_arch: trimesh.Trimesh,
    opponent: trimesh.Trimesh,
):
    """Authoritative raw-case localization for UI and crown generation.

    This path deliberately reuses CrownFlow's mature pieces instead of the early
    AUTO_YangGuan proxies:

    support rank -> CrownFlow abutment shoulder -> CrownFlow insertion solve ->
    shoulder refine -> insertion refine -> CrownFlow Margin-owned PatientFrame.

    Margin is never synthesized as a circle and support PCA never directly owns Z.
    """
    try:
        located = localize_support(target_arch, opponent)
    except SupportLocatorError as exc:
        raise LocalizationError(str(exc)) from exc

    # Stage A: rough support axis is only a seed for the mature shoulder extractor.
    margin0, margin0_report = _margin_once(
        located.support,
        located.rough_axis,
        stage="MARGIN_INITIAL",
    )
    axis1, axis1_report = _axis_once(
        located.support,
        margin0,
        located.rough_axis,
        stage="AXIS_INITIAL",
    )
    axis1, axis1_orientation = _orient_toward_opponent(axis1, opponent, located.jaw)
    axis1_report.update(axis1_orientation)
    axis1_report["axis"] = axis1.tolist()

    # Stage B: rerun shoulder extraction in the solved insertion frame, then solve Z
    # again from the refined non-planar Margin + side-wall evidence.
    margin, margin_report = _margin_once(
        located.support,
        axis1,
        stage="MARGIN_REFINED",
    )
    axis2, axis2_report = _axis_once(
        located.support,
        margin,
        axis1,
        stage="AXIS_REFINED",
    )
    axis2, axis2_orientation = _orient_toward_opponent(axis2, opponent, located.jaw)
    axis2_report.update(axis2_orientation)
    axis2_report["axis"] = axis2.tolist()

    try:
        frame, frame_report = build_patient_frame(
            margin,
            axis2,
            working_jaw=located.jaw,
        )
    except Exception as exc:
        raise LocalizationError(f"CrownFlow PatientFrame build failed: {exc}") from exc

    status = "READY"
    reasons: list[str] = []
    if located.report.get("status") != "READY":
        status = "REVIEW_REQUIRED"
        reasons.append("SUPPORT_SELECTION_REVIEW_REQUIRED")
    if margin_report.get("status") == "REVIEW_REQUIRED":
        status = "REVIEW_REQUIRED"
        reasons.append("MARGIN_REVIEW_REQUIRED")
    if axis2_report.get("status") != AxisStatus.OK.value:
        status = "REVIEW_REQUIRED"
        reasons.append("INSERTION_AXIS_REVIEW_REQUIRED")

    # Structural sanity is a hard generation prerequisite. These checks target the
    # exact failure that produced crowns anchored to the wrong cervical level.
    margin_step = np.linalg.norm(np.roll(margin, -1, axis=0) - margin, axis=1)
    margin_center = np.mean(margin, axis=0)
    support_vertices = np.asarray(located.support.vertices, float)
    axis_height = (support_vertices - margin_center) @ axis2
    coronal_fraction = float(np.mean(axis_height > 0.05)) if len(axis_height) else 0.0
    cervical_fraction = float(np.mean(np.abs(axis_height) <= 0.75)) if len(axis_height) else 0.0
    structural = {
        "margin_point_count": int(len(margin)),
        "margin_max_step_mm": float(np.max(margin_step)),
        "margin_median_step_mm": float(np.median(margin_step)),
        "support_coronal_fraction_above_margin": coronal_fraction,
        "support_fraction_within_0_75mm_of_margin": cervical_fraction,
        "frame_origin_equals_margin_centroid_mm": float(np.linalg.norm(frame.origin - margin_center)),
    }
    if len(margin) < 64 or not np.isfinite(margin).all():
        raise LocalizationError("refined Margin failed structural validity")
    if structural["frame_origin_equals_margin_centroid_mm"] > 1e-6:
        raise LocalizationError("PatientFrame origin drifted away from Margin centroid")
    if coronal_fraction < 0.18:
        raise LocalizationError(
            "refined Margin is too coronal: too little support remains above the finish line"
        )

    report = {
        "status": status,
        "reasons": reasons,
        "pipeline": "CROWNFLOW_MATURE_CORE_LOCALIZATION_V1",
        "ownership": {
            "support": "AUTO_YANGGUAN_SUPPORT_LOCATOR_V3",
            "margin": "CROWNFLOW_ABUTMENT_SHOULDER_1_1",
            "insertion_axis": "CROWNFLOW_MARGIN_WALL_CONE_SEARCH",
            "patient_frame": "CROWNFLOW_MARGIN_PCA_FRAME",
        },
        "abutment": located.report,
        "margin_initial": margin0_report,
        "axis_initial": axis1_report,
        "margin": margin_report,
        "axis": axis2_report,
        "frame": frame_report,
        "structural_gate": structural,
        "margin_points": np.asarray(margin, float).tolist(),
    }
    return located.jaw, located.support, margin, frame, report
