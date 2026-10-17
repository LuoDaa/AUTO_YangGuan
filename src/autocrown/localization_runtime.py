from __future__ import annotations

import numpy as np
import trimesh

from .localization import (
    LocalizationError,
    build_automatic_patient_frame,
    detect_finish_line,
    solve_insertion_axis,
)
from .support_locator import SupportLocatorError, localize_support


def localize_case_geometry(
    target_arch: trimesh.Trimesh,
    opponent: trimesh.Trimesh,
):
    """Authoritative raw-case localization used by the UI and generation path.

    The first prototype guessed ``parts[1]`` and used component PCA directly as
    insertion axis.  This runtime instead mirrors CrownFlow's separation of
    responsibilities: support candidate ranking -> rough occlusal/wall axis ->
    finish-line candidate -> margin+wall insertion solve -> refine margin/axis ->
    right-handed PatientFrame.
    """
    try:
        located = localize_support(target_arch, opponent)
    except SupportLocatorError as exc:
        raise LocalizationError(str(exc)) from exc

    margin0, margin0_report = detect_finish_line(located.support, located.rough_axis)
    axis1, axis1_report = solve_insertion_axis(
        located.support, margin0, located.rough_axis,
    )
    margin, margin_report = detect_finish_line(located.support, axis1)
    axis2, axis2_report = solve_insertion_axis(
        located.support, margin, axis1,
    )

    # Keep the final axis direction toward the opposing jaw.  The formal solver
    # optimizes an unoriented axis line; crown placement needs a signed direction.
    occlusal = np.asarray(opponent.centroid, float) - np.asarray(located.jaw.centroid, float)
    norm = float(np.linalg.norm(occlusal))
    if norm > 1e-10:
        occlusal /= norm
        if float(axis2 @ occlusal) < 0:
            axis2 = -axis2
            axis2_report = dict(axis2_report)
            axis2_report["axis"] = axis2.tolist()
            axis2_report["orientation_flipped_toward_opponent"] = True
        else:
            axis2_report = dict(axis2_report)
            axis2_report["orientation_flipped_toward_opponent"] = False
        axis2_report["axis_alignment_to_opponent"] = float(axis2 @ occlusal)

    frame, frame_report = build_automatic_patient_frame(
        located.jaw, margin, axis2,
    )
    status = "READY"
    reasons: list[str] = []
    if located.report.get("status") != "READY":
        status = "REVIEW_REQUIRED"
        reasons.append("SUPPORT_SELECTION_REVIEW_REQUIRED")
    if axis2_report.get("status") != "OK":
        status = "REVIEW_REQUIRED"
        reasons.append("INSERTION_AXIS_REVIEW_REQUIRED")

    report = {
        "status": status,
        "reasons": reasons,
        "pipeline": "SUPPORT_MARGIN_AXIS_FRAME_V3",
        "abutment": located.report,
        "margin_initial": margin0_report,
        "axis_initial": axis1_report,
        "margin": margin_report,
        "axis": axis2_report,
        "frame": frame_report,
        "margin_points": np.asarray(margin, float).tolist(),
    }
    return located.jaw, located.support, margin, frame, report
