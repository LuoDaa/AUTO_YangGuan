"""Patient-owned local context for outer-crown contact and occlusal solvers.

CrownFlow does not use an entire antagonist arch as a Boolean cutter. This
module similarly isolates a bounded *real patient* surface near the intended
crown before computing triangle distances. The patient frame and the real bite
pose are immutable, and an empty or ambiguous ROI is not silently fabricated.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .frame import PatientFrame


def extract_local_antagonist(
    antagonist: trimesh.Trimesh,
    crown: trimesh.Trimesh,
    frame: PatientFrame,
    *,
    lateral_pad_mm: float = 2.8,
    axial_below_mm: float = 1.0,
    axial_above_mm: float = 8.0,
) -> tuple[trimesh.Trimesh | None, dict]:
    """Extract real antagonist triangles near the current crown envelope.

    XY is patient-frame mesiodistal/buccolingual, and Z points toward the
    opposing arch. Only triangle-face centers are used to propose the patch;
    all subsequent proximity computations use the actual triangle vertices.
    """
    if len(crown.faces) == 0 or len(antagonist.faces) == 0:
        return None, {"status": "REVIEW_REQUIRED", "reason": "EMPTY_CROWN_OR_ANTAGONIST"}
    crown_local = frame.to_local(np.asarray(crown.vertices, float))
    face_local = frame.to_local(np.asarray(antagonist.triangles_center, float))
    if not np.isfinite(crown_local).all() or not np.isfinite(face_local).all():
        return None, {"status": "REVIEW_REQUIRED", "reason": "NONFINITE_CONTEXT"}
    lo = np.quantile(crown_local, 0.01, axis=0)
    hi = np.quantile(crown_local, 0.99, axis=0)
    xy = (
        (face_local[:, 0] >= lo[0] - lateral_pad_mm)
        & (face_local[:, 0] <= hi[0] + lateral_pad_mm)
        & (face_local[:, 1] >= lo[1] - lateral_pad_mm)
        & (face_local[:, 1] <= hi[1] + lateral_pad_mm)
    )
    z = (
        (face_local[:, 2] >= lo[2] - axial_below_mm)
        & (face_local[:, 2] <= hi[2] + axial_above_mm)
    )
    mask = xy & z
    face_ids = np.flatnonzero(mask)
    report = {
        "method": "PATIENT_FRAME_BOUNDED_ACTUAL_ANTAGONIST_PATCH_V1",
        "source_faces": int(len(antagonist.faces)),
        "roi_faces": int(len(face_ids)),
        "crown_envelope_p01_mm": lo.tolist(),
        "crown_envelope_p99_mm": hi.tolist(),
        "lateral_pad_mm": float(lateral_pad_mm),
        "source_geometry_untouched": True,
    }
    if len(face_ids) < 8:
        return None, {
            **report,
            "status": "REVIEW_REQUIRED",
            "reason": "ANTAGONIST_NOT_RESOLVED_IN_LOCAL_CROWN_ROI",
        }
    patch = antagonist.submesh([face_ids], append=True, repair=False)
    if patch is None or len(patch.faces) < 8:
        return None, {
            **report,
            "status": "REVIEW_REQUIRED",
            "reason": "LOCAL_ANTAGONIST_PATCH_EMPTY",
        }
    return patch, {**report, "status": "PASS_ENGINEERING", "roi_vertices": int(len(patch.vertices))}
