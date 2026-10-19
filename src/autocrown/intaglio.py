from __future__ import annotations

import numpy as np
import trimesh

from .frame import PatientFrame, axial_height_from_margin


def build_gap_surface_preview(support: trimesh.Trimesh, margin: np.ndarray,
                              frame: PatientFrame, *, cement_gap_mm: float = 0.06,
                              submargin_tolerance_mm: float = 0.15):
    """Create an *experimental* support-derived gap-surface preview.

    This is deliberately not called a final intaglio or cutter. It offsets the
    support surface along vertex normals and reports that directional blockout is
    still missing. The 1.0 outer-crown MVP can therefore evolve independently
    without pretending that undercut-safe seating has already been solved.
    """
    if cement_gap_mm < 0:
        raise ValueError("cement_gap_mm must be non-negative")
    out = support.copy()
    v = np.asarray(out.vertices, float)
    n = np.asarray(out.vertex_normals, float)
    if len(v) != len(n):
        raise ValueError("support vertex normals unavailable")
    nn = np.linalg.norm(n, axis=1)
    good = nn > 1e-12
    n[good] /= nn[good, None]
    n[~good] = 0.0
    height = axial_height_from_margin(v, margin, frame)
    clinical = height >= -float(submargin_tolerance_mm)
    moved = v.copy()
    moved[clinical] += n[clinical] * float(cement_gap_mm)
    out.vertices = moved
    return out, {
        "status": "PREVIEW_ONLY",
        "method": "SUPPORT_NORMAL_OFFSET_PREVIEW",
        "cement_gap_mm": float(cement_gap_mm),
        "clinical_vertices": int(clinical.sum()),
        "directional_blockout": "NOT_IMPLEMENTED_IN_MVP_0_1",
        "final_intaglio_claim": False,
        "note": (
            "Do not use this preview as a manufacturing cutter. The final intaglio "
            "must add insertion-axis blockout, face provenance, support ownership, "
            "Margin-owned clipping and cervical assembly."
        ),
    }


def build_engineering_cutter(
    support: trimesh.Trimesh,
    margin: np.ndarray,
    frame: PatientFrame,
    *,
    cement_gap_mm: float = 0.06,
    cement_gap_start_mm: float = 0.10,
    cement_gap_top_mm: float | None = None,
    machine_axes: int = 5,
    resolution_mm: float | None = None,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Ported CrownFlow support → directional blockout → variable-gap cutter.

    Deliberately returns an ENGINEERING_ONLY cutter, not a completed crown.
    In particular, a valid cutter does *not* establish clinical fit, thickness,
    a shared three-dimensional Margin seam, or insertion clearance of the final
    assembled crown. Never substitute the cutter for the final intaglio STL.
    """
    from .crownflow_intaglio_core import (
        BlockoutPolicy, BlockoutStatus, GapFieldParams,
        build_blockout, apply_gap_field,
    )
    if not np.isfinite(cement_gap_mm) or cement_gap_mm <= 0:
        raise ValueError("cement_gap_mm must be finite and strictly positive")
    if not np.isfinite(cement_gap_start_mm) or cement_gap_start_mm < 0:
        raise ValueError("cement_gap_start_mm must be finite and non-negative")
    if cement_gap_top_mm is not None and (
        not np.isfinite(cement_gap_top_mm) or cement_gap_top_mm < 0
    ):
        raise ValueError("cement_gap_top_mm must be finite and non-negative")
    if machine_axes not in (3, 5):
        raise ValueError("machine_axes must be 3 or 5 (unknown policy is not guessed)")
    policy = BlockoutPolicy(
        machine_axes=machine_axes,
        additional_blockout_after_gap=0.0 if machine_axes == 3 else None,
        untouchable_zone_mm=0.0 if machine_axes == 3 else 0.10,
        provenance={"source": "AUTO_YANGGUAN_ENGINEERING_PROFILE"},
    )
    gap = GapFieldParams(
        cement_gap_mm=float(cement_gap_mm),
        cement_gap_start_mm=float(cement_gap_start_mm),
        cement_gap_top_mm=None if cement_gap_top_mm is None else float(cement_gap_top_mm),
        additional_spacing_xyz_mm=None,
        additional_spacing_z_on_top_mm=None,
        provenance={"source": "AUTO_YANGGUAN_ENGINEERING_PROFILE",
                    "top_gap": "UNSPECIFIED_NOT_GUESSED" if cement_gap_top_mm is None else "EXPLICIT"},
    )
    result = build_blockout(support, margin, frame.z, policy, resolution_mm=resolution_mm)
    if result.status != BlockoutStatus.OK:
        raise ValueError("DIRECTIONAL_BLOCKOUT_REVIEW_REQUIRED: " + str(result.as_dict()))
    cutter = apply_gap_field(result, support, margin, frame.z, gap)
    if (
        not np.isfinite(cutter.vertices).all()
        or not cutter.is_watertight
        or len(cutter.split(only_watertight=False)) != 1
        or not cutter.is_winding_consistent
    ):
        raise ValueError("CUTTER_TOPOLOGY_REVIEW_REQUIRED")
    report = {
        "status": "CUTTER_READY_ENGINEERING_ONLY",
        "method": "CROWNFLOW_DIRECTIONAL_HEIGHTFIELD_BLOCKOUT_VARIABLE_GAP",
        "blockout": result.as_dict(),
        "gap": gap.as_dict(),
        "cutter": {
            "watertight": bool(cutter.is_watertight),
            "winding_consistent": bool(cutter.is_winding_consistent),
            "faces": int(len(cutter.faces)),
            "volume_mm3": float(abs(cutter.volume)) if cutter.is_volume else None,
        },
        "clinical_intaglio_proven": False,
        "final_crown_ready": False,
        "limitations": [
            "SUPPORT_PREPARED_SURFACE_OWNERSHIP_NOT_YET_VERIFIED",
            "MARGIN_OWNED_CERVICAL_ASSEMBLY_NOT_YET_IMPLEMENTED",
            "FINAL_THICKNESS_AND_INSERTION_QC_NOT_YET_VERIFIED",
        ],
    }
    return result.blockout_mesh, cutter, report
