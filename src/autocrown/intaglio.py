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
