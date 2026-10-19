"""CrownFlow-derived engineering cutter stage; outer geometry is immutable.

The source mesh (support), Margin and frame are not modified. A cutter is a
construction/diagnostic artifact, not the clinical intaglio or final STL.
This module is intentionally isolated from the outer-crown solver so a W3
failure cannot invalidate a previously generated outer candidate.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import trimesh

from .frame import PatientFrame
from .intaglio import build_engineering_cutter
from .meshops import save_mesh


ENGINEERING_PROFILE = {
    "cement_gap_mm": 0.06,
    "cement_gap_start_mm": 0.10,
    "cement_gap_top_mm": None,
    "machine_axes": 5,
}


def generate_cutter_artifacts(
    support: trimesh.Trimesh, margin: np.ndarray, frame: PatientFrame,
    out_dir: Path, *, resolution_mm: float | None = None,
) -> dict:
    """Fail closed for the cutter only, preserving all previous outer artifacts.

    Outputs are named independently; no file is published as final/intaglio.
    A retry always invalidates old cutter files before writing a new candidate.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("06_blockout.stl", "06_cutter.stl", "06_intaglio_report.json"):
        (out_dir / name).unlink(missing_ok=True)
    report = {
        "status": "REVIEW_REQUIRED",
        "method": "CROWNFLOW_ENGINEERING_CUTTER_STAGE_V1",
        "source": "EXTRACTED_ABUTMENT_COMPONENT_NOT_PREPARED_SUPPORT_CONTRACT",
        "support_source_unchanged": True,
        "accepted_outer_unchanged": True,
        "clinical_final_ready": False,
    }
    try:
        blockout, cutter, built = build_engineering_cutter(
            support, margin, frame, resolution_mm=resolution_mm,
            **ENGINEERING_PROFILE,
        )
        save_mesh(blockout, out_dir / "06_blockout.stl")
        save_mesh(cutter, out_dir / "06_cutter.stl")
        report.update(built)
        report["artifacts"] = {
            "blockout": "06_blockout.stl",
            "cutter": "06_cutter.stl",
        }
    except Exception as exc:  # W3 is nonblocking: unexpected numerical failures also preserve the outer.
        report.update({
            "status": "REVIEW_REQUIRED",
            "reason": type(exc).__name__,
            "error": str(exc)[:500],
            "clinical_intaglio_proven": False,
            "final_crown_ready": False,
        })
        # Do not publish incomplete geometry.
        (out_dir / "06_blockout.stl").unlink(missing_ok=True)
        (out_dir / "06_cutter.stl").unlink(missing_ok=True)
    (out_dir / "06_intaglio_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
