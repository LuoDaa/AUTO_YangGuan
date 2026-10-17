from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import trimesh

from .config import PipelineConfig
from .contact import adapt_proximal
from .frame import PatientFrame
from .meshops import save_mesh
from .occlusion import adapt_occlusion
from .placement import place_template
from .qc import evaluate_outer


def _write_stage(mesh: trimesh.Trimesh, out_dir: Path | None, name: str) -> str | None:
    if out_dir is None:
        return None
    path = out_dir / f"{name}.stl"
    save_mesh(mesh, path)
    return str(path)


def build_outer_crown(template: trimesh.Trimesh, margin: np.ndarray, frame: PatientFrame,
                      *, mesial: trimesh.Trimesh | None = None,
                      distal: trimesh.Trimesh | None = None,
                      antagonist: trimesh.Trimesh | None = None,
                      mesial_sign: float = 1.0,
                      config: PipelineConfig | None = None,
                      out_dir: str | Path | None = None):
    """Build a conservative outer-crown candidate.

    ``frame.x`` is a deterministic right-handed arch tangent and ``frame.y`` is
    the buccal/outward direction. ``mesial_sign`` tells the proximal solver which
    sign of the arch tangent corresponds to the FDI mesial side. This keeps the
    placement frame stable instead of rotating the whole crown 180 degrees merely
    to satisfy a side label.
    """
    cfg = config or PipelineConfig()
    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "pipeline": "AUTO_YANGGUAN_HYBRID_MVP_2",
        "goal": "FAST_CONSERVATIVE_20_40_POINT_CROWN",
        "frame_contract": "X_RIGHT_HANDED_ARCH_TANGENT_Y_BUCCAL_Z_INSERTION",
        "mesial_sign_on_frame_x": float(mesial_sign),
        "config": cfg.as_dict(),
        "stages": [],
        "artifacts": {},
    }

    placed, placement_report = place_template(template, margin, frame, cfg.placement)
    report["stages"].append({"name": "PLACEMENT", **placement_report})
    p = _write_stage(placed, out_path, "01_placed")
    if p:
        report["artifacts"]["placed"] = p
    reference = placed.copy()
    current = placed.copy()

    if mesial is not None:
        current, r = adapt_proximal(
            current, mesial, frame, margin, "mesial",
            cervical_lock_mm=cfg.quality.cervical_lock_mm,
            config=cfg.proximal,
            mesial_sign=mesial_sign,
        )
        report["stages"].append({"name": "MESIAL", **r})
        p = _write_stage(current, out_path, "02_mesial")
        if p:
            report["artifacts"]["mesial"] = p
    else:
        report["stages"].append({"name": "MESIAL", "status": "SKIPPED", "reason": "NO_MESIAL_MESH"})

    if distal is not None:
        current, r = adapt_proximal(
            current, distal, frame, margin, "distal",
            cervical_lock_mm=cfg.quality.cervical_lock_mm,
            config=cfg.proximal,
            mesial_sign=mesial_sign,
        )
        report["stages"].append({"name": "DISTAL", **r})
        p = _write_stage(current, out_path, "03_distal")
        if p:
            report["artifacts"]["distal"] = p
    else:
        report["stages"].append({"name": "DISTAL", "status": "SKIPPED", "reason": "NO_DISTAL_MESH"})

    if antagonist is not None:
        current, r = adapt_occlusion(
            current, antagonist, frame, margin,
            cervical_lock_mm=cfg.quality.cervical_lock_mm,
            config=cfg.occlusal,
        )
        report["stages"].append({"name": "OCCLUSAL", **r})
        p = _write_stage(current, out_path, "04_occlusal")
        if p:
            report["artifacts"]["occlusal"] = p
    else:
        report["stages"].append({"name": "OCCLUSAL", "status": "SKIPPED", "reason": "NO_ANTAGONIST_MESH"})

    qc = evaluate_outer(reference, current, cfg.quality)
    report["outer_qc"] = qc
    report["status"] = "OUTER_CANDIDATE_READY" if qc["status"] == "PASS_ENGINEERING" else "OUTER_REVIEW_REQUIRED"

    p = _write_stage(current, out_path, "05_outer_candidate")
    if p:
        report["artifacts"]["outer_candidate"] = p
    if out_path is not None:
        report_path = out_path / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["artifacts"]["report"] = str(report_path)

    return current, report
