from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import trimesh

from .config import PipelineConfig
from .contact import adapt_proximal
from .frame import PatientFrame, axial_height_from_margin
from .meshops import save_mesh
from .occlusion import adapt_occlusion
from .placement import place_template
from .qc import evaluate_outer
from .geometry_guards import accept_anatomy_stage
from .patient_constraints import extract_local_antagonist


def _write_stage(mesh: trimesh.Trimesh, out_dir: Path | None, name: str) -> str | None:
    if out_dir is None:
        return None
    path = out_dir / f"{name}.stl"
    save_mesh(mesh, path)
    return str(path)


def _robust_top_z(
    mesh: trimesh.Trimesh | None,
    frame: PatientFrame,
    *,
    quantile: float = 0.90,
) -> float | None:
    if mesh is None or len(mesh.vertices) < 8:
        return None
    local = frame.to_local(np.asarray(mesh.vertices, float))
    z = np.asarray(local[:, 2], float)
    z = z[np.isfinite(z)]
    if len(z) < 8:
        return None
    return float(np.quantile(z, quantile))


def _fit_height_to_neighbors(
    mesh: trimesh.Trimesh,
    frame: PatientFrame,
    margin: np.ndarray,
    *,
    mesial: trimesh.Trimesh | None,
    distal: trimesh.Trimesh | None,
    cervical_lock_mm: float,
) -> tuple[trimesh.Trimesh, dict]:
    """Match global crown height to the adjacent occlusal level without moving Margin.

    Placement owns X/Y and the cervical anchor. This stage owns only the missing
    global Z degree of freedom. The occlusal body is translated along the confirmed
    insertion axis with a smooth cervical ramp, while the locked Margin band stays
    exactly fixed. The later OCCLUSAL stage remains a local relief-only safety pass.
    """
    current = mesh.copy()
    vertices = np.asarray(current.vertices, float)
    if len(vertices) < 8 or not np.isfinite(vertices).all():
        return current, {"status": "NOT_ASSESSED", "reason": "INVALID_CROWN_MESH"}

    neighbor_levels: dict[str, float] = {}
    for name, neighbor in (("mesial", mesial), ("distal", distal)):
        value = _robust_top_z(neighbor, frame, quantile=0.90)
        if value is not None:
            neighbor_levels[name] = value

    if not neighbor_levels:
        return current, {
            "status": "NOT_ASSESSED",
            "reason": "NO_NEIGHBOR_OCCLUSAL_REFERENCE",
            "method": "ADJACENT_OCCLUSAL_HEIGHT_FIT_V1",
        }

    local = frame.to_local(vertices)
    z = np.asarray(local[:, 2], float)
    finite = np.isfinite(z)
    if int(finite.sum()) < 8:
        return current, {"status": "NOT_ASSESSED", "reason": "NO_FINITE_CROWN_Z"}

    current_top = float(np.quantile(z[finite], 0.90))
    levels = np.asarray(list(neighbor_levels.values()), float)
    spread = float(np.ptp(levels)) if len(levels) > 1 else 0.0
    if spread > 2.50:
        # If neighbor extraction disagrees strongly, avoid over-erupting the crown.
        target_top = float(np.min(levels))
        target_policy = "LOWER_NEIGHBOR_CONSERVATIVE_FALLBACK"
    else:
        target_top = float(np.median(levels))
        target_policy = "MEDIAN_ADJACENT_OCCLUSAL_LEVEL"

    requested_shift = target_top - current_top
    applied_shift = float(np.clip(requested_shift, -1.50, 2.00))
    if abs(applied_shift) < 0.06:
        return current, {
            "status": "NO_CORRECTION_NEEDED",
            "method": "ADJACENT_OCCLUSAL_HEIGHT_FIT_V1",
            "neighbor_top_z_mm": neighbor_levels,
            "neighbor_spread_mm": spread,
            "target_policy": target_policy,
            "current_top_z_mm": current_top,
            "target_top_z_mm": target_top,
            "requested_shift_mm": requested_shift,
            "applied_shift_mm": 0.0,
        }

    height = axial_height_from_margin(vertices, margin, frame)
    lock = float(cervical_lock_mm)
    blend_span = 1.60
    u = np.clip((height - lock) / blend_span, 0.0, 1.0)
    weight = u * u * (3.0 - 2.0 * u)

    new_vertices = vertices + (applied_shift * weight)[:, None] * np.asarray(frame.z, float)
    locked = height <= lock
    new_vertices[locked] = vertices[locked]
    current.vertices = new_vertices

    after_top = float(np.quantile(frame.to_local(new_vertices)[:, 2], 0.90))
    return current, {
        "status": "APPLIED",
        "method": "ADJACENT_OCCLUSAL_HEIGHT_FIT_V1",
        "neighbor_top_z_mm": neighbor_levels,
        "neighbor_spread_mm": spread,
        "target_policy": target_policy,
        "current_top_z_mm": current_top,
        "target_top_z_mm": target_top,
        "requested_shift_mm": requested_shift,
        "applied_shift_mm": applied_shift,
        "shift_clamped": bool(abs(applied_shift - requested_shift) > 1e-8),
        "top_after_z_mm": after_top,
        "cervical_locked_vertices": int(locked.sum()),
        "full_shift_vertices": int((weight >= 1.0 - 1e-9).sum()),
        "blend_span_mm": blend_span,
    }


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

    current, height_report = _fit_height_to_neighbors(
        placed,
        frame,
        margin,
        mesial=mesial,
        distal=distal,
        cervical_lock_mm=cfg.quality.cervical_lock_mm,
    )
    current, height_gate = accept_anatomy_stage(
        placed, current, margin, frame, stage="HEIGHT",
        cervical_lock_mm=cfg.quality.cervical_lock_mm, max_stage_move_mm=2.10,
    )
    report["stages"].append({"name": "HEIGHT", **height_report, "geometry_guard": height_gate})
    p = _write_stage(current, out_path, "01_height_fit")
    if p:
        report["artifacts"]["height_fit"] = p

    # Height fitting is part of placement, so downstream QC compares shape edits
    # against the height-corrected anatomical prior rather than penalizing the
    # required global occlusal-level correction itself.
    reference = current.copy()

    if mesial is not None:
        before_step = current.copy()
        current, r = adapt_proximal(
            current, mesial, frame, margin, "mesial",
            cervical_lock_mm=cfg.quality.cervical_lock_mm,
            config=cfg.proximal,
            mesial_sign=mesial_sign,
        )
        current, guard = accept_anatomy_stage(
            before_step, current, margin, frame, stage="MESIAL",
            cervical_lock_mm=cfg.quality.cervical_lock_mm,
            max_stage_move_mm=max(cfg.proximal.max_growth_mm, cfg.proximal.max_relief_mm) + 0.02,
        )
        report["stages"].append({"name": "MESIAL", **r, "geometry_guard": guard})
        p = _write_stage(current, out_path, "02_mesial")
        if p:
            report["artifacts"]["mesial"] = p
    else:
        report["stages"].append({"name": "MESIAL", "status": "SKIPPED", "reason": "NO_MESIAL_MESH"})

    if distal is not None:
        before_step = current.copy()
        current, r = adapt_proximal(
            current, distal, frame, margin, "distal",
            cervical_lock_mm=cfg.quality.cervical_lock_mm,
            config=cfg.proximal,
            mesial_sign=mesial_sign,
        )
        current, guard = accept_anatomy_stage(
            before_step, current, margin, frame, stage="DISTAL",
            cervical_lock_mm=cfg.quality.cervical_lock_mm,
            max_stage_move_mm=max(cfg.proximal.max_growth_mm, cfg.proximal.max_relief_mm) + 0.02,
        )
        report["stages"].append({"name": "DISTAL", **r, "geometry_guard": guard})
        p = _write_stage(current, out_path, "03_distal")
        if p:
            report["artifacts"]["distal"] = p
    else:
        report["stages"].append({"name": "DISTAL", "status": "SKIPPED", "reason": "NO_DISTAL_MESH"})

    if antagonist is not None:
        actual_patch, roi_report = extract_local_antagonist(antagonist, current, frame)
        if actual_patch is None:
            report["stages"].append({
                "name": "OCCLUSAL", "status": "NOT_ASSESSED",
                "reason": roi_report.get("reason"), "reference_roi": roi_report,
            })
        else:
            before_step = current.copy()
            current, r = adapt_occlusion(
                current, actual_patch, frame, margin,
                cervical_lock_mm=cfg.quality.cervical_lock_mm,
                config=cfg.occlusal,
            )
            current, guard = accept_anatomy_stage(
                before_step, current, margin, frame, stage="OCCLUSAL",
                cervical_lock_mm=cfg.quality.cervical_lock_mm,
                max_stage_move_mm=cfg.occlusal.max_relief_mm + 0.02,
            )
            report["stages"].append({
                "name": "OCCLUSAL", **r, "reference_roi": roi_report,
                "geometry_guard": guard,
            })
            p = _write_stage(current, out_path, "04_occlusal")
            if p:
                report["artifacts"]["occlusal"] = p
    else:
        report["stages"].append({"name": "OCCLUSAL", "status": "SKIPPED", "reason": "NO_ANTAGONIST_MESH"})

    qc = evaluate_outer(reference, current, cfg.quality)
    report["outer_qc"] = qc
    rejected_stages = [
        x["name"] for x in report["stages"]
        if (x.get("geometry_guard") or {}).get("status") in {"REJECTED", "ACCEPTED_BACKOFF"}
        or (x["name"] == "OCCLUSAL" and x.get("status") == "NOT_ASSESSED" and antagonist is not None)
    ]
    report["stage_guard_reviews"] = rejected_stages
    report["status"] = (
        "OUTER_CANDIDATE_READY"
        if qc["status"] == "PASS_ENGINEERING" and not rejected_stages
        else "OUTER_REVIEW_REQUIRED"
    )

    p = _write_stage(current, out_path, "05_outer_candidate")
    if p:
        report["artifacts"]["outer_candidate"] = p
    if out_path is not None:
        report_path = out_path / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["artifacts"]["report"] = str(report_path)

    return current, report
