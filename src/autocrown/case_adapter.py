from __future__ import annotations

from pathlib import Path
import json
import re
import uuid

import numpy as np
import trimesh

from .config import PipelineConfig
from .frame import PatientFrame
from .localization import localize_case_geometry
from .meshops import load_mesh, save_mesh
from .pipeline import build_outer_crown


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def template_dir() -> Path:
    return repo_root() / "templates" / "32牙位基础解剖模板库" / "32位牙列-分"


def testdata_dir() -> Path:
    return repo_root() / "testdata" / "1.0测试数据"


def work_dir() -> Path:
    p = repo_root() / "work"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        raise ValueError("zero-length direction")
    return v / n


def list_test_cases() -> list[dict]:
    root = testdata_dir()
    out = []
    if not root.exists():
        return out
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name == "代码":
            continue
        match = re.search(r"牙位\s*(\d{2})", folder.name)
        if not match:
            continue
        fdi = match.group(1)
        stls = sorted(folder.glob("*.stl"))
        out.append({"name": folder.name, "fdi": fdi, "files": [p.name for p in stls]})
    return out


def template_path(fdi: str) -> Path:
    p = template_dir() / f"FDI_{str(fdi)}.stl"
    if not p.exists():
        raise FileNotFoundError(f"template not found for FDI {fdi}: {p}")
    return p


def _case_arch_paths(folder: Path, fdi: str) -> tuple[Path, Path]:
    stls = sorted(folder.glob("*.stl"))
    if len(stls) < 2:
        raise ValueError(f"case needs upper/lower STL files: {folder}")
    target_upper = str(fdi)[0] in {"1", "2"}
    target_word = "上颌" if target_upper else "下颌"
    opponent_word = "下颌" if target_upper else "上颌"
    target = next((p for p in stls if target_word in p.name and "基台" in p.name), None)
    if target is None:
        target = next((p for p in stls if target_word in p.name), None)
    opponent = next((p for p in stls if opponent_word in p.name), None)
    if target is None or opponent is None:
        raise ValueError(f"cannot identify target/opponent arches in {folder}")
    return target, opponent


def create_case_from_testdata(name: str) -> dict:
    folder = testdata_dir() / name
    if not folder.exists():
        raise FileNotFoundError(name)
    match = re.search(r"牙位\s*(\d{2})", name)
    if not match:
        raise ValueError("test case name does not contain FDI")
    fdi = match.group(1)
    target, opponent = _case_arch_paths(folder, fdi)
    case_id = uuid.uuid4().hex[:12]
    case_folder = work_dir() / case_id
    case_folder.mkdir(parents=True, exist_ok=True)
    record = {
        "id": case_id,
        "name": name,
        "source": "testdata",
        "fdi": fdi,
        "target_arch": str(target.resolve()),
        "opponent_arch": str(opponent.resolve()),
        "template": str(template_path(fdi).resolve()),
        "status": "READY_FOR_LOCALIZATION",
    }
    _save_record(record)
    return record


def save_uploaded_case(
    *,
    fdi: str,
    upper_bytes: bytes,
    lower_bytes: bytes,
    upper_name: str = "upper.stl",
    lower_name: str = "lower.stl",
) -> dict:
    case_id = uuid.uuid4().hex[:12]
    case_folder = work_dir() / case_id
    case_folder.mkdir(parents=True, exist_ok=True)
    upper = case_folder / upper_name
    lower = case_folder / lower_name
    upper.write_bytes(upper_bytes)
    lower.write_bytes(lower_bytes)
    target_upper = str(fdi)[0] in {"1", "2"}
    target, opponent = (upper, lower) if target_upper else (lower, upper)
    record = {
        "id": case_id,
        "name": f"上传病例 FDI {fdi}",
        "source": "upload",
        "fdi": str(fdi),
        "target_arch": str(target.resolve()),
        "opponent_arch": str(opponent.resolve()),
        "template": str(template_path(str(fdi)).resolve()),
        "status": "READY_FOR_LOCALIZATION",
    }
    _save_record(record)
    return record


def _save_record(record: dict) -> None:
    case_folder = work_dir() / record["id"]
    case_folder.mkdir(parents=True, exist_ok=True)
    (case_folder / "case.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_case(case_id: str) -> dict:
    p = work_dir() / case_id / "case.json"
    if not p.exists():
        raise FileNotFoundError(case_id)
    return json.loads(p.read_text(encoding="utf-8"))


def _frame_payload(frame: PatientFrame) -> dict:
    return {
        "origin": np.asarray(frame.origin, float).tolist(),
        "x": np.asarray(frame.x, float).tolist(),
        "y": np.asarray(frame.y, float).tolist(),
        "z": np.asarray(frame.z, float).tolist(),
    }


def _frame_from_payload(value: dict) -> PatientFrame:
    return PatientFrame(
        origin=np.asarray(value["origin"], float),
        x=np.asarray(value["x"], float),
        y=np.asarray(value["y"], float),
        z=np.asarray(value["z"], float),
    )


def _apply_fdi_mesial_sign(frame: PatientFrame, fdi: str) -> tuple[PatientFrame, bool]:
    # Z points from Margin toward the antagonist and Y points to arch exterior.
    # Flip Q2/Q4 so +X follows the FDI mesial direction in the usual arch frame.
    quadrant = int(str(fdi)[0])
    if quadrant not in {2, 4}:
        return frame, False
    return PatientFrame(
        origin=np.asarray(frame.origin, float),
        x=-np.asarray(frame.x, float),
        y=-np.asarray(frame.y, float),
        z=np.asarray(frame.z, float),
    ), True


def localize_case(case_id: str) -> dict:
    record = load_case(case_id)
    case_folder = work_dir() / case_id
    out = case_folder / "output"
    out.mkdir(parents=True, exist_ok=True)

    target = load_mesh(record["target_arch"])
    opponent = load_mesh(record["opponent_arch"])
    jaw, support, margin, frame, report = localize_case_geometry(target, opponent)
    frame, flipped = _apply_fdi_mesial_sign(frame, record["fdi"])
    report["frame"]["fdi_mesial_sign_flip_applied"] = bool(flipped)
    report["frame"].update(_frame_payload(frame))
    report["margin_points"] = np.asarray(margin, float).tolist()

    save_mesh(support, out / "00_support_component.stl")
    save_mesh(jaw, out / "00_target_jaw_component.stl")
    (out / "00_margin.json").write_text(
        json.dumps({"points": report["margin_points"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "00_localization.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    record["localization"] = report
    record["localization_assets"] = {
        "support": str((out / "00_support_component.stl").resolve()),
        "target_component": str((out / "00_target_jaw_component.stl").resolve()),
        "margin": str((out / "00_margin.json").resolve()),
        "report": str((out / "00_localization.json").resolve()),
    }
    record["status"] = (
        "LOCALIZATION_READY"
        if report["status"] == "READY"
        else "LOCALIZATION_REVIEW_REQUIRED"
    )
    record["output"] = str(out.resolve())
    _save_record(record)
    return record


def _template_orientation_hints(fdi: str) -> dict:
    fdi = str(fdi)
    quadrant = int(fdi[0])
    pos = int(fdi[1])
    target = load_mesh(template_path(fdi))
    target_center = np.asarray(target.centroid, float)

    mesial = None
    if pos > 1:
        mp = template_dir() / f"FDI_{quadrant}{pos - 1}.stl"
        if mp.exists():
            c = np.asarray(load_mesh(mp).centroid, float)
            if np.linalg.norm(c - target_center) > 0.5:
                mesial = c - target_center

    arch_quadrants = {1, 2} if quadrant in {1, 2} else {3, 4}
    centers = []
    for q in arch_quadrants:
        for p in range(1, 9):
            fp = template_dir() / f"FDI_{q}{p}.stl"
            if fp.exists():
                centers.append(np.asarray(load_mesh(fp).centroid, float))
    buccal = None
    if len(centers) >= 8:
        arch_center = np.mean(np.asarray(centers), axis=0)
        radial = target_center - arch_center
        if np.linalg.norm(radial) > 0.5:
            buccal = radial

    return {"mesial": mesial, "buccal": buccal}


def _canonicalize_full_tooth(
    template: trimesh.Trimesh,
    fdi: str,
) -> tuple[trimesh.Trimesh, dict]:
    """Extract a crown prior and orient it to +X mesial/+Y buccal/+Z occlusal."""
    v = np.asarray(template.vertices, float)
    c = v.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov((v - c).T))
    z_axis = _unit(vecs[:, int(np.argmax(vals))])
    zraw = (v - c) @ z_axis

    def radial_spread(mask, axis):
        pts = v[mask] - c
        axial = np.outer(pts @ axis, axis)
        return float(np.median(np.linalg.norm(pts - axial, axis=1))) if int(mask.sum()) else 0.0

    lo = zraw <= np.quantile(zraw, 0.16)
    hi = zraw >= np.quantile(zraw, 0.84)
    if radial_spread(lo, z_axis) > radial_spread(hi, z_axis):
        z_axis = -z_axis
        zraw = -zraw

    e1 = np.array([1.0, 0.0, 0.0])
    e1 -= z_axis * float(e1 @ z_axis)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([0.0, 1.0, 0.0])
        e1 -= z_axis * float(e1 @ z_axis)
    e1 = _unit(e1)
    e2 = _unit(np.cross(z_axis, e1))
    radial = np.sqrt(((v - c) @ e1) ** 2 + ((v - c) @ e2) ** 2)
    qlo, qhi = np.quantile(zraw, [0.02, 0.98])
    edges = np.linspace(float(qlo), float(qhi), 48)
    mids = (edges[:-1] + edges[1:]) * 0.5
    profile = np.full(len(mids), np.nan)
    for i in range(len(mids)):
        mask = (zraw >= edges[i]) & (zraw < edges[i + 1])
        if int(mask.sum()) >= 8:
            profile[i] = np.quantile(radial[mask], 0.78)
    good = np.isfinite(profile)
    if good.sum() >= 10:
        ids = np.arange(len(profile))
        profile[~good] = np.interp(ids[~good], ids[good], profile[good])
        from scipy.ndimage import gaussian_filter1d
        smooth = gaussian_filter1d(profile, 1.4)
        zn = (mids - qlo) / max(qhi - qlo, 1e-6)
        allowed = (zn >= 0.28) & (zn <= 0.62)
        neck_id = int(np.argmin(np.where(allowed, smooth, np.inf)))
        crown_cut = float(mids[neck_id])
        cut_mode = "RADIAL_NECK_PROFILE"
    else:
        crown_cut = float(np.quantile(zraw, 0.46))
        cut_mode = "QUANTILE_FALLBACK"

    face_z = zraw[np.asarray(template.faces, int)].mean(axis=1)
    keep = face_z >= crown_cut
    crown = template.submesh([np.where(keep)[0]], append=True, repair=False)
    if crown is None or len(crown.faces) < max(100, int(0.18 * len(template.faces))):
        crown_cut = float(np.quantile(zraw, 0.44))
        keep = face_z >= crown_cut
        crown = template.submesh([np.where(keep)[0]], append=True, repair=False)
        cut_mode += "_SAFE_FALLBACK"
    if crown is None or len(crown.faces) < 100:
        raise ValueError("template crown extraction produced insufficient geometry")

    hints = _template_orientation_hints(fdi)
    cv = np.asarray(crown.vertices, float)
    cc = cv.mean(axis=0)
    centered = cv - cc

    x_hint = hints.get("mesial")
    orientation_source = "LIBRARY_NEIGHBOR_CENTROID"
    if x_hint is not None:
        x_axis = np.asarray(x_hint, float)
        x_axis -= z_axis * float(x_axis @ z_axis)
    else:
        plane = centered - np.outer(centered @ z_axis, z_axis)
        _, pvecs = np.linalg.eigh(np.cov(plane.T))
        x_axis = pvecs[:, -1]
        x_axis -= z_axis * float(x_axis @ z_axis)
        orientation_source = "IN_PLANE_PCA_FALLBACK"
    x_axis = _unit(x_axis)
    y_axis = _unit(np.cross(z_axis, x_axis))
    x_axis = _unit(np.cross(y_axis, z_axis))

    b_hint = hints.get("buccal")
    if b_hint is not None:
        b = np.asarray(b_hint, float)
        b -= z_axis * float(b @ z_axis)
        if np.linalg.norm(b) > 1e-6 and float(y_axis @ b) < 0:
            x_axis = -x_axis
            y_axis = -y_axis

    local = np.column_stack((centered @ x_axis, centered @ y_axis, centered @ z_axis))
    local[:, 2] -= float(np.quantile(local[:, 2], 0.02))
    local[:, 0] -= float(np.median(local[:, 0]))
    local[:, 1] -= float(np.median(local[:, 1]))
    crown.vertices = local

    return crown, {
        "mode": "FULL_TOOTH_TO_CANONICAL_CROWN_V2",
        "source_faces": int(len(template.faces)),
        "crown_faces": int(len(crown.faces)),
        "cut_value_axis_mm": float(crown_cut),
        "cut_mode": cut_mode,
        "orientation_source": orientation_source,
        "canonical_axes": "+X mesial, +Y buccal, +Z occlusal/insertion",
    }


def _automatic_neighbor_patch(
    jaw: trimesh.Trimesh,
    frame: PatientFrame,
    margin: np.ndarray,
    side: str,
) -> tuple[trimesh.Trimesh | None, dict]:
    centers = np.asarray(jaw.triangles_center, float)
    local = frame.to_local(centers)
    ml = frame.to_local(np.asarray(margin, float))
    half_x = max(float(np.ptp(ml[:, 0])) / 2.0, 0.8)
    half_y = max(float(np.ptp(ml[:, 1])) / 2.0, 0.8)
    z_hi = float(np.quantile(ml[:, 2], 0.90))
    sign = 1.0 if side == "mesial" else -1.0
    sx = sign * local[:, 0]
    mask = (
        (sx > half_x + 0.15)
        & (sx < half_x + 10.5)
        & (np.abs(local[:, 1]) < half_y + 6.0)
        & (local[:, 2] > z_hi - 0.3)
        & (local[:, 2] < z_hi + 9.0)
    )
    ids = np.flatnonzero(mask)
    if len(ids) < 8:
        return None, {"status": "ABSENT_OR_UNRESOLVED", "candidate_faces": int(len(ids))}
    nearest = float(np.quantile(sx[ids], 0.06))
    ids = np.flatnonzero(mask & (sx <= nearest + 4.0))
    if len(ids) < 8:
        return None, {"status": "UNRESOLVED", "candidate_faces": int(len(ids))}
    region = jaw.submesh([ids], append=True, repair=False)
    parts = list(region.split(only_watertight=False))
    if parts:
        parts.sort(
            key=lambda m: float(
                np.linalg.norm(np.asarray(m.vertices, float).mean(axis=0) - frame.origin)
            )
        )
        region = parts[0]
    if len(region.faces) < 8:
        return None, {"status": "UNRESOLVED", "candidate_faces": int(len(ids))}
    return region, {
        "status": "READY",
        "method": "PATIENT_FRAME_MD_BOUNDED_CONNECTED_PATCH_V1",
        "faces": int(len(region.faces)),
        "side": side,
    }


def generate_outer_for_case(case_id: str) -> dict:
    record = load_case(case_id)
    if not record.get("localization"):
        record = localize_case(case_id)

    case_folder = work_dir() / case_id
    out = case_folder / "output"
    out.mkdir(parents=True, exist_ok=True)
    opponent = load_mesh(record["opponent_arch"])
    raw_template = load_mesh(record["template"])

    loc = record["localization"]
    frame = _frame_from_payload(loc["frame"])
    margin = np.asarray(loc["margin_points"], float)
    jaw = load_mesh(record["localization_assets"]["target_component"])

    template, template_report = _canonicalize_full_tooth(raw_template, record["fdi"])
    save_mesh(template, out / "00_template_canonical.stl")

    mesial, mesial_report = _automatic_neighbor_patch(jaw, frame, margin, "mesial")
    distal, distal_report = _automatic_neighbor_patch(jaw, frame, margin, "distal")
    if mesial is not None:
        save_mesh(mesial, out / "00_neighbor_mesial.stl")
    if distal is not None:
        save_mesh(distal, out / "00_neighbor_distal.stl")

    crown, report = build_outer_crown(
        template,
        margin,
        frame,
        mesial=mesial,
        distal=distal,
        antagonist=opponent,
        config=PipelineConfig(),
        out_dir=out,
    )
    report["localization"] = loc
    report["template_normalization"] = template_report
    report["neighbor_detection"] = {"mesial": mesial_report, "distal": distal_report}
    report["limitations"] = [
        "AUTO_MARGIN_IS_ENGINEERING_CANDIDATE_NOT_HUMAN_CONFIRMED",
        "FINAL_INTAGLIO_NOT_IN_THIS_OUTER_MVP",
    ]
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record["status"] = report["status"]
    record["output"] = str(out.resolve())
    record["report"] = report
    _save_record(record)
    return record
