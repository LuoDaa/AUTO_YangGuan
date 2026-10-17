from __future__ import annotations

from pathlib import Path
import json
import re
import uuid

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d

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
    path = repo_root() / "work"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unit(value: np.ndarray, name: str = "vector") -> np.ndarray:
    value = np.asarray(value, float)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or norm < 1e-10:
        raise ValueError(f"{name} is degenerate")
    return value / norm


def list_test_cases() -> list[dict]:
    root = testdata_dir()
    out: list[dict] = []
    if not root.exists():
        return out
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name == "代码":
            continue
        match = re.search(r"牙位\s*(\d{2})", folder.name)
        if not match:
            continue
        stls = sorted(folder.glob("*.stl"))
        out.append({
            "name": folder.name,
            "fdi": match.group(1),
            "files": [p.name for p in stls],
        })
    return out


def template_path(fdi: str) -> Path:
    path = template_dir() / f"FDI_{str(fdi)}.stl"
    if not path.exists():
        raise FileNotFoundError(f"template not found for FDI {fdi}: {path}")
    return path


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


def _save_record(record: dict) -> None:
    folder = work_dir() / record["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "case.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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
    if not re.fullmatch(r"[1-4][1-8]", str(fdi)):
        raise ValueError("FDI must be 11-18, 21-28, 31-38 or 41-48")
    case_id = uuid.uuid4().hex[:12]
    folder = work_dir() / case_id
    folder.mkdir(parents=True, exist_ok=True)
    upper = folder / (Path(upper_name).name or "upper.stl")
    lower = folder / (Path(lower_name).name or "lower.stl")
    upper.write_bytes(upper_bytes)
    lower.write_bytes(lower_bytes)
    target_upper = str(fdi)[0] in {"1", "2"}
    target, opponent = (upper, lower) if target_upper else (lower, upper)
    # Load once here so invalid uploads fail before a case is advertised as ready.
    load_mesh(target)
    load_mesh(opponent)
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


def load_case(case_id: str) -> dict:
    path = work_dir() / case_id / "case.json"
    if not path.exists():
        raise FileNotFoundError(case_id)
    return json.loads(path.read_text(encoding="utf-8"))


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


def mesial_sign_for_fdi(fdi: str) -> float:
    """Map the right-handed arch tangent X to the anatomical mesial side.

    PatientFrame is kept stable: Y always points arch-outward/buccal and Z points
    from the finish line toward the antagonist. Depending on quadrant, anatomical
    mesial is therefore +X or -X. We pass this sign to contact logic instead of
    flipping X+Y and accidentally turning buccal into lingual.
    """
    quadrant = int(str(fdi)[0])
    return 1.0 if quadrant in {1, 3} else -1.0


def localize_case(case_id: str) -> dict:
    record = load_case(case_id)
    out = work_dir() / case_id / "output"
    out.mkdir(parents=True, exist_ok=True)

    target = load_mesh(record["target_arch"])
    opponent = load_mesh(record["opponent_arch"])
    jaw, support, margin, frame, report = localize_case_geometry(target, opponent)
    sign = mesial_sign_for_fdi(record["fdi"])
    report["frame"].update(_frame_payload(frame))
    report["frame"].update({
        "x_semantics": "RIGHT_HANDED_ARCH_TANGENT",
        "y_semantics": "BUCCAL_ARCH_OUTWARD",
        "z_semantics": "INSERTION_TOWARD_ANTAGONIST",
        "mesial_sign_on_x": float(sign),
    })
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
        "LOCALIZATION_READY" if report["status"] == "READY"
        else "LOCALIZATION_REVIEW_REQUIRED"
    )
    record["output"] = str(out.resolve())
    # Localization changes invalidate any old generated candidate.
    record.pop("report", None)
    _save_record(record)
    return record


def _library_arch_centers(quadrant: int) -> list[np.ndarray]:
    quadrants = {1, 2} if quadrant in {1, 2} else {3, 4}
    centers: list[np.ndarray] = []
    for q in sorted(quadrants):
        for pos in range(1, 9):
            path = template_dir() / f"FDI_{q}{pos}.stl"
            if not path.exists():
                continue
            try:
                centers.append(np.asarray(load_mesh(path).centroid, float))
            except Exception:
                continue
    return centers


def _template_buccal_hint(fdi: str, z_axis: np.ndarray) -> np.ndarray | None:
    quadrant = int(str(fdi)[0])
    target = np.asarray(load_mesh(template_path(fdi)).centroid, float)
    centers = _library_arch_centers(quadrant)
    if len(centers) < 6:
        return None
    arch_center = np.mean(np.asarray(centers), axis=0)
    radial = target - arch_center
    radial -= z_axis * float(radial @ z_axis)
    if float(np.linalg.norm(radial)) < 0.5:
        return None
    return _unit(radial, "template buccal")


def _canonicalize_full_tooth(
    template: trimesh.Trimesh,
    fdi: str,
) -> tuple[trimesh.Trimesh, dict]:
    """Extract a crown prior in the same right-handed frame used by patients.

    +Z = crown/occlusal direction, +Y = library arch outward (buccal),
    +X = Y×Z right-handed arch tangent. Anatomical mesial is stored separately as
    a quadrant-dependent sign; this avoids a 180° buccal/lingual flip.
    """
    vertices = np.asarray(template.vertices, float)
    center = vertices.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((vertices - center).T))
    z_axis = _unit(vectors[:, int(np.argmax(values))], "template long axis")
    axial = (vertices - center) @ z_axis

    def radial_spread(mask: np.ndarray, axis: np.ndarray) -> float:
        points = vertices[mask] - center
        axial_part = np.outer(points @ axis, axis)
        return float(np.median(np.linalg.norm(points - axial_part, axis=1))) if int(mask.sum()) else 0.0

    low = axial <= np.quantile(axial, 0.16)
    high = axial >= np.quantile(axial, 0.84)
    # Crown side is normally broader than the root-apical side.
    if radial_spread(low, z_axis) > radial_spread(high, z_axis):
        z_axis = -z_axis
        axial = -axial

    # Estimate cervical neck from the radius-vs-axis profile instead of deleting a
    # fixed half of the complete natural tooth.
    helper = np.array([1.0, 0.0, 0.0])
    helper -= z_axis * float(helper @ z_axis)
    if float(np.linalg.norm(helper)) < 1e-6:
        helper = np.array([0.0, 1.0, 0.0])
        helper -= z_axis * float(helper @ z_axis)
    e1 = _unit(helper)
    e2 = _unit(np.cross(z_axis, e1))
    rel = vertices - center
    radius = np.sqrt((rel @ e1) ** 2 + (rel @ e2) ** 2)
    qlo, qhi = np.quantile(axial, [0.02, 0.98])
    edges = np.linspace(float(qlo), float(qhi), 52)
    mids = (edges[:-1] + edges[1:]) * 0.5
    profile = np.full(len(mids), np.nan)
    for i in range(len(mids)):
        mask = (axial >= edges[i]) & (axial < edges[i + 1])
        if int(mask.sum()) >= 8:
            profile[i] = np.quantile(radius[mask], 0.78)
    good = np.isfinite(profile)
    if int(good.sum()) >= 10:
        ids = np.arange(len(profile))
        profile[~good] = np.interp(ids[~good], ids[good], profile[good])
        smooth = gaussian_filter1d(profile, 1.35)
        zn = (mids - qlo) / max(float(qhi - qlo), 1e-6)
        allowed = (zn >= 0.24) & (zn <= 0.66)
        neck_id = int(np.argmin(np.where(allowed, smooth, np.inf)))
        crown_cut = float(mids[neck_id])
        cut_mode = "RADIAL_NECK_PROFILE"
    else:
        crown_cut = float(np.quantile(axial, 0.44))
        cut_mode = "QUANTILE_FALLBACK"

    face_axial = axial[np.asarray(template.faces, int)].mean(axis=1)
    keep = face_axial >= crown_cut
    crown = template.submesh([np.flatnonzero(keep)], append=True, repair=False)
    if crown is None or len(crown.faces) < max(100, int(0.15 * len(template.faces))):
        crown_cut = float(np.quantile(axial, 0.40))
        crown = template.submesh([np.flatnonzero(face_axial >= crown_cut)], append=True, repair=False)
        cut_mode += "_SAFE_FALLBACK"
    if crown is None or len(crown.faces) < 100:
        raise ValueError("template crown extraction produced insufficient geometry")

    cv = np.asarray(crown.vertices, float)
    cc = cv.mean(axis=0)
    centered = cv - cc
    y_hint = _template_buccal_hint(str(fdi), z_axis)
    orientation_source = "LIBRARY_ARCH_RADIAL"
    if y_hint is None:
        # Fallback keeps a right-handed basis but is explicitly reported for review.
        plane = centered - np.outer(centered @ z_axis, z_axis)
        _, pvecs = np.linalg.eigh(np.cov(plane.T))
        y_axis = pvecs[:, -1]
        y_axis -= z_axis * float(y_axis @ z_axis)
        y_axis = _unit(y_axis, "template transverse fallback")
        orientation_source = "IN_PLANE_PCA_REVIEW_REQUIRED"
    else:
        y_axis = y_hint
    x_axis = _unit(np.cross(y_axis, z_axis), "template arch tangent")
    y_axis = _unit(np.cross(z_axis, x_axis), "template buccal orthogonalized")

    local = np.column_stack((centered @ x_axis, centered @ y_axis, centered @ z_axis))
    local[:, 2] -= float(np.quantile(local[:, 2], 0.02))
    local[:, 0] -= float(np.median(local[:, 0]))
    local[:, 1] -= float(np.median(local[:, 1]))
    crown.vertices = local

    return crown, {
        "mode": "FULL_TOOTH_TO_CANONICAL_CROWN_V3",
        "source_faces": int(len(template.faces)),
        "crown_faces": int(len(crown.faces)),
        "cut_value_axis_mm": float(crown_cut),
        "cut_mode": cut_mode,
        "orientation_source": orientation_source,
        "canonical_axes": "+X arch-tangent, +Y buccal/outward, +Z occlusal/insertion",
        "mesial_sign_on_x": float(mesial_sign_for_fdi(str(fdi))),
    }


def _automatic_neighbor_patch(
    jaw: trimesh.Trimesh,
    frame: PatientFrame,
    margin: np.ndarray,
    side: str,
    *,
    mesial_sign: float,
) -> tuple[trimesh.Trimesh | None, dict]:
    centers = np.asarray(jaw.triangles_center, float)
    local = frame.to_local(centers)
    margin_local = frame.to_local(np.asarray(margin, float))
    half_x = max(float(np.ptp(margin_local[:, 0])) / 2.0, 0.8)
    half_y = max(float(np.ptp(margin_local[:, 1])) / 2.0, 0.8)
    z_hi = float(np.quantile(margin_local[:, 2], 0.90))
    anatomical_sign = float(mesial_sign) if side == "mesial" else -float(mesial_sign)
    side_x = anatomical_sign * local[:, 0]
    presence = (
        (side_x > half_x + 0.12)
        & (side_x < half_x + 11.0)
        & (np.abs(local[:, 1]) < half_y + 6.5)
        & (local[:, 2] > z_hi - 0.8)
        & (local[:, 2] < z_hi + 9.5)
    )
    candidate_ids = np.flatnonzero(presence)
    if len(candidate_ids) < 8:
        return None, {
            "status": "ABSENT_OR_UNRESOLVED",
            "side": side,
            "candidate_faces": int(len(candidate_ids)),
            "mesial_sign": float(mesial_sign),
        }

    nearest_x = float(np.quantile(side_x[candidate_ids], 0.06))
    bounded = presence & (side_x <= nearest_x + 4.5)
    ids = np.flatnonzero(bounded)
    if len(ids) < 8:
        return None, {"status": "UNRESOLVED", "side": side, "candidate_faces": int(len(ids))}

    subset = jaw.submesh([ids], append=True, repair=False)
    parts = list(subset.split(only_watertight=False))
    if parts:
        # Prefer the connected patch nearest the expected proximal side of Margin.
        target = frame.origin + frame.x * anatomical_sign * (half_x + 1.5)
        parts.sort(key=lambda part: float(np.min(np.linalg.norm(np.asarray(part.vertices) - target, axis=1))))
        subset = parts[0]
    if len(subset.faces) < 8:
        return None, {"status": "UNRESOLVED", "side": side, "candidate_faces": int(len(ids))}

    return subset, {
        "status": "READY",
        "method": "PATIENT_FRAME_FDI_SIDE_CONNECTED_PATCH_V2",
        "side": side,
        "mesial_sign": float(mesial_sign),
        "faces": int(len(subset.faces)),
        "nearest_side_coordinate_mm": nearest_x,
    }


def generate_outer_for_case(case_id: str) -> dict:
    record = load_case(case_id)
    if not record.get("localization"):
        record = localize_case(case_id)

    out = work_dir() / case_id / "output"
    out.mkdir(parents=True, exist_ok=True)
    opponent = load_mesh(record["opponent_arch"])
    raw_template = load_mesh(record["template"])
    localization = record["localization"]
    frame = _frame_from_payload(localization["frame"])
    margin = np.asarray(localization["margin_points"], float)
    jaw = load_mesh(record["localization_assets"]["target_component"])
    mesial_sign = mesial_sign_for_fdi(record["fdi"])

    template, template_report = _canonicalize_full_tooth(raw_template, record["fdi"])
    save_mesh(template, out / "00_template_canonical.stl")

    mesial, mesial_report = _automatic_neighbor_patch(
        jaw, frame, margin, "mesial", mesial_sign=mesial_sign,
    )
    distal, distal_report = _automatic_neighbor_patch(
        jaw, frame, margin, "distal", mesial_sign=mesial_sign,
    )
    if mesial is not None:
        save_mesh(mesial, out / "00_neighbor_mesial.stl")
    if distal is not None:
        save_mesh(distal, out / "00_neighbor_distal.stl")

    _, report = build_outer_crown(
        template,
        margin,
        frame,
        mesial=mesial,
        distal=distal,
        antagonist=opponent,
        mesial_sign=mesial_sign,
        config=PipelineConfig(),
        out_dir=out,
    )
    report["localization"] = localization
    report["template_normalization"] = template_report
    report["neighbor_detection"] = {
        "mesial": mesial_report,
        "distal": distal_report,
    }
    report["limitations"] = [
        "AUTO_MARGIN_AND_AXIS_ARE_ENGINEERING_CANDIDATES_REQUIRE_VISUAL_REVIEW",
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
