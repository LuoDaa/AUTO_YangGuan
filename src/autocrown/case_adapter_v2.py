from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import uuid

import numpy as np
import trimesh

from .config import PipelineConfig
from .frame import PatientFrame
from .localization_runtime import localize_case_geometry
from .template_normalization import canonicalize_full_tooth as _canonicalize_full_tooth
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
    if name not in {row["name"] for row in list_test_cases()}:
        raise FileNotFoundError("unknown test case")
    if not folder.is_dir():
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
    # Separate fixed asset names: two uploads with the same basename must not
    # overwrite one another and change the registered bite geometry.
    upper = folder / "upper.stl"
    lower = folder / "lower.stl"
    if not upper_bytes or not lower_bytes:
        raise ValueError("both jaw STL files must be non-empty")
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
    if not re.fullmatch(r"[0-9a-f]{12}", str(case_id)):
        raise FileNotFoundError("invalid case identifier")
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


def _source_signature(record: dict) -> str:
    """Bind generated geometry to the exact bytes of both scans and template."""
    digest = hashlib.sha256()
    digest.update(str(record["fdi"]).encode("ascii"))
    for key in ("target_arch", "opponent_arch", "template"):
        path = Path(record[key])
        if not path.is_file():
            raise FileNotFoundError(f"case input missing: {key}")
        digest.update(key.encode("ascii"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _localization_signature(input_digest: str, fdi: str, report: dict) -> str:
    data = {
        "input_digest": input_digest,
        "fdi": str(fdi),
        "margin_points": report["margin_points"],
        "frame": report["frame"],
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def acknowledge_localization(case_id: str) -> dict:
    """Persist a visible operator acknowledgment bound to immutable geometry."""
    record = load_case(case_id)
    token = record.get("localization_token")
    if not token or not record.get("localization"):
        raise ValueError("localization must be completed before acknowledgment")
    if _source_signature(record) != record.get("input_signature"):
        raise ValueError("input STL/template changed after localization; localize again")
    record["acknowledged_localization_token"] = token
    _save_record(record)
    return record


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
    source_signature = _source_signature(record)
    record["input_signature"] = source_signature
    record["localization_token"] = _localization_signature(
        source_signature, record["fdi"], report
    )
    # Never carry a prior visual acknowledgment across a fresh localization.
    record.pop("acknowledged_localization_token", None)
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


def generate_outer_for_case(case_id: str, *, require_acknowledgement: bool = False) -> dict:
    record = load_case(case_id)
    if not record.get("localization"):
        record = localize_case(case_id)
    if _source_signature(record) != record.get("input_signature"):
        raise ValueError("source geometry changed after localization; relocalize")
    acknowledged = bool(
        record.get("localization_token")
        and record.get("localization_token") == record.get("acknowledged_localization_token")
    )
    if require_acknowledgement and not acknowledged:
        raise ValueError("请先在 3D 中核对基台、Margin 和 XYZ，再确认定位")

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
    neighbor_review = [
        side for side, row in (("mesial", mesial_report), ("distal", distal_report))
        if row.get("status") != "READY"
    ]
    caution = []
    if localization.get("status") != "READY":
        caution.append("LOCALIZATION_REVIEW_REQUIRED")
    if not template_report.get("clinical_cervical_valid", False):
        caution.append("TEMPLATE_CERVICAL_SOURCE_NOT_PROVEN")
    if neighbor_review:
        caution.append("NEIGHBOR_REFERENCE_UNVERIFIED:" + ",".join(neighbor_review))
    if report["status"] != "OUTER_CANDIDATE_READY":
        caution.append("OUTER_MORPHOLOGY_QC_REVIEW_REQUIRED")
    report["readiness"] = {
        "status": "REVIEW_REQUIRED" if caution else "PASS_ENGINEERING",
        "reasons": caution,
        "localization_token": record.get("localization_token"),
        "localization_acknowledged": acknowledged,
        "template_cervical_valid": bool(template_report.get("clinical_cervical_valid")),
        "clinical_manufacturing_approved": False,
    }
    if caution:
        report["status"] = "OUTER_REVIEW_REQUIRED"
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
