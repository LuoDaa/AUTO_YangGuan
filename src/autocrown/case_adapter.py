from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import uuid

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .frame import build_confirmed_frame
from .meshops import load_mesh, save_mesh
from .pipeline import build_outer_crown
from .config import PipelineConfig


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


def _mesh_components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    parts = list(mesh.split(only_watertight=False))
    parts.sort(key=lambda m: len(m.faces), reverse=True)
    return parts


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
        out.append({
            "name": folder.name,
            "fdi": fdi,
            "files": [p.name for p in stls],
        })
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
        "status": "READY_FOR_GENERATION",
    }
    (case_folder / "case.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def save_uploaded_case(*, fdi: str, upper_bytes: bytes, lower_bytes: bytes,
                       upper_name: str = "upper.stl", lower_name: str = "lower.stl") -> dict:
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
        "status": "READY_FOR_GENERATION",
    }
    (case_folder / "case.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def load_case(case_id: str) -> dict:
    p = work_dir() / case_id / "case.json"
    if not p.exists():
        raise FileNotFoundError(case_id)
    return json.loads(p.read_text(encoding="utf-8"))


def _opponent_oriented_axis(support: trimesh.Trimesh, opponent: trimesh.Trimesh) -> np.ndarray:
    v = np.asarray(support.vertices, float)
    c = v.mean(axis=0)
    cov = np.cov((v - c).T)
    vals, vecs = np.linalg.eigh(cov)
    axis = _unit(vecs[:, int(np.argmax(vals))])
    span = float(np.ptp((v - c) @ axis))
    tree = cKDTree(np.asarray(opponent.vertices, float)[:: max(1, len(opponent.vertices) // 25000)])
    d_plus = float(tree.query(c + axis * span * 0.55)[0])
    d_minus = float(tree.query(c - axis * span * 0.55)[0])
    return axis if d_plus < d_minus else -axis


def _estimate_frame_margin(target: trimesh.Trimesh, opponent: trimesh.Trimesh):
    parts = _mesh_components(target)
    if len(parts) < 2:
        raise ValueError("target arch does not contain a separable abutment component")
    jaw, support = parts[0], parts[1]
    axis = _opponent_oriented_axis(support, opponent)
    sc = np.asarray(support.centroid, float)
    jc = np.asarray(jaw.centroid, float)
    buccal = sc - jc
    buccal = buccal - axis * float(np.dot(buccal, axis))
    if np.linalg.norm(buccal) < 1e-6:
        verts = np.asarray(jaw.vertices, float)
        cov = np.cov((verts - verts.mean(axis=0)).T)
        vals, vecs = np.linalg.eigh(cov)
        buccal = vecs[:, int(np.argsort(vals)[-2])]
        buccal -= axis * float(np.dot(buccal, axis))
    buccal = _unit(buccal)
    mesial = _unit(np.cross(buccal, axis))

    sv = np.asarray(support.vertices, float)
    z = (sv - sc) @ axis
    z0 = float(np.quantile(z, 0.08))
    band = np.abs(z - z0) <= max(0.35, 0.08 * float(np.ptp(z)))
    if int(band.sum()) < 12:
        band = z <= np.quantile(z, 0.18)
    base_center = sv[band].mean(axis=0)
    rel = sv[band] - base_center
    r = np.sqrt((rel @ mesial) ** 2 + (rel @ buccal) ** 2)
    radius = float(np.quantile(r, 0.88)) if len(r) else 2.0
    radius = float(np.clip(radius, 1.5, 6.5))
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    margin = (base_center[None, :] + radius * np.cos(theta)[:, None] * mesial[None, :]
              + radius * np.sin(theta)[:, None] * buccal[None, :])
    frame = build_confirmed_frame(margin, mesial, buccal, axis)
    return jaw, support, margin, frame, {
        "support_component_faces": int(len(support.faces)),
        "jaw_component_faces": int(len(jaw.faces)),
        "estimated_margin_radius_mm": radius,
        "mode": "RAW_CASE_FAST_PROXY_V1",
    }


def _canonicalize_full_tooth(template: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    """Convert a full natural-tooth template into a crown-like canonical prior.

    This is intentionally conservative and only for the 20-40 point MVP. The
    long PCA axis is treated as root-to-crown. The broader end is taken as the
    crown side; the rootward half is discarded before patient placement.
    """
    v = np.asarray(template.vertices, float)
    c = v.mean(axis=0)
    cov = np.cov((v - c).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    z_axis = _unit(vecs[:, order[0]])
    zraw = (v - c) @ z_axis
    lo = zraw <= np.quantile(zraw, 0.16)
    hi = zraw >= np.quantile(zraw, 0.84)
    def radial_spread(mask):
        pts = v[mask] - c
        axial = np.outer(pts @ z_axis, z_axis)
        return float(np.median(np.linalg.norm(pts - axial, axis=1))) if int(mask.sum()) else 0.0
    if radial_spread(lo) > radial_spread(hi):
        z_axis = -z_axis
        zraw = -zraw

    crown_cut = float(np.quantile(zraw, 0.46))
    face_z = zraw[np.asarray(template.faces, int)].mean(axis=1)
    keep = face_z >= crown_cut
    crown = template.submesh([np.where(keep)[0]], append=True, repair=False)
    if crown is None or len(crown.faces) < 100:
        crown = template.copy()
    cv = np.asarray(crown.vertices, float)
    cc = cv.mean(axis=0)
    centered = cv - cc
    # Choose the widest in-plane PCA direction as canonical mesial X.
    plane = centered - np.outer(centered @ z_axis, z_axis)
    pcov = np.cov(plane.T)
    pvals, pvecs = np.linalg.eigh(pcov)
    x_axis = pvecs[:, int(np.argmax(pvals))]
    x_axis = _unit(x_axis - z_axis * float(np.dot(x_axis, z_axis)))
    y_axis = _unit(np.cross(z_axis, x_axis))
    x_axis = _unit(np.cross(y_axis, z_axis))
    local = np.column_stack((centered @ x_axis, centered @ y_axis, centered @ z_axis))
    local[:, 2] -= float(np.quantile(local[:, 2], 0.03))
    local[:, 0] -= float(np.median(local[:, 0]))
    local[:, 1] -= float(np.median(local[:, 1]))
    crown.vertices = local
    return crown, {
        "mode": "FULL_TOOTH_TO_CROWN_PCA_PROXY_V1",
        "source_faces": int(len(template.faces)),
        "crown_faces": int(len(crown.faces)),
        "cut_quantile": 0.46,
    }


def generate_outer_for_case(case_id: str) -> dict:
    record = load_case(case_id)
    case_folder = work_dir() / case_id
    out = case_folder / "output"
    out.mkdir(parents=True, exist_ok=True)
    target = load_mesh(record["target_arch"])
    opponent = load_mesh(record["opponent_arch"])
    raw_template = load_mesh(record["template"])
    template, template_report = _canonicalize_full_tooth(raw_template)
    jaw, support, margin, frame, adapter_report = _estimate_frame_margin(target, opponent)
    save_mesh(template, out / "00_template_canonical.stl")
    save_mesh(support, out / "00_support_component.stl")
    save_mesh(jaw, out / "00_target_jaw_component.stl")
    np.savetxt(out / "00_margin_proxy.xyz", margin, fmt="%.7f")

    crown, report = build_outer_crown(
        template,
        margin,
        frame,
        antagonist=opponent,
        config=PipelineConfig(),
        out_dir=out,
    )
    report["raw_case_adapter"] = adapter_report
    report["template_normalization"] = template_report
    report["limitations"] = [
        "MARGIN_IS_FAST_PROXY_NOT_CLINICAL_FINAL",
        "PROXIMAL_NEIGHBOUR_SEGMENTATION_NOT_YET_ENABLED",
        "INTAGLIO_NOT_IN_THIS_OUTER_MVP",
    ]
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    record["status"] = report["status"]
    record["output"] = str(out.resolve())
    record["report"] = report
    (case_folder / "case.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
