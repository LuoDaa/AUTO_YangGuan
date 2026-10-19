"""Behavioral regression for the CrownFlow-derived conservative geometry contracts."""
from pathlib import Path
import json

import numpy as np
import pytest
import trimesh

import autocrown.case_adapter as case_adapter
from autocrown.frame import axial_height_from_margin, build_confirmed_frame
from autocrown.geometry_guards import accept_anatomy_stage
from autocrown.template_cervical import _boundary_loops, extract_clean_crown_surface


def _frame_and_margin():
    corners = np.array([
        [-2.0, -2.0, 0.0], [2.0, -2.0, 2.0],
        [2.0, 2.0, 1.0], [-2.0, 2.0, -0.5],
    ])
    margin = np.vstack([
        corners[i] + t * (corners[(i + 1) % 4] - corners[i])
        for i in range(4) for t in np.linspace(0, 1, 4, endpoint=False)
    ])
    frame = build_confirmed_frame(margin, [1, 0, 0], [0, 1, 0], [0, 0, 1])
    return frame, margin


def test_nonplanar_margin_height_is_continuous_along_segments():
    frame, margin = _frame_and_margin()
    # Midpoint of the first edge owns Z=1, *not* one of its two end Zs.
    p = np.array([[0.0, -2.0, 1.50], [0.0, -2.0, 0.5]])
    result = axial_height_from_margin(p, margin, frame)
    assert np.allclose(result, [0.50, -0.50], atol=1e-8)
    assert np.allclose(
        axial_height_from_margin(margin, margin, frame), 0.0, atol=1e-8
    )


def test_geometry_gate_rejects_any_motion_of_confirmed_cervical_band():
    frame, margin = _frame_and_margin()
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh.apply_translation([0, 0, 0.2])
    before = np.asarray(mesh.vertices, float).copy()
    low = axial_height_from_margin(before, margin, frame) <= 0.45
    assert np.any(low)
    proposed = mesh.copy()
    v = np.asarray(proposed.vertices, float).copy()
    v[np.flatnonzero(low)[0], 2] += 0.4
    proposed.vertices = v

    accepted, gate = accept_anatomy_stage(
        mesh, proposed, margin, frame, stage="TEST",
        cervical_lock_mm=0.45, max_stage_move_mm=0.8,
    )
    assert gate["status"] == "REJECTED"
    assert "CONFIRMED_CERVICAL_BAND_MOVED" in gate["full_candidate"]["reasons"]
    assert np.array_equal(accepted.vertices, before)
    assert np.array_equal(mesh.vertices, before)


def test_geometry_gate_backs_off_excessive_anatomical_movement():
    frame, margin = _frame_and_margin()
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh.apply_translation([0.0, 0.0, 5.0])
    proposed = mesh.copy()
    v = np.asarray(proposed.vertices, float).copy()
    v += np.array([1.0, 0.0, 0.0])
    proposed.vertices = v
    accepted, gate = accept_anatomy_stage(
        mesh, proposed, margin, frame, stage="TEST",
        cervical_lock_mm=0.45, max_stage_move_mm=0.30,
    )
    assert gate["status"] == "ACCEPTED_BACKOFF"
    assert 0 < gate["accepted_fraction"] < 1.0
    assert np.linalg.norm(np.asarray(accepted.vertices)[0] - mesh.vertices[0]) <= 0.30 + 1e-8


def test_clean_cervical_source_has_single_planar_open_loop():
    mesh = trimesh.creation.cylinder(radius=2.0, height=4.0, sections=64)
    result, report = extract_clean_crown_surface(
        mesh, np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.0
    )
    assert report["status"] == "PASS_ENGINEERING", report
    assert result is not None
    rings = _boundary_loops(result)
    assert rings is not None and len(rings) == 1
    assert np.allclose(result.vertices[np.asarray(rings[0]), 2], 0.0, atol=1e-7)


def test_case_path_safety_and_localization_revision_acknowledgment(tmp_path, monkeypatch):
    impl = case_adapter._impl
    monkeypatch.setattr(impl, "work_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        impl.load_case("../private")

    f = tmp_path / "dummy.stl"
    f.write_bytes(b"geometry-1")
    case_id = "0123456789ab"
    folder = tmp_path / case_id
    folder.mkdir()
    record = {
        "id": case_id, "fdi": "36",
        "target_arch": str(f), "opponent_arch": str(f), "template": str(f),
        "localization": {"status": "READY"},
    }
    digest = impl._source_signature(record)
    record["input_signature"] = digest
    record["localization_token"] = "example-localization-digest"
    (folder / "case.json").write_text(json.dumps(record), encoding="utf8")
    acknowledged = impl.acknowledge_localization(case_id)
    assert acknowledged["acknowledged_localization_token"] == record["localization_token"]

    f.write_bytes(b"geometry-2")
    with pytest.raises(ValueError, match="changed"):
        impl.acknowledge_localization(case_id)
