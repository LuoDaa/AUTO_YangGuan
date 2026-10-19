"""Numerical regressions for the CrownFlow-derived W3 engineering cutter.

These tests intentionally make NO claim that a cutter is a clinical intaglio or
a completed crown. That requires support ownership + shared Margin assembly.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from autocrown.intaglio import build_engineering_cutter
from autocrown.intaglio_workflow import generate_cutter_artifacts


def _cylindrical_case():
    support = trimesh.creation.cylinder(radius=2.2, height=4.8, sections=48)
    support.apply_translation([0.0, 0.0, 2.4])
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    margin = np.c_[2.2 * np.cos(angles), 2.2 * np.sin(angles),
                   np.full(len(angles), 0.25)]
    return support, margin, SimpleNamespace(z=np.array([0.0, 0.0, 1.0]))


def test_port_rejects_unknown_gap_and_machine_policy():
    support, margin, frame = _cylindrical_case()
    with pytest.raises(ValueError, match="strictly positive"):
        build_engineering_cutter(support, margin, frame, cement_gap_mm=0)
    with pytest.raises(ValueError, match="machine_axes"):
        build_engineering_cutter(support, margin, frame, machine_axes=4)


def test_directional_blockout_and_gap_yield_watertight_engineering_cutter():
    support, margin, frame = _cylindrical_case()
    original = np.asarray(support.vertices, float).copy()
    blockout, cutter, report = build_engineering_cutter(
        support, margin, frame, cement_gap_mm=0.08,
        cement_gap_start_mm=0.0, machine_axes=3, resolution_mm=0.30,
    )
    assert np.allclose(support.vertices, original)
    assert blockout.is_watertight
    assert cutter.is_watertight
    assert cutter.is_winding_consistent
    assert cutter.volume > blockout.volume
    assert report["status"] == "CUTTER_READY_ENGINEERING_ONLY"
    assert report["final_crown_ready"] is False
    assert report["clinical_intaglio_proven"] is False
    assert report["gap"]["cement_gap_top_mm"] is None


def test_failed_cutter_never_deletes_outer_or_publishes_stale_inner(tmp_path, monkeypatch):
    import autocrown.intaglio_workflow as w3
    support, margin, frame = _cylindrical_case()
    outer = tmp_path / "05_outer_candidate.stl"
    outer.write_bytes(b"the previously accepted outer candidate")
    (tmp_path / "06_cutter.stl").write_bytes(b"old stale cutter")
    (tmp_path / "06_blockout.stl").write_bytes(b"old stale blockout")

    def raise_failure(*args, **kwargs):
        raise RuntimeError("synthetic numerical backend failure")

    monkeypatch.setattr(w3, "build_engineering_cutter", raise_failure)
    report = generate_cutter_artifacts(support, margin, frame, tmp_path)
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["clinical_final_ready"] is False
    assert outer.read_bytes() == b"the previously accepted outer candidate"
    assert not (tmp_path / "06_cutter.stl").exists()
    assert not (tmp_path / "06_blockout.stl").exists()
    assert (tmp_path / "06_intaglio_report.json").exists()
