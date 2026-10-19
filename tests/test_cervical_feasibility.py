"""Pre-stitch W3 regression: valid clearance, collapsed margin band and source immutability."""
import numpy as np
import trimesh

from autocrown.frame import PatientFrame
from autocrown.cervical_feasibility import audit_pre_stitch_feasibility
from autocrown.support_owned_intaglio import extract_support_owned_candidate


def _context(outer_radius=3.0):
    frame = PatientFrame(
        origin=np.zeros(3),
        x=np.array([1.0, 0.0, 0.0]),
        y=np.array([0.0, 1.0, 0.0]),
        z=np.array([0.0, 0.0, 1.0]),
    )
    theta = np.linspace(0, 2*np.pi, 96, endpoint=False)
    margin = np.c_[2.2*np.cos(theta), 2.2*np.sin(theta),
                   np.full(len(theta), 0.2)]
    support = trimesh.creation.cylinder(radius=2.2, height=4.8, sections=48)
    support.apply_translation([0, 0, 2.4])
    cutter = trimesh.creation.cylinder(radius=2.28, height=5.9, sections=64)
    cutter.apply_translation([0, 0, 1.95])
    inner, ownership = extract_support_owned_candidate(
        cutter, support, margin, frame, transition_mm=0.35
    )
    assert inner is not None, ownership
    outer = trimesh.creation.cylinder(radius=outer_radius, height=6.0, sections=64)
    outer.apply_translation([0, 0, 2.5])
    return outer, inner, margin, frame


def test_pre_stitch_gate_passes_safe_thick_source_pair_without_mutation():
    outer, inner, margin, frame = _context(outer_radius=3.0)
    frozen_outer = np.asarray(outer.vertices, float).copy()
    frozen_inner = np.asarray(inner.vertices, float).copy()
    skin, report = audit_pre_stitch_feasibility(
        outer, inner, margin, frame, transition_mm=0.35
    )
    assert report["status"] == "PASS_ENGINEERING", report
    assert skin is not None
    assert skin.is_watertight is False
    assert report["outer_boundary"]["loop_count"] == 1
    assert report["inner_boundary"]["loop_count"] == 1
    assert report["outer_ring_to_inner_surface_mm"]["p05"] > 0.1
    assert report["clinical_final_ready"] is False
    assert np.array_equal(outer.vertices, frozen_outer)
    assert np.array_equal(inner.vertices, frozen_inner)


def test_pre_stitch_gate_catches_cervical_source_collapse():
    outer, inner, margin, frame = _context(outer_radius=2.29)
    skin, report = audit_pre_stitch_feasibility(
        outer, inner, margin, frame, transition_mm=0.35
    )
    assert report["status"] == "REVIEW_REQUIRED", report
    assert report["reason"] == "CERVICAL_SOURCE_COLLAPSE"
    assert report["clinical_final_ready"] is False
    assert skin is not None


def test_pre_stitch_refuses_invalid_transition_without_editing_outer():
    outer, inner, margin, frame = _context()
    before = outer.vertices.copy()
    skin, report = audit_pre_stitch_feasibility(
        outer, inner, margin, frame, transition_mm=0
    )
    assert skin is None
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["reason"] == "INVALID_TRANSITION_MM"
    assert np.array_equal(outer.vertices, before)
