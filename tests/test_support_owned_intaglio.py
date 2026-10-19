"""Support-owned supramarginal intaglio regression tests.

Clinical inner = fitting surface owned by the preparation, NOT the component
closest to the margin. Outputs remain engineering candidates pending a complete
shared-margin cervical assembly and insertion/thickness audit.
"""
import numpy as np
import trimesh

from autocrown.frame import PatientFrame
from autocrown.support_owned_intaglio import (
    _boundary_loops, _triangle_clip_above, extract_support_owned_candidate,
)


def _case():
    frame = PatientFrame(
        origin=np.zeros(3),
        x=np.array([1.0, 0.0, 0.0]),
        y=np.array([0.0, 1.0, 0.0]),
        z=np.array([0.0, 0.0, 1.0]),
    )
    support = trimesh.creation.cylinder(radius=2.2, height=4.8, sections=48)
    support.apply_translation([0, 0, 2.4])
    angles = np.linspace(0, 2*np.pi, 96, endpoint=False)
    margin = np.c_[2.2*np.cos(angles), 2.2*np.sin(angles),
                   np.full(len(angles), 0.2)]
    # A watertight cutter with a wider radius, bottom closure below Margin
    # and a slightly higher top simulates a cement-gap shell.
    cutter = trimesh.creation.cylinder(radius=2.28, height=5.9, sections=64)
    cutter.apply_translation([0, 0, 1.95])  # range -1.0 .. +4.9
    return support, cutter, margin, frame


def test_triangle_clip_creates_one_open_ring_without_mutating_source():
    support, cutter, margin, frame = _case()
    original = np.asarray(cutter.vertices).copy()
    mesh, diag = _triangle_clip_above(
        cutter, np.asarray(cutter.vertices)[:, 2] - 0.2, level=0.35
    )
    loops, boundary = _boundary_loops(mesh)
    assert boundary["status"] == "OK", boundary
    assert len(loops) == 1
    assert not mesh.is_watertight
    assert np.allclose(original, cutter.vertices)
    assert diag["inserted_vertices"] > 0


def test_support_owned_candidate_ignores_near_margin_fragment():
    support, cutter, margin, frame = _case()
    tiny = trimesh.creation.icosphere(subdivisions=2, radius=0.32)
    tiny.apply_translation([4.8, 0.0, 0.48])
    composite = trimesh.util.concatenate([cutter, tiny])
    selected, report = extract_support_owned_candidate(
        composite, support, margin, frame, transition_mm=0.35,
    )
    assert report["component_count"] == 2, report
    assert report["status"] == "INNER_CANDIDATE_ENGINEERING_ONLY", report
    assert selected is not None
    assert report["selected_area_fraction"] > 0.80
    assert report["selected_hard_valid"] is True
    assert len(selected.faces) > 100
    assert report["clinical_final_ready"] is False


def test_fragment_only_never_claimed_as_clinical_intaglio():
    support, _cutter, margin, frame = _case()
    tiny = trimesh.creation.icosphere(subdivisions=2, radius=0.32)
    tiny.apply_translation([4.8, 0.0, 0.48])
    selected, report = extract_support_owned_candidate(
        tiny, support, margin, frame, transition_mm=0.35,
    )
    assert selected is None
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["reason"] == "NO_SUPPORT_OWNED_SINGLE_RING_COMPONENT"


def test_open_unclipped_fragment_does_not_become_final_by_name():
    support, cutter, margin, frame = _case()
    selected, report = extract_support_owned_candidate(
        cutter, support, margin, frame, transition_mm=0.35,
    )
    assert selected is not None, report
    assert selected.is_watertight is False
    assert report["clinical_final_ready"] is False
    assert report["clinical_intaglio_proven"] is False
