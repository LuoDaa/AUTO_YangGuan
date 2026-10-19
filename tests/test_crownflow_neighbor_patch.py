"""CrownFlow-inspired coronal core + connected halo regression."""
import numpy as np
import trimesh

from autocrown.frame import PatientFrame
from autocrown.crownflow_neighbor_patch import extract_neighbor_patch_v5


def _context():
    frame = PatientFrame(
        origin=np.zeros(3),
        x=np.array([1.0, 0.0, 0.0]),
        y=np.array([0.0, 1.0, 0.0]),
        z=np.array([0.0, 0.0, 1.0]),
    )
    angle = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    margin = np.c_[2.0 * np.cos(angle), 1.7 * np.sin(angle),
                   np.zeros(len(angle))]
    return frame, margin


def _neighbor(x):
    mesh = trimesh.creation.box(extents=[2.0, 4.0, 5.0])
    mesh.apply_translation([x, 0.0, 2.5])
    return mesh


def test_connected_patch_uses_true_anatomical_side():
    frame, margin = _context()
    jaw = trimesh.util.concatenate([_neighbor(-5.8), _neighbor(5.8)])
    mesial, report_m = extract_neighbor_patch_v5(
        jaw, frame, margin, "mesial", mesial_sign=1.0
    )
    distal, report_d = extract_neighbor_patch_v5(
        jaw, frame, margin, "distal", mesial_sign=1.0
    )
    assert report_m["status"] == "READY", report_m
    assert report_d["status"] == "READY", report_d
    assert float(np.mean(mesial.vertices[:, 0])) > 4.0
    assert float(np.mean(distal.vertices[:, 0])) < -4.0
    assert report_m["core_faces"] >= 8
    assert report_m["reference_faces"] >= report_m["core_faces"]


def test_fdi_mesial_sign_flips_side_not_buccal_frame():
    frame, margin = _context()
    jaw = trimesh.util.concatenate([_neighbor(-5.8), _neighbor(5.8)])
    mesial, report = extract_neighbor_patch_v5(
        jaw, frame, margin, "mesial", mesial_sign=-1.0
    )
    assert report["status"] == "READY", report
    assert float(np.mean(mesial.vertices[:, 0])) < -4.0
    assert np.array_equal(frame.y, [0.0, 1.0, 0.0])


def test_missing_side_is_not_silently_claimed_confirmed():
    frame, margin = _context()
    only_mesial = _neighbor(5.8)
    region, report = extract_neighbor_patch_v5(
        only_mesial, frame, margin, "distal", mesial_sign=1.0
    )
    assert region is None
    assert report["status"] == "REVIEW_REQUIRED"
