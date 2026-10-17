import numpy as np
import trimesh

import autocrown.localization_runtime as runtime
from autocrown.crownflow_abutment_margin import extract_abutment_margin
from autocrown.crownflow_insertion import AxisStatus, solve_insertion_axis
from autocrown.crownflow_patient_frame import build_patient_frame


def _shouldered_abutment(sections=128):
    """Closed synthetic abutment with a unique circumferential shoulder at z=0."""
    theta = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    profiles = [
        (3.00, -0.35),  # smaller base/truncation ring; not the clinical shoulder
        (3.35, 0.00),   # unique outer shoulder edge
        (2.75, 0.00),   # horizontal shoulder table
        (2.35, 3.80),   # tapered preparation wall
        (2.20, 4.25),   # top edge
    ]
    vertices = []
    for radius, z in profiles:
        vertices.extend(np.c_[radius * np.cos(theta), radius * np.sin(theta), np.full(sections, z)])
    vertices = np.asarray(vertices, float)
    faces = []
    rings = len(profiles)
    for j in range(rings - 1):
        a0 = j * sections
        b0 = (j + 1) * sections
        for i in range(sections):
            k = (i + 1) % sections
            faces.append([a0 + i, a0 + k, b0 + k])
            faces.append([a0 + i, b0 + k, b0 + i])

    bottom_center = len(vertices)
    top_center = bottom_center + 1
    vertices = np.vstack([vertices, [0.0, 0.0, profiles[0][1]], [0.0, 0.0, profiles[-1][1]]])
    for i in range(sections):
        k = (i + 1) % sections
        faces.append([bottom_center, k, i])
        top0 = (rings - 1) * sections
        faces.append([top_center, top0 + i, top0 + k])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, int), process=True)
    mesh.fix_normals()
    return mesh


def test_crownflow_shoulder_port_finds_cervical_ring_not_top():
    support = _shouldered_abutment()
    margin, report = extract_abutment_margin(support, np.array([0.0, 0.0, 1.0]))
    margin = np.asarray(margin, float)

    assert report["algorithm_version"].startswith("crownflow/abutment-shoulder/")
    assert len(margin) == 128
    assert np.isfinite(margin).all()
    radius = np.linalg.norm(margin[:, :2], axis=1)
    # The mature DP should prefer the unique prominent large-radius shoulder
    # around z=0, not the smaller base ring or occlusal/top edge.
    assert abs(float(np.median(margin[:, 2]))) < 0.20
    assert float(np.median(radius)) > 3.2


def test_crownflow_axis_port_uses_margin_and_sidewall_evidence():
    support = _shouldered_abutment()
    margin, _ = extract_abutment_margin(support, np.array([0.0, 0.0, 1.0]))
    rough = np.array([0.09, -0.05, 0.994])
    rough /= np.linalg.norm(rough)
    solved = solve_insertion_axis(support, np.asarray(margin, float), rough)

    assert solved.status in {AxisStatus.OK, AxisStatus.REVIEW_REQUIRED}
    assert float(solved.axis @ np.array([0.0, 0.0, 1.0])) > 0.90
    assert solved.report["initial_source"] in {
        "MARGIN_PLANE_PLUS_SIDEWALL",
        "MARGIN_PLANE_DOMINANT_WALL_DISAGREES",
    }
    assert solved.report["algorithm_version"] == "crownflow/insertion-axis/ported-v1"


def test_crownflow_patient_frame_is_margin_owned_and_right_handed():
    t = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    margin = np.c_[8.0 + 3.2 * np.cos(t), 4.0 + 2.2 * np.sin(t), np.zeros_like(t)]
    jaw = trimesh.creation.box(extents=(30.0, 22.0, 2.0))
    jaw.apply_translation([0.0, 0.0, -2.0])
    frame, report = build_patient_frame(margin, np.array([0.0, 0.0, 1.0]), working_jaw=jaw)

    assert np.linalg.norm(frame.origin - margin.mean(axis=0)) < 1e-9
    assert np.isclose(np.linalg.det(frame.matrix), 1.0, atol=1e-6)
    assert report["y_semantics"] == "BUCCAL_ARCH_OUTWARD"
    assert report["z_semantics"] == "INSERTION_TOWARD_ANTAGONIST"


def test_runtime_has_retired_early_finish_line_and_axis_proxies():
    # This guards the production import graph, not just the existence of port files.
    assert runtime.extract_abutment_margin is extract_abutment_margin
    assert runtime.solve_insertion_axis is solve_insertion_axis
