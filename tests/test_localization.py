import numpy as np
import trimesh

from autocrown.case_adapter import mesial_sign_for_fdi
from autocrown.localization import build_automatic_patient_frame, solve_insertion_axis
from autocrown.support_locator import localize_support


def _translated(mesh, xyz):
    out = mesh.copy()
    out.apply_translation(np.asarray(xyz, float))
    return out


def test_localize_abutment_selects_disconnected_support_and_orients_to_opponent():
    # Deliberately give the support relatively dense tessellation: main-jaw
    # selection must use physical arch scale/area rather than raw face count.
    jaw = trimesh.creation.box(extents=(24.0, 18.0, 2.0))
    for _ in range(2):
        jaw = jaw.subdivide()
    jaw = _translated(jaw, (0.0, 0.0, -2.0))
    support = _translated(
        trimesh.creation.cylinder(radius=2.4, height=6.0, sections=96),
        (6.0, 0.0, 1.0),
    )
    distractor = _translated(
        trimesh.creation.icosphere(subdivisions=1, radius=1.0),
        (-8.0, 5.0, 0.0),
    )
    target = trimesh.util.concatenate([jaw, support, distractor])
    opponent = _translated(
        trimesh.creation.box(extents=(24.0, 18.0, 2.0)),
        (0.0, 0.0, 10.0),
    )

    result = localize_support(target, opponent)

    assert len(result.support.faces) == len(support.faces)
    assert float(result.rough_axis @ np.array([0.0, 0.0, 1.0])) > 0.85
    assert result.report["selected_component_index"] != result.report["main_jaw_component_index"]
    assert result.report["confidence"] in {"high", "review"}
    assert result.report["method"] == "ARCH_AREA_SUPPORT_RANK_WALL_NORMAL_V3"


def test_axis_solver_recovers_cylindrical_insertion_direction_from_margin():
    support = _translated(
        trimesh.creation.cylinder(radius=2.5, height=6.0, sections=128),
        (4.0, -2.0, 2.0),
    )
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    margin = np.column_stack((
        4.0 + 2.5 * np.cos(theta),
        -2.0 + 2.5 * np.sin(theta),
        np.full_like(theta, -0.7),
    ))
    rough = np.array([0.10, -0.06, 0.993])
    rough /= np.linalg.norm(rough)

    axis, report = solve_insertion_axis(support, margin, rough)

    assert float(axis @ np.array([0.0, 0.0, 1.0])) > 0.90
    assert report["status"] in {"OK", "REVIEW_REQUIRED"}
    assert 0.0 <= report["confidence"] <= 1.0


def test_patient_frame_keeps_buccal_outward_and_right_handed():
    jaw = _translated(
        trimesh.creation.box(extents=(30.0, 20.0, 2.0)),
        (0.0, 0.0, -2.0),
    )
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    margin = np.column_stack((
        8.0 + 2.0 * np.cos(theta),
        3.0 + 2.0 * np.sin(theta),
        np.zeros_like(theta),
    ))
    frame, report = build_automatic_patient_frame(
        jaw, margin, np.array([0.0, 0.0, 1.0])
    )
    radial = np.asarray(margin).mean(axis=0) - np.median(np.asarray(jaw.vertices), axis=0)
    radial -= frame.z * float(radial @ frame.z)

    assert float(frame.y @ radial) > 0.0
    assert np.isclose(np.linalg.det(frame.matrix), 1.0, atol=1e-6)
    assert np.isclose(report["handedness"], 1.0, atol=1e-6)


def test_fdi_mesial_sign_is_separate_from_patient_frame_orientation():
    assert mesial_sign_for_fdi("16") == 1.0
    assert mesial_sign_for_fdi("36") == 1.0
    assert mesial_sign_for_fdi("26") == -1.0
    assert mesial_sign_for_fdi("46") == -1.0
