import numpy as np
import trimesh

from autocrown.config import PipelineConfig
from autocrown.frame import build_confirmed_frame
from autocrown.pipeline import _fit_height_to_neighbors, build_outer_crown


def _margin(n=64, rx=2.0, ry=1.6):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[rx * np.cos(t), ry * np.sin(t), np.zeros_like(t)]


def test_confirmed_frame_is_right_handed():
    margin = _margin()
    frame = build_confirmed_frame(
        margin,
        mesial=[1, 0, 0],
        buccal=[0, 1, 0],
        insertion=[0, 0, 1],
    )
    m = frame.matrix
    assert np.allclose(m.T @ m, np.eye(3), atol=1e-7)
    assert np.linalg.det(m) > 0.99


def test_height_fit_raises_occlusal_body_and_keeps_cervical_band_fixed():
    margin = _margin()
    frame = build_confirmed_frame(
        margin,
        mesial=[1, 0, 0],
        buccal=[0, 1, 0],
        insertion=[0, 0, 1],
    )

    crown = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    crown.vertices *= np.array([2.0, 1.8, 2.0])
    crown.vertices[:, 2] += 2.0

    mesial = trimesh.creation.box(extents=[2.0, 4.0, 6.0])
    mesial.apply_translation([4.0, 0.0, 3.0])
    distal = trimesh.creation.box(extents=[2.0, 4.0, 6.0])
    distal.apply_translation([-4.0, 0.0, 3.0])

    before = np.asarray(crown.vertices, float).copy()
    fitted, report = _fit_height_to_neighbors(
        crown,
        frame,
        margin,
        mesial=mesial,
        distal=distal,
        cervical_lock_mm=0.45,
    )
    after = np.asarray(fitted.vertices, float)

    assert report["status"] == "APPLIED"
    assert report["applied_shift_mm"] > 1.0
    assert np.quantile(after[:, 2], 0.90) > np.quantile(before[:, 2], 0.90) + 1.0

    locked = before[:, 2] <= 0.45 + 1e-9
    assert int(locked.sum()) >= 1
    assert np.allclose(after[locked], before[locked], atol=1e-9)


def test_outer_pipeline_smoke():
    margin = _margin()
    frame = build_confirmed_frame(
        margin,
        mesial=[1, 0, 0],
        buccal=[0, 1, 0],
        insertion=[0, 0, 1],
    )

    template = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    template.vertices *= np.array([2.6, 2.1, 3.0])
    template.vertices[:, 2] += 3.0

    mesial = trimesh.creation.box(extents=[2.0, 5.0, 7.0])
    mesial.apply_translation([4.0, 0.0, 3.0])
    distal = trimesh.creation.box(extents=[2.0, 5.0, 7.0])
    distal.apply_translation([-4.0, 0.0, 3.0])
    antagonist = trimesh.creation.box(extents=[12.0, 12.0, 1.0])
    antagonist.apply_translation([0.0, 0.0, 6.3])

    result, report = build_outer_crown(
        template,
        margin,
        frame,
        mesial=mesial,
        distal=distal,
        antagonist=antagonist,
        config=PipelineConfig(),
    )

    assert len(result.vertices) == len(template.vertices)
    assert len(result.faces) == len(template.faces)
    assert np.isfinite(np.asarray(result.vertices)).all()
    assert report["pipeline"] == "AUTO_YANGGUAN_HYBRID_MVP_2"
    assert report["frame_contract"] == "X_RIGHT_HANDED_ARCH_TANGENT_Y_BUCCAL_Z_INSERTION"
    assert report["mesial_sign_on_frame_x"] == 1.0
    assert report["status"] in {"OUTER_CANDIDATE_READY", "OUTER_REVIEW_REQUIRED"}
    names = [stage["name"] for stage in report["stages"]]
    assert names == ["PLACEMENT", "HEIGHT", "MESIAL", "DISTAL", "OCCLUSAL"]
