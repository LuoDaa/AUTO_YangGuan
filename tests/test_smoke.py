import numpy as np
import trimesh

from autocrown.config import PipelineConfig
from autocrown.frame import build_confirmed_frame
from autocrown.pipeline import build_outer_crown


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
    assert report["pipeline"] == "AUTO_YANGGUAN_HYBRID_MVP_1"
    assert report["status"] in {"OUTER_CANDIDATE_READY", "OUTER_REVIEW_REQUIRED"}
    names = [stage["name"] for stage in report["stages"]]
    assert names == ["PLACEMENT", "MESIAL", "DISTAL", "OCCLUSAL"]
