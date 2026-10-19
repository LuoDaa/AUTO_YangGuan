import numpy as np
import pytest

import autocrown.case_adapter as adapter
from autocrown.case_adapter import template_path
from autocrown.meshops import load_mesh
from autocrown.template_normalization import canonicalize_full_tooth, library_occlusal_reference


def test_public_adapter_uses_library_referenced_template_normalizer():
    assert adapter._impl._canonicalize_full_tooth is canonicalize_full_tooth


@pytest.mark.parametrize("fdi", ["16", "26", "36", "46"])
def test_molar_template_canonical_axis_is_occlusally_referenced(fdi):
    raw = load_mesh(template_path(fdi))
    crown, report = canonicalize_full_tooth(raw, fdi)

    assert report["mode"] == "FULL_TOOTH_TO_CANONICAL_CROWN_V4_LIBRARY_OCCLUSAL"
    assert len(crown.faces) >= 100
    assert np.isfinite(np.asarray(crown.vertices)).all()

    reference, _ = library_occlusal_reference(fdi)
    z_world = np.asarray(report["z_axis_world"], float)
    assert float(z_world @ reference) >= 0.50

    axis = report["axis"]
    assert axis["source"] in {
        "LIBRARY_OCCLUSAL_GUIDED_PCA",
        "LIBRARY_OCCLUSAL_NORMAL_DIRECT_FALLBACK",
    }
    assert 0.0 <= axis["selected_alignment_abs"] <= 1.0

    v = np.asarray(crown.vertices, float)
    assert abs(float(np.quantile(v[:, 2], 0.02))) < 1e-6
    assert report["cervical_to_body_offset_ratio"] < 0.55
    assert 0.18 <= report["crown_height_width_ratio"] <= 3.20
    assert report["clinical_cervical_valid"], {
        "fdi": fdi,
        "cervical_cut": report["cervical_cut"],
    }


def test_fdi36_cervical_band_owns_template_xy_origin():
    raw = load_mesh(template_path("36"))
    crown, report = canonicalize_full_tooth(raw, "36")
    v = np.asarray(crown.vertices, float)
    low = v[:, 2] <= np.quantile(v[:, 2], 0.12)
    assert int(low.sum()) >= 8
    cervical_xy = np.median(v[low, :2], axis=0)
    assert np.linalg.norm(cervical_xy) < 1e-6
    assert report["cervical_to_body_offset_ratio"] < 0.45
