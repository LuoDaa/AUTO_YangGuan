from pathlib import Path

import numpy as np

from autocrown.case_adapter import create_case_from_testdata, localize_case, work_dir


def test_real_fdi36_case_localizes_support_margin_axis_and_frame():
    """Regression on the repository's real FDI36 abutment case.

    This deliberately stops before crown generation. A wrong support/axis makes
    every downstream crown meaningless, so localization is a first-class gate.
    """
    record = create_case_from_testdata("基台测试2-牙位36")
    try:
        result = localize_case(record["id"])
        loc = result["localization"]

        assert loc["abutment"]["selected_component_index"] >= 1
        assert len(loc["margin_points"]) >= 64
        axis = np.asarray(loc["axis"]["axis"], float)
        assert np.isfinite(axis).all()
        assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-5)

        frame = loc["frame"]
        matrix = np.column_stack((frame["x"], frame["y"], frame["z"]))
        assert np.all(np.isfinite(matrix))
        assert np.isclose(abs(np.linalg.det(matrix)), 1.0, atol=2e-3)
        assert frame["y_semantics"] == "BUCCAL_ARCH_OUTWARD"
        assert frame["z_semantics"] == "INSERTION_TOWARD_ANTAGONIST"

        assets = result["localization_assets"]
        assert Path(assets["support"]).is_file()
        assert Path(assets["margin"]).is_file()
        assert Path(assets["report"]).is_file()
    finally:
        import shutil
        shutil.rmtree(work_dir() / record["id"], ignore_errors=True)
