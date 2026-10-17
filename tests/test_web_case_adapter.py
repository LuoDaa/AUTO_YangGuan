from autocrown.case_adapter import list_test_cases, template_path
from autocrown.web import app


def test_testdata_cases_are_discoverable():
    cases = list_test_cases()
    names = {c["name"] for c in cases}
    assert "基台测试2-牙位36" in names
    assert "基台测试4-牙位26" in names
    assert all(len(c["fdi"]) == 2 for c in cases)


def test_fdi_template_library_is_addressable():
    assert template_path("26").name == "FDI_26.stl"
    assert template_path("36").name == "FDI_36.stl"
    assert template_path("46").name == "FDI_46.stl"


def test_web_routes_exist():
    paths = {route.path for route in app.routes}
    assert "/api/test-cases" in paths
    assert "/api/cases/from-testdata" in paths
    assert "/api/cases/{case_id}/generate" in paths
    assert "/api/cases/{case_id}/asset/{asset_name}" in paths
