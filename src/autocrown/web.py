from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .case_adapter import (
    create_case_from_testdata,
    generate_outer_for_case,
    list_test_cases,
    load_case,
    localize_case,
    save_uploaded_case,
    template_path,
    work_dir,
)

app = FastAPI(title="AUTO_YangGuan", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": "AUTO_YangGuan"}


@app.get("/api/test-cases")
def test_cases():
    return {"cases": list_test_cases()}


@app.get("/api/templates")
def templates():
    items = []
    for quadrant in "1234":
        for pos in "12345678":
            fdi = f"{quadrant}{pos}"
            try:
                p = template_path(fdi)
            except FileNotFoundError:
                continue
            items.append({"fdi": fdi, "name": p.name})
    return {"templates": items}


@app.post("/api/cases/from-testdata")
def case_from_testdata(payload: dict):
    try:
        record = create_case_from_testdata(str(payload.get("name", "")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _public_case(record)


@app.post("/api/cases/upload")
async def case_upload(
    fdi: str = Form(...),
    upper: UploadFile = File(...),
    lower: UploadFile = File(...),
):
    try:
        record = save_uploaded_case(
            fdi=fdi,
            upper_bytes=await upper.read(),
            lower_bytes=await lower.read(),
            upper_name=upper.filename or "upper.stl",
            lower_name=lower.filename or "lower.stl",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _public_case(record)


@app.get("/api/cases/{case_id}")
def case_detail(case_id: str):
    try:
        return _public_case(load_case(case_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="case not found") from exc


@app.post("/api/cases/{case_id}/localize")
def localize(case_id: str):
    try:
        record = localize_case(case_id)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"localization failed: {exc}") from exc
    return _public_case(record)


@app.post("/api/cases/{case_id}/generate")
def generate(case_id: str):
    try:
        record = generate_outer_for_case(case_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc
    return _public_case(record)


@app.get("/api/cases/{case_id}/asset/{asset_name}")
def case_asset(case_id: str, asset_name: str):
    try:
        record = load_case(case_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="case not found") from exc

    allowed = {
        "target": Path(record["target_arch"]),
        "opponent": Path(record["opponent_arch"]),
        "template": Path(record["template"]),
    }
    out = Path(record.get("output", work_dir() / case_id / "output"))
    allowed.update({
        "canonical_template": out / "00_template_canonical.stl",
        "support": out / "00_support_component.stl",
        "target_component": out / "00_target_jaw_component.stl",
        "margin": out / "00_margin.json",
        "localization": out / "00_localization.json",
        "mesial": out / "00_neighbor_mesial.stl",
        "distal": out / "00_neighbor_distal.stl",
        "placed": out / "01_placed.stl",
        "occlusal": out / "04_occlusal.stl",
        "outer": out / "05_outer_candidate.stl",
        "report": out / "report.json",
    })
    path = allowed.get(asset_name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"asset not found: {asset_name}")
    media = "application/json" if path.suffix.lower() == ".json" else "application/sla"
    return FileResponse(path, media_type=media, filename=path.name)


def _public_case(record: dict) -> dict:
    case_id = record["id"]
    output = Path(record.get("output", work_dir() / case_id / "output"))
    localized = bool(record.get("localization"))
    generated = bool(record.get("report")) and (output / "05_outer_candidate.stl").exists()
    localization = record.get("localization")
    return {
        "id": case_id,
        "name": record.get("name"),
        "fdi": record.get("fdi"),
        "source": record.get("source"),
        "status": record.get("status"),
        "localized": localized,
        "generated": generated,
        "assets": {
            "target": f"/api/cases/{case_id}/asset/target",
            "opponent": f"/api/cases/{case_id}/asset/opponent",
            "template": f"/api/cases/{case_id}/asset/template",
            "canonical_template": f"/api/cases/{case_id}/asset/canonical_template" if (output / "00_template_canonical.stl").exists() else None,
            "support": f"/api/cases/{case_id}/asset/support" if localized else None,
            "target_component": f"/api/cases/{case_id}/asset/target_component" if localized else None,
            "margin": f"/api/cases/{case_id}/asset/margin" if localized else None,
            "localization": f"/api/cases/{case_id}/asset/localization" if localized else None,
            "mesial": f"/api/cases/{case_id}/asset/mesial" if (output / "00_neighbor_mesial.stl").exists() else None,
            "distal": f"/api/cases/{case_id}/asset/distal" if (output / "00_neighbor_distal.stl").exists() else None,
            "outer": f"/api/cases/{case_id}/asset/outer" if generated else None,
            "report": f"/api/cases/{case_id}/asset/report" if generated else None,
        },
        "localization": localization,
        "report": record.get("report"),
    }


def run():
    import uvicorn
    uvicorn.run("autocrown.web:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run()
