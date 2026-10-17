# AUTO_YangGuan

A standalone, conservative dental crown CAD MVP focused on **fast, stable 20–40 point crown generation** rather than high-complexity optimization.

This repository is intentionally independent from `LuoDaa/CrownFlow`. It does not import from or modify CrownFlow.

## 1.0 strategy

The 1.0 outer-crown pipeline uses a tooth library mesh as an **anatomical prior** and applies only bounded, geometry-driven corrections:

1. rigid placement in a confirmed patient frame (Margin centroid + insertion axis + mesial/buccal axes),
2. mild anisotropic scaling only,
3. cervical protection / Margin-owned root,
4. small proximal growth when the crown is too far from a neighbour,
5. deterministic projection-back when the crown invades a neighbour,
6. deterministic occlusal projection-back when it invades the antagonist,
7. local smoothing with a locked cervical band,
8. topology / movement QC,
9. stage-by-stage STL and JSON diagnostics.

The default path deliberately does **not**:

- globally squash the library tooth to fill a box,
- use adjacent teeth or the antagonist as Boolean cutters,
- pull a low occlusal cusp upward just to force contact,
- allow large unconstrained Laplacian deformations,
- use AI / diffusion / implicit generation,
- silently modify an accepted outer crown during final intaglio generation.

## Template contract

The current MVP expects a **crown-only** template in a canonical local coordinate system:

- `+X` = mesial,
- `+Y` = buccal,
- `+Z` = occlusal,
- the cervical part is near the minimum local Z,
- no root should be present.

The code intentionally clamps initial X/Y scale to a small range instead of forcing the template to fill the entire patient space.

## Inputs

The CLI expects meshes already in a shared bite coordinate system:

- `template.stl` – canonical crown-only library anatomy,
- `support.stl` – prepared tooth / abutment (currently used only by the experimental intaglio preview),
- `mesial.stl` – optional mesial neighbour or local neighbour patch,
- `distal.stl` – optional distal neighbour or local neighbour patch,
- `antagonist.stl` – optional opposing jaw / local opponent patch,
- `margin.json` – confirmed 3D Margin points,
- `frame.json` – confirmed patient frame vectors.

Example `frame.json`:

```json
{
  "mesial": [1, 0, 0],
  "buccal": [0, 1, 0],
  "insertion": [0, 0, 1]
}
```

`margin.json` can be either a raw `[[x,y,z], ...]` list or an object containing `points` / `margin`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

autocrown outer \
  --template template.stl \
  --margin margin.json \
  --frame frame.json \
  --mesial mesial.stl \
  --distal distal.stl \
  --antagonist antagonist.stl \
  --out output_case
```

The command writes inspectable stage artifacts:

```text
01_placed.stl
02_mesial.stl
03_distal.stl
04_occlusal.stl
05_outer_candidate.stl
report.json
```

## Intaglio status

`autocrown intaglio-preview` currently produces only a **support normal-offset diagnostic preview**. It is intentionally labelled `PREVIEW_ONLY` and must **not** be treated as a manufacturing cutter.

A real final intaglio is the next subsystem and must add:

1. insertion-axis directional blockout,
2. cement-gap field,
3. face provenance / clinical-surface ownership,
4. support-coverage validation,
5. Margin-owned supramarginal clipping,
6. deterministic cervical collar / final assembly.

This separation is deliberate: the outer-crown MVP can be validated on real cases without pretending that seating-safe intaglio geometry is already solved.

## Current scope

This is an engineering MVP, **not a clinically certified CAD system**. The current goal is to produce a conservative, editable outer-crown candidate with no gross neighbour / antagonist violation while preserving template anatomy as much as possible.

The next validation milestone is to run the pipeline on real posterior cases and tune only a small number of explicit limits (initial scale, local proximal growth, local collision relief) before adding a final intaglio implementation.
