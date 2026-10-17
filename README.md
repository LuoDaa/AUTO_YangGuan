# AUTO_YangGuan

A standalone, conservative dental crown CAD MVP focused on **fast, stable 20–40 point crown generation** rather than high-complexity optimization.

This repository is intentionally independent from `LuoDaa/CrownFlow`.

## 1.0 strategy

The 1.0 pipeline uses a tooth library mesh as an **anatomical prior** and applies only bounded, geometry-driven corrections:

1. rigid placement in a patient frame (Margin centroid + insertion axis + mesial/buccal axes),
2. mild anisotropic scaling only,
3. cervical protection / Margin-owned root,
4. small proximal growth when the crown is too far from a neighbour,
5. deterministic projection-back when the crown invades a neighbour,
6. deterministic occlusal projection-back when it invades the antagonist,
7. local smoothing with a locked cervical band,
8. topology / movement / clearance QC,
9. a separate deterministic intaglio module (experimental in this MVP).

The default path deliberately does **not**:

- globally squash the library tooth to fill a box,
- use adjacent teeth or the antagonist as Boolean cutters,
- allow large unconstrained Laplacian deformations,
- use AI / diffusion / implicit generation,
- silently modify an accepted outer crown during final intaglio generation.

## Inputs

The CLI expects meshes already in a shared bite coordinate system:

- `template.stl` – library crown / crown-like anatomy,
- `support.stl` – prepared tooth / abutment,
- `mesial.stl` – optional mesial neighbour,
- `distal.stl` – optional distal neighbour,
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

The command writes stage meshes plus `report.json` so every deformation remains inspectable.

## Current scope

This is an engineering MVP, **not a clinically certified CAD system**. The default goal is to produce a conservative, editable crown candidate with no gross neighbour / antagonist penetration while preserving template anatomy as much as possible.
