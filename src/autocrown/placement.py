from __future__ import annotations

import numpy as np
import trimesh

from .config import PlacementConfig
from .frame import PatientFrame, margin_local_extents


def _finite_span(values: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return 0.0
    return float(np.quantile(values, hi) - np.quantile(values, lo))


def place_template(template: trimesh.Trimesh, margin: np.ndarray, frame: PatientFrame,
                   config: PlacementConfig | None = None):
    """Place a canonically oriented crown template into the confirmed patient frame.

    Contract for the MVP template library:
      * +X = mesial, +Y = buccal, +Z = occlusal,
      * crown-only (no root),
      * cervical portion lies near the minimum Z of the mesh.

    Scaling is intentionally mild. The Margin only suggests cervical X/Y size;
    it never forces the whole tooth to fill the patient space.
    """
    cfg = config or PlacementConfig()
    out = template.copy()
    v = np.asarray(out.vertices, float)
    if not np.isfinite(v).all() or len(v) < 8:
        raise ValueError("template mesh is invalid")

    z_cerv = float(np.quantile(v[:, 2], cfg.cervical_quantile))
    low = v[:, 2] <= z_cerv + float(cfg.cervical_band_mm)
    if int(low.sum()) < 8:
        low = v[:, 2] <= np.quantile(v[:, 2], 0.20)

    tx = _finite_span(v[low, 0])
    ty = _finite_span(v[low, 1])
    ext = margin_local_extents(margin, frame)
    mx = float(ext["x_max"] - ext["x_min"])
    my = float(ext["y_max"] - ext["y_min"])
    target_x = max(mx + 2.0 * cfg.emergence_pad_mm, 1e-6)
    target_y = max(my + 2.0 * cfg.emergence_pad_mm, 1e-6)

    sx_raw = target_x / tx if tx > 1e-6 else 1.0
    sy_raw = target_y / ty if ty > 1e-6 else 1.0
    sx = float(np.clip(sx_raw, cfg.scale_min, cfg.scale_max))
    sy = float(np.clip(sy_raw, cfg.scale_min, cfg.scale_max))
    sz = 1.0

    center_xy = np.median(v[low, :2], axis=0)
    local = v.copy()
    local[:, 0] = (local[:, 0] - center_xy[0]) * sx
    local[:, 1] = (local[:, 1] - center_xy[1]) * sy
    local[:, 2] = (local[:, 2] - z_cerv) * sz + float(ext["z_median"])
    out.vertices = frame.to_world(local)

    report = {
        "method": "CANONICAL_TEMPLATE_MILD_SCALE_V1",
        "template_contract": "CROWN_ONLY_X_MESIAL_Y_BUCCAL_Z_OCCLUSAL",
        "raw_scale": {"x": sx_raw, "y": sy_raw, "z": 1.0},
        "applied_scale": {"x": sx, "y": sy, "z": 1.0},
        "scale_clamped": bool(abs(sx - sx_raw) > 1e-8 or abs(sy - sy_raw) > 1e-8),
        "margin_cervical_span_mm": {"x": mx, "y": my},
        "template_cervical_span_mm": {"x": tx, "y": ty},
    }
    return out, report
