from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


class FrameError(ValueError):
    pass


@dataclass(frozen=True)
class PatientFrame:
    origin: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        return np.stack([self.x, self.y, self.z], axis=1)

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, float) - self.origin) @ self.matrix

    def to_world(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, float) @ self.matrix.T + self.origin


def _unit(v, name: str) -> np.ndarray:
    v = np.asarray(v, float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise FrameError(f"{name} must be a finite 3-vector")
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise FrameError(f"{name} is degenerate")
    return v / n


def build_confirmed_frame(margin: np.ndarray, mesial, buccal, insertion) -> PatientFrame:
    """Build a right-handed patient frame from confirmed directions.

    The confirmed insertion axis owns Z. Mesial is projected onto the plane
    normal to Z, then Y is derived by the right-hand rule and sign-aligned to the
    confirmed buccal direction. This keeps the frame deterministic while still
    respecting user-confirmed anatomy directions.
    """
    margin = np.asarray(margin, float)
    if margin.ndim != 2 or margin.shape[1] != 3 or len(margin) < 8:
        raise FrameError("margin must contain at least 8 three-dimensional points")
    if not np.isfinite(margin).all():
        raise FrameError("margin contains NaN/Inf")
    z = _unit(insertion, "insertion")
    x0 = _unit(mesial, "mesial")
    x = x0 - float(x0 @ z) * z
    x = _unit(x, "mesial projected to insertion plane")
    y = _unit(np.cross(z, x), "buccal candidate")
    b = _unit(buccal, "buccal")
    if float(y @ b) < 0.0:
        x = -x
        y = -y
    x = _unit(np.cross(y, z), "mesial orthogonalized")
    origin = np.mean(margin, axis=0)
    return PatientFrame(origin=origin, x=x, y=y, z=z)


def axial_height_from_margin(points: np.ndarray, margin: np.ndarray, frame: PatientFrame) -> np.ndarray:
    """Approximate non-planar supramarginal height using nearest Margin XY sample.

    The calculation is performed in the confirmed patient frame. For each point,
    the nearest Margin sample in the X/Y plane owns the local Margin Z value.
    Positive values are coronal / supramarginal along the insertion axis.
    """
    p = frame.to_local(points)
    m = frame.to_local(margin)
    tree = cKDTree(m[:, :2])
    _, idx = tree.query(p[:, :2], k=1)
    return p[:, 2] - m[np.asarray(idx, int), 2]


def margin_local_extents(margin: np.ndarray, frame: PatientFrame) -> dict:
    local = frame.to_local(margin)
    return {
        "x_min": float(np.quantile(local[:, 0], 0.02)),
        "x_max": float(np.quantile(local[:, 0], 0.98)),
        "y_min": float(np.quantile(local[:, 1], 0.02)),
        "y_max": float(np.quantile(local[:, 1], 0.98)),
        "z_median": float(np.median(local[:, 2])),
    }
