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
    """Continuous signed axial height over a confirmed non-planar Margin polyline.

    An STL Margin is a cyclic *polyline*, not independent nearest-point samples.
    Nearest-vertex Z causes discontinuous cervical masks when the margin is
    scalloped. Project to the nearest 2D Margin *segment* instead, interpolate
    the two end-point axial heights, and preserve the confirmed 3D curve without
    fitting or flattening it.

    The field is used only to protect/edit existing surfaces. It does not invent
    an interior cutting surface or replace a 3D cervical stitch.
    """
    p = frame.to_local(points)
    m = frame.to_local(margin)
    if m.ndim != 2 or m.shape[1] != 3 or len(m) < 3:
        raise FrameError("Margin requires a valid 3D closed curve")
    if not np.isfinite(m).all() or not np.isfinite(p).all():
        raise FrameError("Margin and query points must be finite")
    if np.linalg.norm(m[0] - m[-1]) <= 1e-9:
        m = m[:-1]
    if len(m) < 3:
        raise FrameError("Margin closed curve has too few unique samples")
    a = m
    b = np.roll(m, -1, axis=0)
    segment = b[:, :2] - a[:, :2]
    length2 = np.einsum("ij,ij->i", segment, segment)
    if np.any(length2 < 1e-16):
        raise FrameError("Margin contains zero-length XY edges")

    middle = 0.5 * (a[:, :2] + b[:, :2])
    index = cKDTree(middle)
    k = min(8, len(m))
    _, candidates = index.query(p[:, :2], k=k)
    candidates = np.asarray(candidates, int).reshape((len(p), k))
    aa = a[candidates]
    bb = b[candidates]
    ee = bb[:, :, :2] - aa[:, :, :2]
    xy = p[:, None, :2] - aa[:, :, :2]
    t = np.clip(
        np.einsum("ijk,ijk->ij", xy, ee)
        / np.maximum(np.einsum("ijk,ijk->ij", ee, ee), 1e-16),
        0.0, 1.0,
    )
    foot = aa[:, :, :2] + t[:, :, None] * ee
    error2 = np.sum((p[:, None, :2] - foot) ** 2, axis=2)
    choice = np.argmin(error2, axis=1)
    row = np.arange(len(p))
    z_boundary = aa[row, choice, 2] + t[row, choice] * (
        bb[row, choice, 2] - aa[row, choice, 2]
    )
    return p[:, 2] - z_boundary

def margin_local_extents(margin: np.ndarray, frame: PatientFrame) -> dict:
    local = frame.to_local(margin)
    return {
        "x_min": float(np.quantile(local[:, 0], 0.02)),
        "x_max": float(np.quantile(local[:, 0], 0.98)),
        "y_min": float(np.quantile(local[:, 1], 0.02)),
        "y_max": float(np.quantile(local[:, 1], 0.98)),
        "z_median": float(np.median(local[:, 2])),
    }
