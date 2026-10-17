"""Patient-frame construction ported from CrownFlow's mature outer-crown stack.

Core frame is owned only by confirmed/accepted Margin + insertion axis:
Origin = Margin centroid; Z = insertion axis; X = principal axis of Margin projected
onto the plane perpendicular to Z; Y = Z×X; X is recomputed for strict right-handed
orthogonality.  AUTO_YangGuan then resolves the otherwise valid 180° X/Y sign
ambiguity with the working-arch outward radial direction, without changing Z.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .frame import PatientFrame


class PatientFrameError(ValueError):
    pass


def _unit(v, name="vector") -> np.ndarray:
    v = np.asarray(v, float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise PatientFrameError(f"{name} must be a finite 3-vector")
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        raise PatientFrameError(f"{name} is degenerate")
    return v / n


def validate_patient_frame(frame: PatientFrame, *, tol: float = 1e-6) -> dict:
    x, y, z = (np.asarray(v, float) for v in (frame.x, frame.y, frame.z))
    matrix = np.stack([x, y, z], axis=1)
    norms = [float(np.linalg.norm(v)) for v in (x, y, z)]
    dot_xy = float(x @ y)
    dot_xz = float(x @ z)
    dot_yz = float(y @ z)
    det = float(np.linalg.det(matrix))
    valid = bool(
        all(abs(n - 1.0) <= tol for n in norms)
        and abs(dot_xy) <= tol
        and abs(dot_xz) <= tol
        and abs(dot_yz) <= tol
        and det > 0.0
        and np.isfinite(np.asarray(frame.origin, float)).all()
    )
    if not valid:
        raise PatientFrameError("patient frame is not a finite right-handed orthonormal basis")
    return {
        "valid": True,
        "norms": norms,
        "dot_xy": dot_xy,
        "dot_xz": dot_xz,
        "dot_yz": dot_yz,
        "determinant": det,
    }


def build_patient_frame(
    margin_points: np.ndarray,
    insertion_axis: np.ndarray,
    *,
    working_jaw: trimesh.Trimesh | None = None,
) -> tuple[PatientFrame, dict]:
    margin = np.asarray(margin_points, float)
    if margin.ndim != 2 or margin.shape[1] != 3 or len(margin) < 8:
        raise PatientFrameError("Margin must contain at least 8 3D points")
    if not np.isfinite(margin).all():
        raise PatientFrameError("Margin contains NaN/Inf")

    z = _unit(insertion_axis, "insertion axis")
    origin = margin.mean(axis=0)
    rel = margin - origin
    projected = rel - np.outer(rel @ z, z)
    rms = float(np.sqrt(np.mean(np.sum(projected * projected, axis=1))))
    if rms < 1e-6:
        raise PatientFrameError("Margin projection collapses along insertion axis")

    covariance = projected.T @ projected / max(len(projected), 1)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    if eigvals[2] <= 1e-12 or eigvals[1] < 1e-6 * eigvals[2]:
        raise PatientFrameError("Margin projection does not define a stable in-plane frame")
    x = eigvecs[:, -1]
    x = _unit(x - z * float(x @ z), "Margin in-plane principal axis")
    y = _unit(np.cross(z, x), "patient-frame Y")
    x = _unit(np.cross(y, z), "patient-frame X")

    sign_source = "UNRESOLVED_CROWNFLOW_EQUIVALENT"
    radial_length = None
    arch_center = None
    if working_jaw is not None and len(working_jaw.vertices):
        vertices = np.asarray(working_jaw.vertices, float)
        arch_center = np.median(vertices, axis=0)
        radial = origin - arch_center
        radial = radial - z * float(radial @ z)
        radial_length = float(np.linalg.norm(radial))
        if radial_length >= 0.5:
            radial /= radial_length
            # Sign resolution only: keep the CrownFlow geometric frame, choose the
            # equivalent orientation whose +Y points outward/buccal.
            if float(y @ radial) < 0.0:
                x = -x
                y = -y
            sign_source = "ARCH_OUTWARD_BUCCAL_SIGN_ONLY"

    frame = PatientFrame(origin=np.asarray(origin, float), x=x, y=y, z=z)
    validation = validate_patient_frame(frame)
    report = {
        "method": "CROWNFLOW_MARGIN_PCA_FRAME_WITH_ARCH_SIGN_V1",
        "origin": origin.tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
        "z": z.tolist(),
        "margin_projection_rms_mm": rms,
        "margin_inplane_eigenvalues": [float(v) for v in eigvals],
        "sign_source": sign_source,
        "arch_center": None if arch_center is None else np.asarray(arch_center, float).tolist(),
        "arch_radial_length_mm": radial_length,
        "validation": validation,
    }
    return frame, report
