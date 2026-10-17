from __future__ import annotations

from pathlib import Path
import numpy as np
import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    obj = trimesh.load_mesh(str(path), process=False)
    if isinstance(obj, trimesh.Scene):
        if not obj.geometry:
            raise ValueError(f"empty scene: {path}")
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh object: {type(obj)!r}")
    if len(obj.vertices) == 0 or len(obj.faces) == 0:
        raise ValueError(f"mesh is empty: {path}")
    if not np.isfinite(np.asarray(obj.vertices, float)).all():
        raise ValueError(f"mesh contains NaN/Inf: {path}")
    return obj


def save_mesh(mesh: trimesh.Trimesh, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def closest_points(mesh: trimesh.Trimesh, points: np.ndarray):
    """Triangle closest-point query with a no-rtree fallback."""
    points = np.asarray(points, float)
    try:
        cp, dist, fid = trimesh.proximity.closest_point(mesh, points)
    except Exception:
        cp, dist, fid = trimesh.proximity.closest_point_naive(mesh, points)
    return np.asarray(cp, float), np.asarray(dist, float), np.asarray(fid, int)


def oriented_face_normals_toward(mesh: trimesh.Trimesh, face_ids: np.ndarray,
                                 closest: np.ndarray, target: np.ndarray) -> np.ndarray:
    face_ids = np.asarray(face_ids, int)
    normals = np.asarray(mesh.face_normals, float)[face_ids].copy()
    n = np.linalg.norm(normals, axis=1)
    good = n > 1e-12
    normals[good] /= n[good, None]
    normals[~good] = np.array([0.0, 0.0, 1.0])
    toward = np.asarray(target, float)[None, :] - np.asarray(closest, float)
    flip = np.einsum("ij,ij->i", normals, toward) < 0.0
    normals[flip] *= -1.0
    return normals


def vertex_adjacency(mesh: trimesh.Trimesh) -> list[list[int]]:
    adj = [set() for _ in range(len(mesh.vertices))]
    for a, b in np.asarray(mesh.edges_unique, int):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return [sorted(v) for v in adj]


def diffuse_displacement(mesh: trimesh.Trimesh, seed_displacement: np.ndarray,
                         seed_mask: np.ndarray, editable_mask: np.ndarray,
                         locked_mask: np.ndarray, *, iterations: int, alpha: float) -> np.ndarray:
    """Diffuse a bounded displacement field without moving locked vertices.

    Seed vertices remain exact anchors. Non-seed editable vertices receive a
    smooth average of neighbouring displacements. This is intentionally weaker
    than a global Laplacian solve; the MVP favours anatomy preservation.
    """
    disp = np.asarray(seed_displacement, float).copy()
    seed_mask = np.asarray(seed_mask, bool)
    editable_mask = np.asarray(editable_mask, bool)
    locked_mask = np.asarray(locked_mask, bool)
    disp[locked_mask] = 0.0
    adj = vertex_adjacency(mesh)
    out = disp.copy()
    for _ in range(max(0, int(iterations))):
        prev = out.copy()
        for i in np.flatnonzero(editable_mask & ~seed_mask & ~locked_mask):
            nei = [j for j in adj[int(i)] if editable_mask[j] and not locked_mask[j]]
            if not nei:
                continue
            avg = np.mean(prev[np.asarray(nei, int)], axis=0)
            out[i] = (1.0 - alpha) * prev[i] + alpha * avg
        out[seed_mask] = disp[seed_mask]
        out[locked_mask] = 0.0
    return out


def cap_vectors(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    out = np.asarray(vectors, float).copy()
    n = np.linalg.norm(out, axis=1)
    mask = n > float(max_norm)
    if np.any(mask):
        out[mask] *= (float(max_norm) / n[mask])[:, None]
    return out


def face_flip_count(reference: trimesh.Trimesh, current: trimesh.Trimesh) -> int | None:
    if len(reference.faces) != len(current.faces) or len(reference.vertices) != len(current.vertices):
        return None
    f = np.asarray(reference.faces, int)
    if not np.array_equal(f, np.asarray(current.faces, int)):
        return None
    a = np.asarray(reference.face_normals, float)
    b = np.asarray(current.face_normals, float)
    return int(np.sum(np.einsum("ij,ij->i", a, b) <= 0.0))
