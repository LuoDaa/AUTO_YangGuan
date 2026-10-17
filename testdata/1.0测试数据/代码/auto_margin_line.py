import os
import json
import argparse
import numpy as np
import trimesh

from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial import cKDTree


# ============================================================
# 工具函数
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def normalize01(x):
    """
    Robust normalize to [0, 1]
    """
    x = np.asarray(x, dtype=np.float64)

    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x)

    lo = np.nanpercentile(x[valid], 5)
    hi = np.nanpercentile(x[valid], 95)

    if hi - lo < 1e-12:
        return np.zeros_like(x)

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def angular_diff(a, b):
    """
    两个角度的最小周期距离
    """
    d = np.abs(a - b)
    return np.minimum(d, 2.0 * np.pi - d)


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)

    if isinstance(mesh, trimesh.Scene):
        meshes = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(meshes)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("输入文件不是有效的 triangle mesh")

    mesh.remove_unreferenced_vertices()

    return mesh


# ============================================================
# 1. 自动找基台 component
# ============================================================

def choose_abutment_component(mesh, component_rank=1):
    """
    默认：
        最大 component = jaw
        第二大 component = abutment

    component_rank:
        0 = 最大
        1 = 第二大（默认）
        2 = 第三大
    """

    parts = mesh.split(only_watertight=False)

    if len(parts) == 0:
        raise RuntimeError("没有找到 mesh component")

    parts = sorted(
        parts,
        key=lambda m: len(m.faces),
        reverse=True
    )

    print("\nConnected components:")

    for i, p in enumerate(parts[:10]):
        print(
            f"  [{i}] faces={len(p.faces):7d}, "
            f"verts={len(p.vertices):7d}, "
            f"area={p.area:.3f}"
        )

    if len(parts) == 1:
        print("\n只有一个 component，直接使用整个 mesh")
        return parts[0]

    rank = min(component_rank, len(parts) - 1)

    abutment = parts[rank].copy()

    print(
        f"\n选择 component [{rank}] 作为基台候选，"
        f"faces={len(abutment.faces)}"
    )

    return abutment


# ============================================================
# 2. PCA 估计基台轴
# ============================================================

def estimate_axis(mesh):
    verts = np.asarray(mesh.vertices)

    center = np.mean(verts, axis=0)

    x = verts - center

    cov = np.cov(x.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]

    # 最大 PCA 主轴
    axis = normalize(eigvecs[:, order[0]])

    return center, axis


def build_local_basis(axis):
    """
    创建局部坐标：
        Z = 基台轴
        X/Y = 径向平面
    """

    axis = normalize(axis)

    refs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]

    ref = min(
        refs,
        key=lambda x: abs(np.dot(x, axis))
    )

    e1 = normalize(np.cross(axis, ref))
    e2 = normalize(np.cross(axis, e1))

    return e1, e2


def local_coordinates(vertices, center, axis, e1, e2):
    p = vertices - center

    z = p @ axis
    x = p @ e1
    y = p @ e2

    r = np.sqrt(x * x + y * y)

    theta = np.arctan2(y, x)

    return z, r, theta, x, y


# ============================================================
# 3. 自动调整基台轴正负
#    让“比较宽的一头”成为 cervical/base 方向
# ============================================================

def orient_axis(mesh, center, axis):
    e1, e2 = build_local_basis(axis)

    verts = np.asarray(mesh.vertices)

    z, r, _, _, _ = local_coordinates(
        verts, center, axis, e1, e2
    )

    z10 = np.percentile(z, 10)
    z25 = np.percentile(z, 25)

    z75 = np.percentile(z, 75)
    z90 = np.percentile(z, 90)

    low_mask = (z >= z10) & (z <= z25)
    high_mask = (z >= z75) & (z <= z90)

    low_r = np.median(r[low_mask])
    high_r = np.median(r[high_mask])

    # 希望比较宽的一侧放在 z 较低的一头
    if high_r > low_r:
        axis = -axis

    return normalize(axis)


# ============================================================
# 4. 计算每个 vertex 的锐边程度
# ============================================================

def compute_vertex_sharpness(mesh):
    """
    利用相邻三角面的夹角。

    每个 vertex 保存与它相连 edge 中最大的 face adjacency angle。
    """

    n_vertices = len(mesh.vertices)

    sharpness = np.zeros(n_vertices, dtype=np.float64)

    edges = np.asarray(mesh.face_adjacency_edges)

    if len(edges) == 0:
        return sharpness

    angles = np.asarray(mesh.face_adjacency_angles)

    np.maximum.at(sharpness, edges[:, 0], angles)
    np.maximum.at(sharpness, edges[:, 1], angles)

    return sharpness


# ============================================================
# 5. 自动定位 shoulder band
# ============================================================

def detect_shoulder_band(
    mesh,
    center,
    axis,
    e1,
    e2,
    bins=100,
    manual_width=None
):
    verts = np.asarray(mesh.vertices)

    z, r, _, _, _ = local_coordinates(
        verts,
        center,
        axis,
        e1,
        e2
    )

    vertex_normals = np.asarray(mesh.vertex_normals)

    normal_axis = np.abs(vertex_normals @ axis)

    sharpness = compute_vertex_sharpness(mesh)

    z_min = np.percentile(z, 1)
    z_max = np.percentile(z, 99)

    height = z_max - z_min

    edges = np.linspace(z_min, z_max, bins + 1)

    centers = 0.5 * (edges[:-1] + edges[1:])

    radius_profile = np.full(bins, np.nan)
    sharp_profile = np.full(bins, np.nan)
    normal_profile = np.full(bins, np.nan)

    for i in range(bins):

        mask = (
            (z >= edges[i]) &
            (z < edges[i + 1])
        )

        if np.sum(mask) < 5:
            continue

        # 外轮廓半径
        radius_profile[i] = np.percentile(
            r[mask], 90
        )

        # 锐边强度
        sharp_profile[i] = np.percentile(
            sharpness[mask], 90
        )

        # surface normal 与 axis 的关系
        normal_profile[i] = np.percentile(
            normal_axis[mask], 75
        )

    # ---------------------------
    # 填充空 bin
    # ---------------------------

    def fill_nan(a):
        a = a.copy()

        valid = np.isfinite(a)

        if np.sum(valid) < 2:
            return np.nan_to_num(a)

        ids = np.arange(len(a))

        a[~valid] = np.interp(
            ids[~valid],
            ids[valid],
            a[valid]
        )

        return a

    radius_profile = fill_nan(radius_profile)
    sharp_profile = fill_nan(sharp_profile)
    normal_profile = fill_nan(normal_profile)

    # 平滑 profile
    radius_smooth = gaussian_filter1d(
        radius_profile,
        sigma=2.0
    )

    sharp_smooth = gaussian_filter1d(
        sharp_profile,
        sigma=1.5
    )

    normal_smooth = gaussian_filter1d(
        normal_profile,
        sigma=1.5
    )

    # 半径一阶/二阶变化
    dr = np.gradient(
        radius_smooth,
        centers
    )

    d2r = np.gradient(
        dr,
        centers
    )

    # -------------------------------------------
    # shoulder score
    #
    # 1. 半径轮廓转折
    # 2. 锐边
    # 3. 法向变化
    # 4. 偏向外轮廓较大的区域
    # -------------------------------------------

    score = (
        1.30 * normalize01(np.abs(d2r))
        +
        1.00 * normalize01(sharp_smooth)
        +
        0.45 * normalize01(normal_smooth)
        +
        0.35 * normalize01(radius_smooth)
    )

    # -------------------------------------------
    # 避免找到顶部边缘
    #
    # shoulder 通常位于 cervical -> middle
    # -------------------------------------------

    z_norm = (
        (centers - z_min) /
        max(height, 1e-6)
    )

    allowed = (
        (z_norm >= 0.07) &
        (z_norm <= 0.68)
    )

    restricted_score = score.copy()

    restricted_score[~allowed] = -1e9

    best_idx = np.argmax(restricted_score)

    shoulder_center = float(centers[best_idx])

    if manual_width is None:
        shoulder_width = np.clip(
            height * 0.14,
            0.60,
            1.50
        )
    else:
        shoulder_width = float(manual_width)

    collar_min = shoulder_center - shoulder_width / 2
    collar_max = shoulder_center + shoulder_width / 2

    debug = {
        "z_min": float(z_min),
        "z_max": float(z_max),
        "height": float(height),
        "shoulder_center": shoulder_center,
        "shoulder_width": float(shoulder_width),
        "collar_min": float(collar_min),
        "collar_max": float(collar_max)
    }

    return collar_min, collar_max, debug


# ============================================================
# 6. 每一个角度寻找 Top-K margin candidates
# ============================================================

def build_candidates(
    mesh,
    center,
    axis,
    e1,
    e2,
    collar_min,
    collar_max,
    num_sectors=240,
    top_k=5
):
    verts = np.asarray(mesh.vertices)

    z, r, theta, _, _ = local_coordinates(
        verts,
        center,
        axis,
        e1,
        e2
    )

    vertex_normals = np.asarray(mesh.vertex_normals)

    normal_axis = np.abs(
        vertex_normals @ axis
    )

    sharpness = compute_vertex_sharpness(mesh)

    sharp_deg = np.degrees(sharpness)

    # shoulder band
    band_mask = (
        (z >= collar_min) &
        (z <= collar_max)
    )

    band_ids = np.where(band_mask)[0]

    if len(band_ids) < 20:
        raise RuntimeError(
            "Shoulder band 中点数太少"
        )

    band_r = r[band_ids]

    # 去掉明显内部结构，例如 screw hole
    outer_threshold = np.percentile(
        band_r,
        35
    )

    outer_mask = band_mask & (
        r >= outer_threshold
    )

    candidate_ids = np.where(outer_mask)[0]

    d_theta = 2.0 * np.pi / num_sectors

    candidates = []

    for i in range(num_sectors):

        angle = (
            -np.pi +
            (i + 0.5) * d_theta
        )

        diff = angular_diff(
            theta[candidate_ids],
            angle
        )

        # 正常搜索窗口
        local_ids = candidate_ids[
            diff <= d_theta * 1.7
        ]

        # 如果没有点，扩大角度
        if len(local_ids) < 2:

            local_ids = candidate_ids[
                diff <= d_theta * 3.5
            ]

        # 再没有就找角度最近的一些点
        if len(local_ids) == 0:

            nearest = np.argsort(diff)[:max(top_k, 3)]

            local_ids = candidate_ids[nearest]

        rr = r[local_ids]

        ss = sharp_deg[local_ids]

        nn = normal_axis[local_ids]

        zz = z[local_ids]

        # -------------------------------
        # 每个 sector 内做归一化
        # -------------------------------

        r_score = normalize01(rr)

        # 45 度以上基本当作很强锐边
        sharp_score = np.clip(
            ss / 45.0,
            0.0,
            1.0
        )

        normal_score = np.clip(
            nn,
            0.0,
            1.0
        )

        shoulder_center = (
            collar_min + collar_max
        ) / 2.0

        sigma = max(
            (collar_max - collar_min) * 0.55,
            1e-4
        )

        z_score = np.exp(
            -0.5 *
            (
                (zz - shoulder_center)
                / sigma
            ) ** 2
        )

        # -------------------------------
        # 综合评分
        # -------------------------------

        total_score = (
            0.47 * r_score
            +
            0.33 * sharp_score
            +
            0.12 * normal_score
            +
            0.08 * z_score
        )

        order = np.argsort(
            total_score
        )[::-1]

        order = order[:top_k]

        sector_candidates = []

        for j in order:

            idx = int(local_ids[j])

            sector_candidates.append({
                "vertex_id": idx,
                "score": float(total_score[j]),
                "z": float(z[idx]),
                "r": float(r[idx]),
                "theta": float(theta[idx])
            })

        candidates.append(
            sector_candidates
        )

    return candidates


# ============================================================
# 7. Dynamic Programming
#    从全部 Top-K candidate 中寻找最平滑闭环
# ============================================================

def closed_loop_dp(
    candidates,
    collar_width,
    smooth_z=1.8,
    smooth_r=0.35
):
    n = len(candidates)

    if n < 3:
        raise RuntimeError(
            "sector 数量太少"
        )

    median_r = []

    for sec in candidates:
        for p in sec:
            median_r.append(p["r"])

    median_r = max(
        np.median(median_r),
        1e-6
    )

    best_total = np.inf
    best_path = None

    k0 = len(candidates[0])

    # 为了真正形成闭环：
    # 枚举第 0 个 sector 的起点
    for start_j in range(k0):

        previous = np.full(
            len(candidates[0]),
            np.inf
        )

        previous[start_j] = (
            -candidates[0][start_j]["score"]
        )

        backtrack = []

        for i in range(1, n):

            prev_sec = candidates[i - 1]
            curr_sec = candidates[i]

            current = np.full(
                len(curr_sec),
                np.inf
            )

            parent = np.full(
                len(curr_sec),
                -1,
                dtype=np.int32
            )

            for j, curr in enumerate(curr_sec):

                best_cost = np.inf
                best_parent = -1

                for k, prev in enumerate(prev_sec):

                    if not np.isfinite(previous[k]):
                        continue

                    dz = (
                        curr["z"] -
                        prev["z"]
                    ) / max(
                        collar_width,
                        1e-6
                    )

                    dr = (
                        curr["r"] -
                        prev["r"]
                    ) / median_r

                    transition = (
                        smooth_z * dz * dz
                        +
                        smooth_r * dr * dr
                    )

                    unary = -curr["score"]

                    cost = (
                        previous[k]
                        +
                        unary
                        +
                        transition
                    )

                    if cost < best_cost:
                        best_cost = cost
                        best_parent = k

                current[j] = best_cost
                parent[j] = best_parent

            previous = current
            backtrack.append(parent)

        # --------------------------------
        # 闭环：最后一个 sector
        # 与第一个 sector 的起点连接
        # --------------------------------

        start = candidates[0][start_j]

        for last_j, last in enumerate(candidates[-1]):

            dz = (
                last["z"] -
                start["z"]
            ) / max(
                collar_width,
                1e-6
            )

            dr = (
                last["r"] -
                start["r"]
            ) / median_r

            closure = (
                smooth_z * dz * dz
                +
                smooth_r * dr * dr
            )

            total = (
                previous[last_j]
                +
                closure
            )

            if total < best_total:

                best_total = total

                path = [0] * n

                path[-1] = last_j

                current_j = last_j

                for i in range(
                    n - 1,
                    0,
                    -1
                ):
                    parent = backtrack[i - 1]

                    current_j = parent[current_j]

                    path[i - 1] = current_j

                path[0] = start_j

                best_path = path

    if best_path is None:
        raise RuntimeError(
            "Dynamic Programming 未找到 margin"
        )

    return best_path, float(best_total)


# ============================================================
# 8. 转换 DP 结果
# ============================================================

def extract_raw_margin(
    mesh,
    candidates,
    path
):
    verts = np.asarray(mesh.vertices)

    points = []
    scores = []

    for i, j in enumerate(path):

        c = candidates[i][j]

        points.append(
            verts[c["vertex_id"]]
        )

        scores.append(
            c["score"]
        )

    return (
        np.asarray(points),
        np.asarray(scores)
    )


# ============================================================
# 9. 周期平滑
# ============================================================

def smooth_margin(
    raw_points,
    center,
    axis,
    e1,
    e2,
    num_sectors,
    median_size=7,
    gaussian_sigma=1.8
):
    z, r, theta, _, _ = local_coordinates(
        raw_points,
        center,
        axis,
        e1,
        e2
    )

    # 周期中值滤波
    z_clean = median_filter(
        z,
        size=median_size,
        mode="wrap"
    )

    r_clean = median_filter(
        r,
        size=median_size,
        mode="wrap"
    )

    # Gaussian 周期平滑
    z_smooth = gaussian_filter1d(
        z_clean,
        sigma=gaussian_sigma,
        mode="wrap"
    )

    r_smooth = gaussian_filter1d(
        r_clean,
        sigma=gaussian_sigma,
        mode="wrap"
    )

    d_theta = (
        2.0 *
        np.pi /
        num_sectors
    )

    sector_theta = (
        -np.pi
        +
        (
            np.arange(num_sectors)
            + 0.5
        ) * d_theta
    )

    points = []

    for z_i, r_i, t_i in zip(
        z_smooth,
        r_smooth,
        sector_theta
    ):

        radial_dir = (
            np.cos(t_i) * e1
            +
            np.sin(t_i) * e2
        )

        p = (
            center
            +
            z_i * axis
            +
            r_i * radial_dir
        )

        points.append(p)

    return np.asarray(points)


# ============================================================
# 10. 把平滑后的曲线重新吸附到 mesh
# ============================================================

def snap_to_surface(mesh, points):
    """
    优先使用 trimesh closest_point。

    如果系统缺少 rtree，
    自动 fallback 到最近 vertex。
    """

    try:

        closest, distance, _ = (
            trimesh.proximity.closest_point(
                mesh,
                points
            )
        )

        if (
            closest is not None
            and
            np.all(np.isfinite(closest))
        ):
            return closest

    except Exception:
        pass

    # fallback
    tree = cKDTree(
        np.asarray(mesh.vertices)
    )

    _, ids = tree.query(
        points,
        k=1
    )

    return np.asarray(
        mesh.vertices
    )[ids]


# ============================================================
# 11. 输出 OBJ polyline
# ============================================================

def save_polyline_obj(
    path,
    points,
    close_loop=True
):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# Auto Margin Line\n"
        )

        for p in points:

            f.write(
                f"v "
                f"{p[0]:.8f} "
                f"{p[1]:.8f} "
                f"{p[2]:.8f}\n"
            )

        ids = list(
            range(
                1,
                len(points) + 1
            )
        )

        if close_loop:
            ids.append(1)

        f.write(
            "l "
            +
            " ".join(map(str, ids))
            +
            "\n"
        )


def save_csv(path, points):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "index,x_mm,y_mm,z_mm\n"
        )

        for i, p in enumerate(points):

            f.write(
                f"{i},"
                f"{p[0]:.8f},"
                f"{p[1]:.8f},"
                f"{p[2]:.8f}\n"
            )


# ============================================================
# 主算法
# ============================================================

def extract_margin(
    input_stl,
    output_dir,
    whole_mesh=False,
    component_rank=1,
    num_sectors=240,
    top_k=5,
    shoulder_width=None
):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("\n================================")
    print(" Auto Crown Margin Extraction")
    print("================================\n")

    full_mesh = load_mesh(
        input_stl
    )

    print(
        "Input mesh:",
        len(full_mesh.vertices),
        "vertices /",
        len(full_mesh.faces),
        "faces"
    )

    # ----------------------------------
    # 基台
    # ----------------------------------

    if whole_mesh:
        abutment = full_mesh.copy()
    else:
        abutment = choose_abutment_component(
            full_mesh,
            component_rank
        )

    abutment.export(
        os.path.join(
            output_dir,
            "abutment_candidate.stl"
        )
    )

    # ----------------------------------
    # Axis
    # ----------------------------------

    center, axis = estimate_axis(
        abutment
    )

    axis = orient_axis(
        abutment,
        center,
        axis
    )

    e1, e2 = build_local_basis(
        axis
    )

    print("\nAbutment center:")
    print(center)

    print("\nAbutment axis:")
    print(axis)

    # ----------------------------------
    # Shoulder band
    # ----------------------------------

    collar_min, collar_max, shoulder_debug = (
        detect_shoulder_band(
            abutment,
            center,
            axis,
            e1,
            e2,
            manual_width=shoulder_width
        )
    )

    print("\nDetected shoulder band:")

    print(
        " center =",
        shoulder_debug[
            "shoulder_center"
        ]
    )

    print(
        " min    =",
        collar_min
    )

    print(
        " max    =",
        collar_max
    )

    # ----------------------------------
    # Top-K candidates
    # ----------------------------------

    candidates = build_candidates(
        abutment,
        center,
        axis,
        e1,
        e2,
        collar_min,
        collar_max,
        num_sectors=num_sectors,
        top_k=top_k
    )

    # ----------------------------------
    # Closed-loop DP
    # ----------------------------------

    path, dp_energy = closed_loop_dp(
        candidates,
        collar_width=(
            collar_max -
            collar_min
        )
    )

    raw_margin, raw_scores = (
        extract_raw_margin(
            abutment,
            candidates,
            path
        )
    )

    # ----------------------------------
    # Smooth
    # ----------------------------------

    smooth_points = smooth_margin(
        raw_margin,
        center,
        axis,
        e1,
        e2,
        num_sectors=num_sectors
    )

    # ----------------------------------
    # Snap 回原始基台 surface
    # ----------------------------------

    final_margin = snap_to_surface(
        abutment,
        smooth_points
    )

    # ----------------------------------
    # 输出
    # ----------------------------------

    raw_obj = os.path.join(
        output_dir,
        "margin_raw.obj"
    )

    final_obj = os.path.join(
        output_dir,
        "margin.obj"
    )

    csv_path = os.path.join(
        output_dir,
        "margin.csv"
    )

    save_polyline_obj(
        raw_obj,
        raw_margin
    )

    save_polyline_obj(
        final_obj,
        final_margin
    )

    save_csv(
        csv_path,
        final_margin
    )

    # ----------------------------------
    # 简单 confidence
    # ----------------------------------

    mean_score = float(
        np.mean(raw_scores)
    )

    z_raw, _, _, _, _ = (
        local_coordinates(
            final_margin,
            center,
            axis,
            e1,
            e2
        )
    )

    dz = np.diff(
        np.r_[
            z_raw,
            z_raw[0]
        ]
    )

    roughness = float(
        np.mean(np.abs(dz))
    )

    collar_width_value = (
        collar_max -
        collar_min
    )

    smooth_quality = np.exp(
        -roughness /
        max(
            collar_width_value * 0.15,
            1e-6
        )
    )

    confidence = float(
        np.clip(
            0.7 * mean_score
            +
            0.3 * smooth_quality,
            0.0,
            1.0
        )
    )

    result = {
        "input": os.path.abspath(
            input_stl
        ),

        "num_margin_points": int(
            len(final_margin)
        ),

        "abutment_center": (
            center.tolist()
        ),

        "abutment_axis": (
            axis.tolist()
        ),

        "shoulder_band": (
            shoulder_debug
        ),

        "dp_energy": dp_energy,

        "mean_candidate_score":
            mean_score,

        "confidence":
            confidence,

        "output": {
            "margin_obj":
                "margin.obj",

            "margin_raw_obj":
                "margin_raw.obj",

            "margin_csv":
                "margin.csv",

            "abutment_stl":
                "abutment_candidate.stl"
        }
    }

    with open(
        os.path.join(
            output_dir,
            "result.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n================================")

    print(
        "Margin points:",
        len(final_margin)
    )

    print(
        "Confidence:",
        round(
            confidence,
            4
        )
    )

    print(
        "Output:",
        final_obj
    )

    print("================================\n")

    return final_margin


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatic abutment crown margin "
            "closed-loop extraction"
        )
    )

    parser.add_argument(
        "input",
        help="输入 STL 文件"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="margin_output",
        help="输出目录"
    )

    parser.add_argument(
        "--whole-mesh",
        action="store_true",
        help=(
            "如果输入 STL 本身就是单独基台，"
            "使用此参数"
        )
    )

    parser.add_argument(
        "--component-rank",
        type=int,
        default=1,
        help=(
            "connected component 排名。"
            "默认 1，即第二大 component"
        )
    )

    parser.add_argument(
        "--num-sectors",
        type=int,
        default=240,
        help="Margin 环向采样点数量"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="每个角度保留多少候选点"
    )

    parser.add_argument(
        "--shoulder-width",
        type=float,
        default=None,
        help=(
            "手动指定肩台搜索宽度，单位 mm。"
            "默认自动估计"
        )
    )

    args = parser.parse_args()

    extract_margin(
        input_stl=args.input,
        output_dir=args.output,
        whole_mesh=args.whole_mesh,
        component_rank=args.component_rank,
        num_sectors=args.num_sectors,
        top_k=args.top_k,
        shoulder_width=args.shoulder_width
    )


if __name__ == "__main__":
    main()