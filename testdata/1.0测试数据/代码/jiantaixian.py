import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
def extract_margin_line_robust_core(
    mesh, collar_min_x, collar_max_x, num_sectors=360
):
  """提取非对称肩台整圈 3D 曲线及 X_min（蓝线）"""
  verts = mesh.vertices
  r = np.sqrt(verts[:, 1] ** 2 + verts[:, 2] ** 2)

  face_angles_deg = np.degrees(mesh.face_adjacency_angles)
  sharp_mask = face_angles_deg > 18.0
  sharp_edges = mesh.face_adjacency_edges[sharp_mask]

  if len(sharp_edges) > 0:
    sharp_verts_idx = np.unique(sharp_edges)
    sharp_verts = verts[sharp_verts_idx]
    sharp_filter = (
        (sharp_verts[:, 0] >= collar_min_x)
        & (sharp_verts[:, 0] <= collar_max_x)
        & (np.sqrt(sharp_verts[:, 1] ** 2 + sharp_verts[:, 2] ** 2) > 1.8)
    )
    collar_sharp_pts = sharp_verts[sharp_filter]
  else:
    collar_sharp_pts = np.empty((0, 3))

  d_theta = 2 * np.pi / num_sectors
  margin_points = []
  outer_pts = verts[
      (r > 1.8)
      & (verts[:, 0] >= collar_min_x)
      & (verts[:, 0] <= collar_max_x)
  ]
  thetas_outer = np.arctan2(outer_pts[:, 2], outer_pts[:, 1])
  thetas_sharp = (
      np.arctan2(collar_sharp_pts[:, 2], collar_sharp_pts[:, 1])
      if len(collar_sharp_pts) > 0
      else np.array([])
  )

  for i in range(num_sectors):
    angle = -np.pi + i * d_theta
    found_point = None

    if len(thetas_sharp) > 0:
      diff_s = np.abs(thetas_sharp - angle)
      diff_s = np.minimum(diff_s, 2 * np.pi - diff_s)
      sec_sharp = collar_sharp_pts[diff_s <= d_theta * 1.5]
      if len(sec_sharp) > 0:
        sec_r = np.sqrt(sec_sharp[:, 1] ** 2 + sec_sharp[:, 2] ** 2)
        found_point = sec_sharp[np.argmax(sec_r)]

    if found_point is None and len(outer_pts) > 0:
      diff_o = np.abs(thetas_outer - angle)
      diff_o = np.minimum(diff_o, 2 * np.pi - diff_o)
      sec_out = outer_pts[diff_o <= d_theta * 1.2]
      if len(sec_out) >= 3:
        sec_r = np.sqrt(sec_out[:, 1] ** 2 + sec_out[:, 2] ** 2)
        found_point = sec_out[np.argmax(sec_r)]

    if found_point is not None:
      margin_points.append(found_point)

  margin_points = np.array(margin_points)
  clean_x = medfilt(margin_points[:, 0], kernel_size=7)
  clean_y = medfilt(margin_points[:, 1], kernel_size=7)
  clean_z = medfilt(margin_points[:, 2], kernel_size=7)

  smooth_x = gaussian_filter1d(clean_x, sigma=2.0, mode='wrap')
  smooth_y = gaussian_filter1d(clean_y, sigma=2.0, mode='wrap')
  smooth_z = gaussian_filter1d(clean_z, sigma=2.0, mode='wrap')
  margin_curve = np.column_stack([smooth_x, smooth_y, smooth_z])

  xmin_idx = np.argmin(smooth_x)
  x_margin_min = float(smooth_x[xmin_idx])
  margin_min_point = margin_curve[xmin_idx]

  return margin_curve, x_margin_min, margin_min_point
