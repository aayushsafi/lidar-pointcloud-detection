import sys
import numpy as np
import open3d as o3d

path = sys.argv[1] if len(sys.argv) > 1 else "street.pcd"
show = "--no-show" not in sys.argv

if path.endswith(".bin"):   # KITTI format: x, y, z, intensity as float32
    xyz = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
else:
    pcd = o3d.io.read_point_cloud(path)

# 1. Downsample   2. Ground removal (RANSAC)   3. Clustering (DBSCAN)
pcd = pcd.voxel_down_sample(voxel_size=0.1)
_, ground_idx = pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=1000)
ground = pcd.select_by_index(ground_idx)
objects = pcd.select_by_index(ground_idx, invert=True)
labels = np.array(objects.cluster_dbscan(eps=0.5, min_points=4))
pts_all = np.asarray(objects.points)

# NEW 1: oriented box that only rotates around the vertical axis (yaw).
# "L-shape fitting": try every angle 0..89 degrees and keep the one where the
# points lie closest to the box edges (a LiDAR only sees the sides of a car).
def yaw_box(pts):
    xy = pts[:, :2]
    best = None
    for deg in range(90):
        t = np.radians(deg)
        R2 = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        p = xy @ R2                         # points in the rotated frame
        lo, hi = p.min(0), p.max(0)
        d = np.minimum(np.minimum(p - lo, hi - p)[:, 0], np.minimum(p - lo, hi - p)[:, 1])
        score = d.mean()                    # average distance to the nearest edge
        if best is None or score < best[0]:
            best = (score, t, lo, hi)
    _, t, lo, hi = best
    R2 = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    cxy = R2 @ ((lo + hi) / 2)              # centre back in the world frame
    zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
    R = np.eye(3); R[:2, :2] = R2
    center = [cxy[0], cxy[1], (zmin + zmax) / 2]
    extent = [hi[0] - lo[0], hi[1] - lo[1], zmax - zmin]
    return o3d.geometry.OrientedBoundingBox(center, R, extent), np.degrees(t)

def classify(l, w, h):
    l, w = max(l, w), min(l, w)
    if 3.0 < l < 6.5 and 1.4 < w < 2.6 and 1.0 < h < 2.6: return "car"
    if l < 1.0 and w < 1.0 and 1.2 < h < 2.1:            return "pedestrian"
    if l < 0.6 and w < 0.6 and h > 2.5:                  return "pole"
    if l > 8:                                            return "wall/large"
    return "unknown"

colors = {"car": (0, 0.4, 1), "pedestrian": (1, 0, 0), "pole": (1, 0.6, 0),
          "wall/large": (0.5, 0.5, 0.5), "unknown": (0.6, 0, 0.8)}
ground.paint_uniform_color([0.75, 0.75, 0.75])
objects.paint_uniform_color([0.2, 0.2, 0.2])
geoms, counts, dropped = [ground, objects], {}, 0

for k in range(labels.max() + 1):
    pts = pts_all[labels == k]
    # NEW 2: ignore scraps: too few points or less than 30 cm tall
    if len(pts) < 8 or np.ptp(pts[:, 2]) < 0.3:
        dropped += 1
        continue
    box, yaw = yaw_box(pts)
    l, w, h = box.extent
    kind = classify(l, w, h)
    counts[kind] = counts.get(kind, 0) + 1
    cx, cy = box.center[:2]
    dist = np.hypot(cx, cy)
    print(f"{kind:11s} at ({cx:6.1f}, {cy:6.1f})  {dist:4.1f} m away  "
          f"size {max(l, w):4.1f} x {min(l, w):4.1f} x {h:4.1f} m  yaw {yaw:3.0f} deg  ({len(pts)} pts)")
    box.color = colors[kind]
    geoms.append(box)

print(f"Dropped {dropped} tiny clusters")
print("Summary:", counts)

if show:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Detections v2", width=1400, height=900)
    for g in geoms:
        vis.add_geometry(g)
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1]); ctr.set_up([1, 0, 0]); ctr.set_lookat([5, 0, 0]); ctr.set_zoom(0.35)
    for _ in range(3):
        vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image("detections.png")      # picture for your GitHub README
    print("Saved detections.png")
    vis.run()
    vis.destroy_window()
