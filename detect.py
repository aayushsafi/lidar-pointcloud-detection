import sys
import numpy as np
import open3d as o3d

path = sys.argv[1] if len(sys.argv) > 1 else "street.pcd"
if path.endswith(".bin"):   # KITTI format: x, y, z, intensity as float32
    xyz = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
else:
    pcd = o3d.io.read_point_cloud(path)
print(f"Loaded:        {len(pcd.points)} points")

# 1. Downsample
pcd = pcd.voxel_down_sample(voxel_size=0.1)
print(f"Downsampled:   {len(pcd.points)} points")

# 2. Ground removal with RANSAC
plane, ground_idx = pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=1000)
a, b, c, d = plane
print(f"Ground plane:  {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0   ({len(ground_idx)} points)")
ground = pcd.select_by_index(ground_idx)
objects = pcd.select_by_index(ground_idx, invert=True)

# 3. Clustering with DBSCAN
labels = np.array(objects.cluster_dbscan(eps=0.5, min_points=10))
n = labels.max() + 1
print(f"Clusters:      {n}   (noise points: {(labels < 0).sum()})")

# 4. Box + size-based classification
colors = {"car": (0, 0.4, 1), "pedestrian": (1, 0, 0), "pole": (1, 0.6, 0),
          "wall/large": (0.5, 0.5, 0.5), "unknown": (0.6, 0, 0.8)}
def classify(l, w, h):
    l, w = max(l, w), min(l, w)
    if 3.0 < l < 6.5 and 1.4 < w < 2.6 and 1.0 < h < 2.6: return "car"
    if l < 1.0 and w < 1.0 and 1.2 < h < 2.1:            return "pedestrian"
    if l < 0.6 and w < 0.6 and h > 2.5:                  return "pole"
    if l > 8:                                            return "wall/large"
    return "unknown"

ground.paint_uniform_color([0.75, 0.75, 0.75])
objects.paint_uniform_color([0.2, 0.2, 0.2])
geoms, counts = [ground, objects], {}
for k in range(n):
    bb = objects.select_by_index(np.where(labels == k)[0]).get_axis_aligned_bounding_box()
    l, w, h = bb.get_extent()
    kind = classify(l, w, h)
    counts[kind] = counts.get(kind, 0) + 1
    cx, cy, _ = bb.get_center()
    print(f"  #{k:2d} {kind:11s} at ({cx:6.1f}, {cy:6.1f})  size {l:4.1f} x {w:4.1f} x {h:4.1f} m")
    bb.color = colors[kind]
    geoms.append(bb)
print("Summary:", counts)
o3d.visualization.draw_geometries(geoms, window_name="Detections")
