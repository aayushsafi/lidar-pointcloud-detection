import numpy as np
import open3d as o3d

# 1. Load a sample 3D scan of a room (downloaded automatically the first time)
pcd = o3d.io.read_point_cloud(o3d.data.PLYPointCloud().path)
pts = np.asarray(pcd.points)          # the N x 3 array
print(pcd)
print("Shape:", pts.shape)
print("Min corner:", pts.min(axis=0))
print("Max corner:", pts.max(axis=0))
o3d.visualization.draw_geometries([pcd], window_name="1. Original")

# 2. Downsample: keep one point per 3 cm cube
down = pcd.voxel_down_sample(voxel_size=0.03)
print("After downsampling:", len(down.points), "points")
o3d.visualization.draw_geometries([down], window_name="2. Downsampled")

# 3. Remove outliers: points far from their 20 neighbours
clean, keep_idx = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
outliers = down.select_by_index(keep_idx, invert=True)
clean.paint_uniform_color([0.6, 0.6, 0.6])     # kept points: grey
outliers.paint_uniform_color([1.0, 0.0, 0.0])  # removed points: red
print("Outliers removed:", len(outliers.points))
o3d.visualization.draw_geometries([clean, outliers], window_name="3. Outliers in red")
