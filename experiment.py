import sys
import open3d as o3d

voxel = float(sys.argv[1])        # downsampling cube size in metres
ratio = float(sys.argv[2])        # outlier strictness (smaller = stricter)

pcd = o3d.io.read_point_cloud(o3d.data.PLYPointCloud().path)
down = pcd.voxel_down_sample(voxel_size=voxel)
clean, keep = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=ratio)
out = down.select_by_index(keep, invert=True)

pct = 100 * len(out.points) / max(len(down.points), 1)
print(f"voxel={voxel} m  std_ratio={ratio}")
print(f"  original:   {len(pcd.points)} points")
print(f"  downsampled:{len(down.points)} points")
print(f"  outliers:   {len(out.points)} ({pct:.1f} %)")

clean.paint_uniform_color([0.6, 0.6, 0.6])
out.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([clean, out],
    window_name=f"voxel {voxel} m, std_ratio {ratio}")
