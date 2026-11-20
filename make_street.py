import numpy as np
import open3d as o3d

rng = np.random.default_rng(0)
scene = o3d.t.geometry.RaycastingScene()

def add(mesh, x, y, z=0.0, yaw=0.0):
    mesh.compute_vertex_normals()
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, 0, yaw)), center=(0, 0, 0))
    mesh.translate((x, y, z))
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

def box(w, d, h, x, y, yaw=0.0):           # sits on the road
    m = o3d.geometry.TriangleMesh.create_box(w, d, h)
    m.translate((-w / 2, -d / 2, 0))
    add(m, x, y, -1.7, yaw)

def cylinder(r, h, x, y):
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=h)
    m.translate((0, 0, h / 2))
    add(m, x, y, -1.7)

box(120, 120, 0.01, 0, 0)                  # road (sensor is 1.7 m above it)
box(80, 0.5, 6, 10, 9)                     # building walls
box(80, 0.5, 6, 10, -9)
box(4.5, 1.8, 1.5, 8, 5.5)                 # cars
box(4.5, 1.8, 1.5, 15, 5.5)
box(4.5, 1.8, 1.5, 12, -2.5, yaw=0.15)
box(4.5, 1.8, 1.5, -9, 2.5)
box(5.5, 2.0, 2.3, 24, -5.5)               # van
cylinder(0.25, 1.75, 6, -6.5)              # pedestrians
cylinder(0.25, 1.70, 6.8, -7.0)
cylinder(0.25, 1.80, -5, 7.0)
for px in (-15, 0, 15, 30):                # street lamps
    cylinder(0.12, 5.0, px, 7.8)

# 32 laser beams from -25 to +10 degrees, 1800 measurements per rotation
elev = np.radians(np.linspace(-25, 10, 32))
azim = np.radians(np.linspace(-180, 180, 1800, endpoint=False))
E, A = np.meshgrid(elev, azim)
dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
rays = np.hstack([np.zeros_like(dirs), dirs]).astype(np.float32)

t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
hit = np.isfinite(t) & (t < 60)                    # 60 m max range
r = t[hit] + rng.normal(0, 0.02, hit.sum())        # 2 cm range noise
pts = dirs[hit] * r[:, None]

o3d.io.write_point_cloud("street.pcd", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
print(f"Saved street.pcd with {len(pts)} points")
