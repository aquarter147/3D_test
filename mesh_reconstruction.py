import os
import torch
import numpy as np
import mcubes
import trimesh
from model import Nerf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load trained model ===
model = torch.load('model_nerf', map_location=device).to(device)
model.eval()

# === Parameters ===
N = 100               # Grid resolution
scale = 1.5           # Bounding box scale
batch_size = 65536    # How many points to process at once (tune this!)
save_path = "output_mesh.ply"

# === Create 3D grid ===
x = torch.linspace(-scale, scale, N)
y = torch.linspace(-scale, scale, N)
z = torch.linspace(-scale, scale, N)
x, y, z = torch.meshgrid(x, y, z, indexing='ij')

xyz = torch.cat([
    x.reshape(-1, 1),
    y.reshape(-1, 1),
    z.reshape(-1, 1)
], dim=1)  # (N^3, 3)

density_list = []

# === Run inference in batches to avoid OOM ===
with torch.no_grad():
    for i in range(0, xyz.shape[0], batch_size):
        xyz_batch = xyz[i:i + batch_size].to(device)
        d_batch = torch.zeros_like(xyz_batch)  # dummy ray directions
        _, density_batch = model(xyz_batch, d_batch)
        density_list.append(density_batch.cpu())

density = torch.cat(density_list, dim=0).numpy()
density = density.reshape(N, N, N)

# === Run Marching Cubes ===
iso_value = 30 * np.mean(density)
vertices, triangles = mcubes.marching_cubes(density, iso_value)

# Convert voxel to world coordinates
vertices = vertices / N * (2 * scale) - scale

# === Save Mesh ===
mesh = trimesh.Trimesh(vertices, triangles)
mesh.export(save_path)
print(f"Mesh saved to {save_path}")
