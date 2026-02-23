#!/usr/bin/env python3
"""Debug silhouette corrector camera alignment.

Renders a GLB mesh from the corrector's camera and saves the mask
for visual inspection. Also compares with different camera params.
"""
import sys, os
sys.path.insert(0, '/workspace/TRELLIS.2')
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
import trimesh
from PIL import Image
import nvdiffrast.torch as dr
import utils3d


def render_sil(glctx, mesh, yaw, pitch, r=2.0, fov=40.0, res=512):
    """Render silhouette mask."""
    device = 'cuda'
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)

    fov_rad = torch.deg2rad(torch.tensor(float(fov))).to(device)
    yaw_t = torch.tensor(float(yaw)).to(device)
    pitch_t = torch.tensor(float(pitch)).to(device)

    orig_x = torch.sin(yaw_t) * torch.cos(pitch_t)
    orig_y = torch.cos(yaw_t) * torch.cos(pitch_t)
    orig_z = torch.sin(pitch_t)
    orig = torch.tensor([orig_x.item(), orig_z.item(), -orig_y.item()], device=device) * r

    extr = utils3d.torch.extrinsics_look_at(
        orig,
        torch.zeros(3, device=device),
        torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    )
    intr = utils3d.torch.intrinsics_from_fov_xy(fov_rad, fov_rad)

    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    proj = torch.zeros((4, 4), dtype=torch.float32, device=device)
    proj[0, 0] = 2 * fx
    proj[1, 1] = 2 * fy
    proj[0, 2] = 2 * cx - 1
    proj[1, 2] = -2 * cy + 1
    proj[2, 2] = (100 + 0.1) / (100 - 0.1)
    proj[2, 3] = 2 * 0.1 * 100 / (0.1 - 100)
    proj[3, 2] = 1.0

    verts_homo = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
    verts_clip = (verts_homo @ (proj @ extr).T).unsqueeze(0)
    rast, _ = dr.rasterize(glctx, verts_clip, faces, (res, res))
    mask = (rast[0, :, :, 3] > 0).float()

    return mask


def main():
    print("=== Camera Debug ===")

    glctx = dr.RasterizeCudaContext(device='cuda')

    glb_path = '/workspace/TRELLIS.2/tmp/binary_search/C_grey_final.glb'
    mesh = trimesh.load(glb_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"Bounds: min={mesh.vertices.min(axis=0)}, max={mesh.vertices.max(axis=0)}")
    print(f"Center: {mesh.vertices.mean(axis=0)}")
    print(f"Scale: {np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))}")

    ref_path = '/workspace/TRELLIS.2/assets/example_image/T.png'
    ref_img = Image.open(ref_path).convert('RGBA')
    ref_resized = ref_img.resize((512, 512), Image.LANCZOS)
    ref_alpha = np.array(ref_resized)[:, :, 3]
    ref_mask = torch.tensor((ref_alpha > 128).astype(np.float32), device='cuda')

    ref_coverage = ref_mask.sum() / (512 * 512) * 100
    print(f"\nReference alpha coverage: {ref_coverage:.1f}%")

    # Test different camera params
    configs = [
        ("default", 0.0, 0.25, 2.0, 40.0),
        ("closer", 0.0, 0.25, 1.5, 40.0),
        ("wider_fov", 0.0, 0.25, 2.0, 60.0),
        ("no_pitch", 0.0, 0.0, 2.0, 40.0),
        ("high_pitch", 0.0, 0.5, 2.0, 40.0),
        ("closer_wider", 0.0, 0.25, 1.0, 60.0),
        ("very_close", 0.0, 0.25, 0.8, 40.0),
    ]

    for name, yaw, pitch, r, fov in configs:
        mask = render_sil(glctx, mesh, yaw, pitch, r, fov, 512)
        coverage = mask.sum() / (512 * 512) * 100
        intersection = (mask * ref_mask).sum()
        dice = (2 * intersection / (mask.sum() + ref_mask.sum() + 1e-8)).item()

        # Save mask as image
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mask_np).save(f'/tmp/sil_debug_{name}.png')

        print(f"  {name:20s}: coverage={coverage:.1f}%, dice={dice:.4f}, "
              f"r={r}, pitch={pitch}, fov={fov}")

    # Save reference alpha
    Image.fromarray(ref_alpha).save('/tmp/sil_debug_reference.png')
    print("\nMasks saved to /tmp/sil_debug_*.png")


if __name__ == '__main__':
    main()
