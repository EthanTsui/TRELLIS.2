#!/usr/bin/env python3
"""Lightweight SilhouetteCorrector test using existing GLBs.

Does NOT load the full pipeline — only the corrector.
Tests on pre-generated GLB files with reference images.
"""
import sys, os, time
sys.path.insert(0, '/workspace/TRELLIS.2')
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
import trimesh
from PIL import Image


def compute_dice(mask_a, mask_b):
    """Compute Dice coefficient between two binary masks."""
    intersection = (mask_a * mask_b).sum()
    return (2 * intersection / (mask_a.sum() + mask_b.sum() + 1e-8)).item()


def render_silhouette(glctx, mesh, yaw, pitch, r, fov, resolution=512):
    """Render mesh silhouette at given camera params."""
    import nvdiffrast.torch as dr
    import utils3d

    device = 'cuda'
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)

    fov_rad = torch.deg2rad(torch.tensor(float(fov))).to(device)
    yaw_t = torch.tensor(float(yaw)).to(device)
    pitch_t = torch.tensor(float(pitch)).to(device)

    orig_x = torch.sin(yaw_t) * torch.cos(pitch_t)
    orig_y = torch.cos(yaw_t) * torch.cos(pitch_t)
    orig_z = torch.sin(pitch_t)
    orig = torch.tensor([
        orig_x.item(), orig_z.item(), -orig_y.item()
    ], device=device) * r

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
    rast, _ = dr.rasterize(glctx, verts_clip, faces, (resolution, resolution))
    mask = (rast[0, :, :, 3] > 0).float()

    return mask


def main():
    import nvdiffrast.torch as dr
    from trellis2.postprocessing.silhouette_corrector import SilhouetteCorrector

    print("=== Lightweight Silhouette Corrector Test ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Use existing GLB files
    glb_paths = [
        '/workspace/TRELLIS.2/tmp/binary_search/C_grey_final.glb',
        '/workspace/TRELLIS.2/tmp/binary_search/B_clamps.glb',
    ]

    # Use T.png as reference for all (since we don't know which GLB maps to which input)
    ref_path = '/workspace/TRELLIS.2/assets/example_image/T.png'
    ref_img = Image.open(ref_path).convert('RGBA')

    # Get reference alpha mask
    ref_resized = ref_img.resize((512, 512), Image.LANCZOS)
    ref_alpha = np.array(ref_resized)[:, :, 3]
    ref_mask = torch.tensor((ref_alpha > 128).astype(np.float32), device='cuda')

    corrector = SilhouetteCorrector(device='cuda')
    glctx = dr.RasterizeCudaContext(device='cuda')

    yaw, pitch, r, fov = 0.0, 0.25, 2.0, 40.0

    for glb_path in glb_paths:
        if not os.path.exists(glb_path):
            print(f"\nSkipping {glb_path} (not found)")
            continue

        short = os.path.basename(glb_path)
        print(f"\n{'='*60}")
        print(f"GLB: {short}")
        print(f"{'='*60}")

        mesh = trimesh.load(glb_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

        # Measure Dice BEFORE
        mask_before = render_silhouette(glctx, mesh, yaw, pitch, r, fov, 512)
        dice_before = compute_dice(mask_before, ref_mask)
        print(f"  Dice BEFORE: {dice_before:.4f}")

        # Apply correction
        t0 = time.time()
        try:
            mesh_corrected = corrector.correct(
                mesh, ref_img,
                yaw=yaw, pitch=pitch, r=r, fov=fov,
                num_steps=80,
                verbose=True,
            )
            elapsed = time.time() - t0
            print(f"  Correction time: {elapsed:.1f}s")

            # Measure Dice AFTER
            mask_after = render_silhouette(glctx, mesh_corrected, yaw, pitch, r, fov, 512)
            dice_after = compute_dice(mask_after, ref_mask)
            diff = dice_after - dice_before
            print(f"  Dice AFTER: {dice_after:.4f}")
            print(f"  Improvement: {diff:+.4f} ({diff*100:+.1f}%)")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Test complete.")


if __name__ == '__main__':
    main()
