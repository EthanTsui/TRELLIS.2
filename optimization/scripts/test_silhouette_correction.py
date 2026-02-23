#!/usr/bin/env python3
"""Test SilhouetteCorrector on generated GLB meshes.

Generates meshes, applies silhouette correction, and measures
A1 silhouette Dice improvement using V4 evaluation methodology.
"""
import sys, os, time, gc
sys.path.insert(0, '/workspace/TRELLIS.2')
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
from PIL import Image
import trimesh


def compute_dice(mask_a, mask_b):
    """Compute Dice coefficient between two binary masks."""
    intersection = (mask_a * mask_b).sum()
    return (2 * intersection / (mask_a.sum() + mask_b.sum() + 1e-8)).item()


def render_silhouette_for_eval(mesh, yaw, pitch, r, fov, resolution=512):
    """Render mesh silhouette at given camera params (non-differentiable)."""
    import nvdiffrast.torch as dr
    import utils3d

    device = 'cuda'
    glctx = dr.RasterizeCudaContext(device=device)

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
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.postprocessing.silhouette_corrector import SilhouetteCorrector
    import o_voxel.postprocess

    print("=== Silhouette Correction A/B Test ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    corrector = SilhouetteCorrector(device='cuda')

    # Champion config
    ss_params = {"steps": 12, "guidance_strength": 10.0, "guidance_rescale": 0.8,
                 "rescale_t": 5.0, "multistep": True}
    shape_params = {"steps": 12, "guidance_strength": 10.0, "guidance_rescale": 0.5,
                    "rescale_t": 3.0, "multistep": True}
    tex_params = {"steps": 16, "guidance_strength": 12.0, "guidance_rescale": 1.0,
                  "rescale_t": 4.0, "heun_steps": 4, "multistep": True}

    example_dir = '/workspace/TRELLIS.2/assets/example_image'
    test_images = ['T.png',
                   '0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp',
                   '454e7d8a30486c0635369936e7bec5677b78ae5f436d0e46af0d533738be859f.webp']

    results = []

    for img_name in test_images:
        img_path = os.path.join(example_dir, img_name)
        if not os.path.exists(img_path):
            print(f"\nSkipping {img_name} (not found)", flush=True)
            continue

        short = img_name[:20] + '...' if len(img_name) > 20 else img_name
        print(f"\n{'='*60}", flush=True)
        print(f"Image: {short}", flush=True)
        print(f"{'='*60}", flush=True)

        img = Image.open(img_path)
        processed = pipeline.preprocess_image(img)

        # Generate mesh
        print(f"  Generating (seed=42)...", flush=True)
        t0 = time.time()
        outputs = pipeline.run(
            processed, seed=42, preprocess_image=False,
            sparse_structure_sampler_params=ss_params,
            shape_slat_sampler_params=shape_params,
            tex_slat_sampler_params=tex_params,
            pipeline_type="1024_cascade",
            max_num_tokens=65536,
        )
        gen_time = time.time() - t0
        raw_mesh = outputs[0]
        print(f"  Generation: {gen_time:.1f}s", flush=True)

        # Create GLB (without silhouette correction)
        print(f"  Creating GLB...", flush=True)
        glb = o_voxel.postprocess.to_glb(
            vertices=raw_mesh.vertices,
            faces=raw_mesh.faces,
            attr_volume=raw_mesh.attrs,
            coords=raw_mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=1024 // 16,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=800000,
            texture_size=2048,
            remesh=True,
            remesh_band=1,
            remesh_project=0.9,
            max_metallic=0.05,
            min_roughness=0.4,
            enable_normal_map=False,
            enable_ao=False,
            enable_grey_recovery=True,
            use_tqdm=True,
            verbose=False,
        )

        # Get reference alpha mask
        ref_alpha = np.array(processed.convert('RGBA').resize((512, 512), Image.LANCZOS))[:, :, 3]
        ref_mask = torch.tensor((ref_alpha > 128).astype(np.float32), device='cuda')

        # Measure Dice BEFORE correction
        print(f"  Measuring Dice before correction...", flush=True)
        yaw, pitch, r, fov = 0.0, 0.25, 2.0, 40.0
        mask_before = render_silhouette_for_eval(glb, yaw, pitch, r, fov, 512)
        dice_before = compute_dice(mask_before, ref_mask)
        print(f"  Dice BEFORE: {dice_before:.4f}", flush=True)

        # Apply silhouette correction
        print(f"  Applying silhouette correction...", flush=True)
        t1 = time.time()
        glb_corrected = corrector.correct(
            glb, processed,
            yaw=yaw, pitch=pitch, r=r, fov=fov,
            num_steps=80,
            verbose=True,
        )
        correct_time = time.time() - t1

        # Measure Dice AFTER correction
        mask_after = render_silhouette_for_eval(glb_corrected, yaw, pitch, r, fov, 512)
        dice_after = compute_dice(mask_after, ref_mask)
        print(f"  Dice AFTER: {dice_after:.4f}", flush=True)
        print(f"  Correction time: {correct_time:.1f}s", flush=True)

        diff = dice_after - dice_before
        # Convert to approximate A1 score (scale-invariant Dice, thresholded)
        a1_before = min(100, max(0, dice_before * 100))
        a1_after = min(100, max(0, dice_after * 100))

        print(f"\n  Result: Dice {dice_before:.4f} -> {dice_after:.4f} ({diff:+.4f})")
        print(f"  Est. A1: {a1_before:.1f} -> {a1_after:.1f} ({(a1_after-a1_before):+.1f})")

        # Save corrected GLB for inspection
        out_path = f'/tmp/silcorr_{img_name.split(".")[0]}.glb'
        glb_corrected.export(out_path)
        print(f"  Saved: {out_path}")

        results.append({
            'image': img_name,
            'dice_before': dice_before,
            'dice_after': dice_after,
            'diff': diff,
            'correction_time': correct_time,
        })

        del outputs, raw_mesh, glb, glb_corrected
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Silhouette Correction Impact")
    print(f"{'='*60}")

    diffs = [r['diff'] for r in results]
    print(f"Average Dice improvement: {np.mean(diffs):+.4f}")

    for r in results:
        short = r['image'][:20]
        print(f"  {short:22s}: {r['dice_before']:.4f} -> {r['dice_after']:.4f} "
              f"({r['diff']:+.4f}, {r['correction_time']:.1f}s)")


if __name__ == '__main__':
    main()
