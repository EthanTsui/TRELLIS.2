#!/usr/bin/env python3
"""Quick single-image SilhouetteCorrector test with full pipeline.

Generates 1 mesh from T.png, applies silhouette correction, measures Dice.
"""
import sys, os, time
sys.path.insert(0, '/workspace/TRELLIS.2')
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
from PIL import Image


def main():
    import gc

    print("=== Single-Image Silhouette Correction Test ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Load pipeline
    print("Loading pipeline...", flush=True)
    t0 = time.time()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    print(f"Pipeline loaded in {time.time()-t0:.1f}s", flush=True)

    # Load corrector
    from trellis2.postprocessing.silhouette_corrector import SilhouetteCorrector
    import o_voxel.postprocess
    import nvdiffrast.torch as dr

    corrector = SilhouetteCorrector(device='cuda')
    glctx = dr.RasterizeCudaContext(device='cuda')

    # Load image
    img_path = '/workspace/TRELLIS.2/assets/example_image/T.png'
    img = Image.open(img_path)
    processed = pipeline.preprocess_image(img)

    # Generate
    print("\nGenerating (seed=42, 1024 cascade)...", flush=True)
    t0 = time.time()
    ss_params = {"steps": 12, "guidance_strength": 10.0, "guidance_rescale": 0.8,
                 "rescale_t": 5.0, "multistep": True}
    shape_params = {"steps": 12, "guidance_strength": 10.0, "guidance_rescale": 0.5,
                    "rescale_t": 3.0, "multistep": True}
    tex_params = {"steps": 16, "guidance_strength": 12.0, "guidance_rescale": 1.0,
                  "rescale_t": 4.0, "heun_steps": 4, "multistep": True}

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
    print(f"Generation: {gen_time:.1f}s", flush=True)

    # Create GLB
    print("\nCreating GLB...", flush=True)
    t0 = time.time()
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
    glb_time = time.time() - t0
    print(f"GLB creation: {glb_time:.1f}s", flush=True)
    print(f"GLB: {len(glb.vertices)} verts, {len(glb.faces)} faces", flush=True)

    # Save original GLB
    glb.export('/tmp/silcorr_before.glb')

    # Free pipeline memory
    del outputs, raw_mesh, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    print("Pipeline memory freed", flush=True)

    # Render silhouette function (reusable)
    import utils3d

    def render_sil(mesh, yaw, pitch, r=2.0, fov=40.0, res=512):
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device='cuda')
        faces = torch.tensor(mesh.faces, dtype=torch.int32, device='cuda')

        fov_rad = torch.deg2rad(torch.tensor(float(fov))).to('cuda')
        yaw_t = torch.tensor(float(yaw)).to('cuda')
        pitch_t = torch.tensor(float(pitch)).to('cuda')

        orig_x = torch.sin(yaw_t) * torch.cos(pitch_t)
        orig_y = torch.cos(yaw_t) * torch.cos(pitch_t)
        orig_z = torch.sin(pitch_t)
        orig = torch.tensor([orig_x.item(), orig_z.item(), -orig_y.item()], device='cuda') * r

        extr = utils3d.torch.extrinsics_look_at(
            orig, torch.zeros(3, device='cuda'),
            torch.tensor([0, 1, 0], dtype=torch.float32, device='cuda')
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov_rad, fov_rad)

        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        proj = torch.zeros((4, 4), dtype=torch.float32, device='cuda')
        proj[0, 0] = 2 * fx
        proj[1, 1] = 2 * fy
        proj[0, 2] = 2 * cx - 1
        proj[1, 2] = -2 * cy + 1
        proj[2, 2] = 100.1 / 99.9
        proj[2, 3] = -20.0 / 99.9
        proj[3, 2] = 1.0

        verts_homo = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
        verts_clip = (verts_homo @ (proj @ extr).T).unsqueeze(0)
        rast, _ = dr.rasterize(glctx, verts_clip, faces, (res, res))
        return (rast[0, :, :, 3] > 0).float()

    # Prepare reference mask
    ref_resized = processed.convert('RGBA').resize((512, 512), Image.LANCZOS)
    ref_alpha = np.array(ref_resized)[:, :, 3]
    ref_mask = torch.tensor((ref_alpha > 128).astype(np.float32), device='cuda')

    # Measure Dice BEFORE at multiple views
    yaw, pitch = 0.0, 0.25
    mask_before = render_sil(glb, yaw, pitch)
    intersection = (mask_before * ref_mask).sum()
    dice_before = (2 * intersection / (mask_before.sum() + ref_mask.sum() + 1e-8)).item()
    print(f"\n--- Dice BEFORE correction: {dice_before:.4f} ---", flush=True)

    # Apply silhouette correction
    print("\nApplying silhouette correction (80 steps)...", flush=True)
    t0 = time.time()
    glb_corrected = corrector.correct(
        glb, processed,
        yaw=yaw, pitch=pitch, r=2.0, fov=40.0,
        num_steps=80,
        verbose=True,
    )
    corr_time = time.time() - t0
    print(f"Correction time: {corr_time:.1f}s", flush=True)

    # Measure Dice AFTER
    mask_after = render_sil(glb_corrected, yaw, pitch)
    intersection = (mask_after * ref_mask).sum()
    dice_after = (2 * intersection / (mask_after.sum() + ref_mask.sum() + 1e-8)).item()

    diff = dice_after - dice_before
    print(f"\n{'='*60}")
    print(f"RESULT: T.png Silhouette Correction")
    print(f"  Dice: {dice_before:.4f} -> {dice_after:.4f} ({diff:+.4f})")
    print(f"  Est. A1: {dice_before*100:.1f} -> {dice_after*100:.1f} ({diff*100:+.1f})")
    print(f"  Time: generation={gen_time:.1f}s, GLB={glb_time:.1f}s, correction={corr_time:.1f}s")

    # Save corrected
    glb_corrected.export('/tmp/silcorr_after.glb')
    print(f"  Saved: /tmp/silcorr_before.glb, /tmp/silcorr_after.glb")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
