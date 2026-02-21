"""
Gradio app: Image -> 3D mesh -> Auto-rigging (UniRig).

Two-stage workflow:
  1. Upload image -> Generate 3D GLB via TRELLIS.2
  2. Click "Auto-Rig" -> Run UniRig skeleton + skin + merge -> Download rigged GLB

GPU memory strategy:
  - TRELLIS.2 generates on GPU, then moves to CPU + empty_cache
  - UniRig subprocess claims GPU for inference (~8GB bf16-mixed)
  - Subprocess exits -> GPU freed automatically
  - TRELLIS.2 pipeline moved back to GPU for next generation
"""

import gradio as gr

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import logging
import shutil
from datetime import datetime
from typing import *

import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.rigging_pipeline import RiggingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir, ignore_errors=True)


def preprocess_image(image: Image.Image) -> Image.Image:
    return pipeline.preprocess_image(image)


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    texture_size: int,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str]:
    """Generate a 3D GLB from an uploaded image."""
    import o_voxel

    outputs, latents = pipeline.run(
        image,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
        },
        shape_slat_sampler_params={
            "steps": shape_slat_sampling_steps,
            "guidance_strength": shape_slat_guidance_strength,
            "guidance_rescale": shape_slat_guidance_rescale,
            "rescale_t": shape_slat_rescale_t,
        },
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
        },
        pipeline_type={
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution],
        return_latent=True,
    )

    mesh = outputs[0]
    shape_slat, tex_slat, res = latents
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=500000,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f"sample_{timestamp}.glb")
    glb.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()

    return glb_path, glb_path


def auto_rig(
    glb_path: str,
    rig_seed: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str]:
    """Run UniRig auto-rigging on the generated GLB.

    Offloads the TRELLIS.2 pipeline to CPU first, then runs UniRig
    as a subprocess (which claims GPU), and restores TRELLIS.2 after.
    """
    if not glb_path or not os.path.isfile(glb_path):
        raise gr.Error("Please generate a 3D model first before rigging.")

    # Free GPU for UniRig
    logger.info("Moving TRELLIS.2 pipeline to CPU to free GPU memory...")
    pipeline.to("cpu")
    torch.cuda.empty_cache()

    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    rig_output_dir = os.path.join(user_dir, "rig_work")

    try:
        rigged_glb = rigging_pipeline.rig_mesh(
            glb_path=glb_path,
            output_dir=rig_output_dir,
            seed=rig_seed,
        )
    except Exception as e:
        logger.error("UniRig rigging failed: %s", e)
        raise gr.Error(f"Auto-rigging failed: {e}")
    finally:
        # Restore TRELLIS.2 pipeline to GPU
        logger.info("Restoring TRELLIS.2 pipeline to GPU...")
        pipeline.cuda()
        torch.cuda.empty_cache()

    return rigged_glb, rigged_glb


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to Rigged 3D Asset with [TRELLIS.2](https://microsoft.github.io/TRELLIS.2) + [UniRig](https://github.com/VAST-AI-Research/UniRig)
    1. Upload an image and click **Generate** to create a 3D mesh (GLB with PBR textures).
    2. Click **Auto-Rig** to add a skeleton and skin weights via UniRig.
    3. Download the rigged GLB for use in game engines or animation tools.
    """)

    with gr.Row():
        # --- Left column: controls ---
        with gr.Column(scale=1, min_width=360):
            image_prompt = gr.Image(
                label="Image Prompt", format="png", image_mode="RGBA",
                type="pil", height=400,
            )

            resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1024")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)

            generate_btn = gr.Button("Generate", variant="primary")

            with gr.Accordion(label="Advanced Generation Settings", open=False):
                gr.Markdown("Stage 1: Sparse Structure")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)
                gr.Markdown("Stage 2: Shape")
                with gr.Row():
                    shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                    shape_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                gr.Markdown("Stage 3: Material")
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)

            gr.Markdown("---")
            rig_seed = gr.Slider(0, MAX_SEED, label="Rigging Seed", value=12345, step=1)
            rig_btn = gr.Button("Auto-Rig (UniRig)", variant="secondary")

        # --- Right column: outputs ---
        with gr.Column(scale=10):
            glb_output = gr.Model3D(
                label="Generated GLB", height=500, show_label=True,
                display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0),
            )
            download_glb_btn = gr.DownloadButton(label="Download GLB")

            rigged_output = gr.Model3D(
                label="Rigged GLB", height=500, show_label=True,
                display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0),
            )
            download_rigged_btn = gr.DownloadButton(label="Download Rigged GLB")

    # Hidden state to pass GLB path from generate to rig step
    current_glb = gr.State(value=None)

    # --- Event handlers ---
    demo.load(start_session)
    demo.unload(end_session)

    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, seed, resolution, texture_size,
            ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
            shape_slat_guidance_strength, shape_slat_guidance_rescale, shape_slat_sampling_steps, shape_slat_rescale_t,
            tex_slat_guidance_strength, tex_slat_guidance_rescale, tex_slat_sampling_steps, tex_slat_rescale_t,
        ],
        outputs=[glb_output, current_glb],
    ).then(
        lambda p: p,
        inputs=[current_glb],
        outputs=[download_glb_btn],
    )

    rig_btn.click(
        auto_rig,
        inputs=[current_glb, rig_seed],
        outputs=[rigged_output, download_rigged_btn],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    rigging_pipeline = RiggingPipeline()

    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )
