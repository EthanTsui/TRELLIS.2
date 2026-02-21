import gradio as gr

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datetime import datetime
import shutil
import cv2
from typing import *
import torch
import numpy as np
from PIL import Image
import base64
import io
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
MODES = [
    {"name": "Normal", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "Clay render", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "Base color", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI forest", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8
DEFAULT_MODE = 3
DEFAULT_STEP = 3


css = """
/* Overwrite Gradio Default Style */
.stepper-wrapper {
    padding: 0;
}

.stepper-container {
    padding: 0;
    align-items: center;
}

.step-button {
    flex-direction: row;
}

.step-connector {
    transform: none;
}

.step-number {
    width: 16px;
    height: 16px;
}

.step-label {
    position: relative;
    bottom: 0;
}

.wrap.center.full {
    inset: 0;
    height: 100%;
}

.wrap.center.full.translucent {
    background: var(--block-background-fill);
}

.meta-text-center {
    display: block !important;
    position: absolute !important;
    top: unset !important;
    bottom: 0 !important;
    right: 0 !important;
    transform: unset !important;
}

/* Previewer */
.previewer-container {
    position: relative;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    width: 100%;
    height: 722px;
    margin: 0 auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.previewer-container .tips-icon {
    position: absolute;
    right: 10px;
    top: 10px;
    z-index: 10;
    border-radius: 10px;
    color: #fff;
    background-color: var(--color-accent);
    padding: 3px 6px;
    user-select: none;
}

.previewer-container .tips-text {
    position: absolute;
    right: 10px;
    top: 50px;
    color: #fff;
    background-color: var(--color-accent);
    border-radius: 10px;
    padding: 6px;
    text-align: left;
    max-width: 300px;
    z-index: 10;
    transition: all 0.3s;
    opacity: 0%;
    user-select: none;
}

.previewer-container .tips-text p {
    font-size: 14px;
    line-height: 1.2;
}

.tips-icon:hover + .tips-text { 
    display: block;
    opacity: 100%;
}

/* Row 1: Display Modes */
.previewer-container .mode-row {
    width: 100%;
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.previewer-container .mode-btn {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0.5;
    transition: all 0.2s;
    border: 2px solid #ddd;
    object-fit: cover;
}
.previewer-container .mode-btn:hover { opacity: 0.9; transform: scale(1.1); }
.previewer-container .mode-btn.active {
    opacity: 1;
    border-color: var(--color-accent);
    transform: scale(1.1);
}

/* Row 2: Display Image */
.previewer-container .display-row {
    margin-bottom: 20px;
    min-height: 400px;
    width: 100%;
    flex-grow: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}
.previewer-container .previewer-main-image {
    max-width: 100%;
    max-height: 100%;
    flex-grow: 1;
    object-fit: contain;
    display: none;
}
.previewer-container .previewer-main-image.visible {
    display: block;
}

/* Row 3: Custom HTML Slider */
.previewer-container .slider-row {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 0 10px;
}

.previewer-container input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    max-width: 400px;
    background: transparent;
}
.previewer-container input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 8px;
    cursor: pointer;
    background: #ddd;
    border-radius: 5px;
}
.previewer-container input[type=range]::-webkit-slider-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--color-accent);
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -6px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: transform 0.1s;
}
.previewer-container input[type=range]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
}

/* Overwrite Previewer Block Style */
.gradio-container .padded:has(.previewer-container) {
    padding: 0 !important;
}

.gradio-container:has(.previewer-container) [data-testid="block-label"] {
    position: absolute;
    top: 0;
    left: 0;
}
"""


head = """
<script>
    function refreshView(mode, step) {
        // 1. Find current mode and step
        const allImgs = document.querySelectorAll('.previewer-main-image');
        for (let i = 0; i < allImgs.length; i++) {
            const img = allImgs[i];
            if (img.classList.contains('visible')) {
                const id = img.id;
                const [_, m, s] = id.split('-');
                if (mode === -1) mode = parseInt(m.slice(1));
                if (step === -1) step = parseInt(s.slice(1));
                break;
            }
        }
        
        // 2. Hide ALL images
        // We select all elements with class 'previewer-main-image'
        allImgs.forEach(img => img.classList.remove('visible'));

        // 3. Construct the specific ID for the current state
        // Format: view-m{mode}-s{step}
        const targetId = 'view-m' + mode + '-s' + step;
        const targetImg = document.getElementById(targetId);

        // 4. Show ONLY the target
        if (targetImg) {
            targetImg.classList.add('visible');
        }

        // 5. Update Button Highlights
        const allBtns = document.querySelectorAll('.mode-btn');
        allBtns.forEach((btn, idx) => {
            if (idx === mode) btn.classList.add('active');
            else btn.classList.remove('active');
        });
    }
    
    // --- Action: Switch Mode ---
    function selectMode(mode) {
        refreshView(mode, -1);
    }
    
    // --- Action: Slider Change ---
    function onSliderChange(val) {
        refreshView(-1, parseInt(val));
    }
</script>
"""


empty_html = f"""
<div class="previewer-container">
    <svg style=" opacity: .5; height: var(--size-5); color: var(--body-text-color);"
    xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
</div>
"""


def image_to_base64(image):
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def pack_state(latents: Tuple[SparseTensor, SparseTensor, int],
               input_images: List = None, camera_params: List[dict] = None) -> dict:
    shape_slat, tex_slat, res = latents
    state = {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }
    if input_images is not None and camera_params is not None:
        state['input_images'] = [np.array(img) for img in input_images]
        state['camera_params'] = camera_params
    return state
    
    
def unpack_state(state: dict) -> Tuple[SparseTensor, SparseTensor, int]:
    shape_slat = SparseTensor(
        feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(state['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, state['res']


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


# Multi-view composite image layouts: (rows, cols, view_names)
MULTIVIEW_LAYOUTS = {
    "Single Image": None,
    "2x3 Grid (6 views)": (2, 3, ['front', 'right', 'back', 'left', 'top', 'bottom']),
    "3x2 Grid (6 views)": (3, 2, ['front', 'back', 'left', 'right', 'top', 'bottom']),
    "1x3 Strip (3 views)": (1, 3, ['front', 'left', 'right']),
    "1x3 Strip 120° (3 views)": (1, 3, ['front', 'left_120', 'right_120']),
    "1x3 Strip 45° (3 views)": (1, 3, ['front', 'front_left', 'front_right']),
    "2x2 Grid (4 views)": (2, 2, ['front', 'right', 'back', 'left']),
    "2x2 Grid diagonal (4 views)": (2, 2, ['front_left', 'front_right', 'back_left', 'back_right']),
    "Custom Angles": "custom",
}


def split_composite_image(image: Image.Image, layout: str, custom_angles: str = "") -> Dict[str, Image.Image]:
    """Split a composite multi-view image into individual view images."""
    spec = MULTIVIEW_LAYOUTS.get(layout)
    if spec is None:
        return {}
    if spec == "custom":
        return _split_custom_angles(image, custom_angles)
    rows, cols, view_names = spec
    w, h = image.size
    cell_w, cell_h = w // cols, h // rows
    views = {}
    for idx, name in enumerate(view_names):
        r, c = divmod(idx, cols)
        box = (c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h)
        views[name] = image.crop(box)
    return views


def _split_custom_angles(image: Image.Image, angles_str: str) -> Dict[str, Image.Image]:
    """Split composite image using custom yaw angles (degrees, comma-separated)."""
    if not angles_str or not angles_str.strip():
        return {}
    try:
        angles = [float(a.strip()) for a in angles_str.split(",") if a.strip()]
    except ValueError:
        return {}
    if len(angles) < 2:
        return {}
    n = len(angles)
    w, h = image.size
    cell_w = w // n
    views = {}
    for i, angle in enumerate(angles):
        box = (i * cell_w, 0, (i + 1) * cell_w, h)
        views[f"custom_{i}"] = image.crop(box)
    return views


def parse_custom_yaw_angles(angles_str: str) -> list:
    """Parse comma-separated yaw angles (degrees) into list of floats (radians)."""
    import math
    if not angles_str or not angles_str.strip():
        return []
    try:
        return [math.radians(float(a.strip())) for a in angles_str.split(",") if a.strip()]
    except ValueError:
        return []


def image_to_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
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
    tex_slat_cfg_mp_strength: float,
    tex_slat_heun_steps: int,
    multistep: bool,
    back_image: Optional[Image.Image],
    left_image: Optional[Image.Image],
    right_image: Optional[Image.Image],
    top_image: Optional[Image.Image],
    bottom_image: Optional[Image.Image],
    multiview_layout: str,
    multiview_mode: str,
    texture_multiview_mode: str,
    custom_angles: str,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> str:
    # Option 1: Auto-split composite image based on layout
    composite_views = split_composite_image(image, multiview_layout, custom_angles)

    if composite_views:
        run_input = {k: pipeline.preprocess_image(v) for k, v in composite_views.items()}
    else:
        # Option 2: Manual extra views from accordion
        extra_views = {
            'back': back_image, 'left': left_image, 'right': right_image,
            'top': top_image, 'bottom': bottom_image,
        }
        extra_views = {k: v for k, v in extra_views.items() if v is not None}

        if extra_views:
            extra_views = {k: pipeline.preprocess_image(v) for k, v in extra_views.items()}
            run_input = {'front': pipeline.preprocess_image(image), **extra_views}
        else:
            run_input = pipeline.preprocess_image(image)

    # --- Sampling ---
    outputs, latents = pipeline.run(
        run_input,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
            "multistep": multistep,
        },
        shape_slat_sampler_params={
            "steps": shape_slat_sampling_steps,
            "guidance_strength": shape_slat_guidance_strength,
            "guidance_rescale": shape_slat_guidance_rescale,
            "rescale_t": shape_slat_rescale_t,
            "multistep": multistep,
        },
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
            "cfg_mp_strength": tex_slat_cfg_mp_strength,
            "heun_steps": tex_slat_heun_steps,
            "multistep": multistep,
        },
        pipeline_type={
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution],
        return_latent=True,
        multiview_mode=multiview_mode,
        texture_multiview_mode=texture_multiview_mode,
        custom_yaw_angles=parse_custom_yaw_angles(custom_angles) if multiview_layout == "Custom Angles" else None,
    )
    mesh = outputs[0]
    mesh.simplify(16777216) # nvdiffrast limit
    images = render_utils.render_snapshot(mesh, resolution=1024, r=2, fov=36, nviews=STEPS, envmap=envmap)

    # Save multi-view images and camera params for texture projection in extract_glb
    mv_images = None
    mv_cam_params = None
    if isinstance(run_input, dict) and len(run_input) >= 2:
        mv_images = list(run_input.values())
        custom_yaws = parse_custom_yaw_angles(custom_angles) if multiview_layout == "Custom Angles" else None
        if custom_yaws and len(custom_yaws) == len(run_input):
            mv_cam_params = [{'yaw': y, 'pitch': 0.0} for y in custom_yaws]
        else:
            from trellis2.utils.visibility import get_camera_params_from_views
            mv_cam_params = get_camera_params_from_views(list(run_input.keys()))

    state = pack_state(latents, input_images=mv_images, camera_params=mv_cam_params)
    torch.cuda.empty_cache()
    
    # --- HTML Construction ---
    # The Stack of 48 Images
    images_html = ""
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
            # ID Naming Convention: view-m{mode}-s{step}
            unique_id = f"view-m{m_idx}-s{s_idx}"
            
            # Logic: Only Mode 0, Step 0 is visible initially
            is_visible = (m_idx == DEFAULT_MODE and s_idx == DEFAULT_STEP)
            vis_class = "visible" if is_visible else ""
            
            # Image Source
            img_base64 = image_to_base64(Image.fromarray(images[mode['render_key']][s_idx]))
            
            # Render the Tag
            images_html += f"""
                <img id="{unique_id}" 
                     class="previewer-main-image {vis_class}" 
                     src="{img_base64}" 
                     loading="eager">
            """
    
    # Button Row HTML
    btns_html = ""
    for idx, mode in enumerate(MODES):        
        active_class = "active" if idx == DEFAULT_MODE else ""
        # Note: onclick calls the JS function defined in Head
        btns_html += f"""
            <img src="{mode['icon_base64']}" 
                 class="mode-btn {active_class}" 
                 onclick="selectMode({idx})"
                 title="{mode['name']}">
        """
    
    # Assemble the full component
    full_html = f"""
    <div class="previewer-container">
        <div class="tips-wrapper">
            <div class="tips-icon">💡Tips</div>
            <div class="tips-text">
                <p>● <b>Render Mode</b> - Click on the circular buttons to switch between different render modes.</p>
                <p>● <b>View Angle</b> - Drag the slider to change the view angle.</p>
            </div>
        </div>
        
        <!-- Row 1: Viewport containing 48 static <img> tags -->
        <div class="display-row">
            {images_html}
        </div>
        
        <!-- Row 2 -->
        <div class="mode-row" id="btn-group">
            {btns_html}
        </div>

        <!-- Row 3: Slider -->
        <div class="slider-row">
            <input type="range" id="custom-slider" min="0" max="{STEPS - 1}" value="{DEFAULT_STEP}" step="1" oninput="onSliderChange(this.value)">
        </div>
    </div>
    """
    
    return state, full_html


def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        decimation_target (int): The target face count for decimation.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shape_slat, tex_slat, res = unpack_state(state)
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]

    # Retrieve multi-view images and camera params for texture projection
    mv_images = None
    mv_cam_params = None
    if 'input_images' in state and 'camera_params' in state:
        mv_images = [Image.fromarray(arr) for arr in state['input_images']]
        mv_cam_params = state['camera_params']

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0.9,
        max_metallic=0.05,
        min_roughness=0.4,
        enable_normal_map=False,
        enable_ao=False,
        enable_grey_recovery=True,
        use_tqdm=True,
        verbose=True,
    )
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS.2](https://microsoft.github.io/TRELLIS.2)
    * Upload an image (preferably with an alpha-masked foreground object) and click Generate to create a 3D asset.
    * Click Extract GLB to export and download the generated GLB file if you're satisfied with the result. Otherwise, try another time.
    """)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)
            
            resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1536")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            decimation_target = gr.Slider(100000, 2000000, label="Decimation Target", value=800000, step=10000)
            texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)

            multiview_layout = gr.Dropdown(
                choices=list(MULTIVIEW_LAYOUTS.keys()),
                value="Single Image",
                label="Multi-View Layout",
                info="If your image contains multiple views in a grid, select the layout to auto-split.",
            )

            custom_angles = gr.Textbox(
                label="Custom Yaw Angles (degrees)",
                placeholder="e.g. 0, 120, -120",
                info="For 'Custom Angles' layout: comma-separated yaw angles in degrees. 0=front, 90=left, -90=right, 180=back.",
                visible=True,
            )

            multiview_mode = gr.Radio(
                choices=["concat", "view_weighted", "multidiffusion", "stochastic"],
                value="concat",
                label="Multi-View Mode",
                info="Concat: cross-attn feature concatenation. View Weighted: per-view cross-attn with visibility merge (best separation). Multidiffusion: average K predictions. Stochastic: cycles views.",
            )

            texture_multiview_mode = gr.Radio(
                choices=["single", "concat", "view_weighted", "tapa", "multidiffusion"],
                value="single",
                label="Texture Multi-View Mode",
                info="Single: front-view only (safest). Concat: all views. View Weighted: per-view cross-attn with visibility merge. TAPA: concat then single. Multidiffusion: average.",
            )

            with gr.Accordion("Additional Views (Optional)", open=False):
                gr.Markdown("Upload extra views to improve quality. Main image = front view. Ignored when a multi-view layout is selected above.")
                with gr.Row():
                    back_image = gr.Image(label="Back", format="png", image_mode="RGBA", type="pil", height=200)
                    left_image = gr.Image(label="Left", format="png", image_mode="RGBA", type="pil", height=200)
                    right_image = gr.Image(label="Right", format="png", image_mode="RGBA", type="pil", height=200)
                with gr.Row():
                    top_image = gr.Image(label="Top", format="png", image_mode="RGBA", type="pil", height=200)
                    bottom_image = gr.Image(label="Bottom", format="png", image_mode="RGBA", type="pil", height=200)

            generate_btn = gr.Button("Generate")
                
            with gr.Accordion(label="Advanced Settings", open=False):                
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(1.0, 15.0, label="Guidance Strength", value=10.0, step=0.1)
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.8, step=0.01)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)
                gr.Markdown("Stage 2: Shape Generation")
                with gr.Row():
                    shape_slat_guidance_strength = gr.Slider(1.0, 15.0, label="Guidance Strength", value=10.0, step=0.1)
                    shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                    shape_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                gr.Markdown("Stage 3: Material Generation")
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(1.0, 15.0, label="Guidance Strength", value=12.0, step=0.1)
                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=1.0, step=0.01)
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=16, step=1)
                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=4.0, step=0.1)
                    tex_slat_cfg_mp_strength = gr.Slider(0.0, 0.5, label="CFG-MP Strength", value=0.15, step=0.01)
                    tex_slat_heun_steps = gr.Slider(0, 8, label="Heun Steps (final)", value=4, step=1)
                multistep = gr.Checkbox(label="AB2 Multistep (free 2nd-order accuracy)", value=True)

        with gr.Column(scale=10):
            with gr.Walkthrough(selected=0) as walkthrough:
                with gr.Step("Preview", id=0):
                    preview_output = gr.HTML(empty_html, label="3D Asset Preview", show_label=True, container=True)
                    extract_btn = gr.Button("Extract GLB")
                with gr.Step("Extract", id=1):
                    glb_output = gr.Model3D(label="Extracted GLB", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                    download_btn = gr.DownloadButton(label="Download GLB")
                    
        with gr.Column(scale=1, min_width=172):
            examples = gr.Examples(
                examples=[
                    f'assets/example_image/{image}'
                    for image in os.listdir("assets/example_image")
                ],
                inputs=[image_prompt],
                fn=preprocess_image,
                outputs=[image_prompt],
                run_on_click=True,
                examples_per_page=18,
            )
                    
    output_buf = gr.State()
    

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)
    
    # NOTE: No preprocessing on upload — all preprocessing happens in image_to_3d
    # This avoids the race condition where layout selection hasn't propagated yet

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        lambda: gr.Walkthrough(selected=0), outputs=walkthrough
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, seed, resolution,
            ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
            shape_slat_guidance_strength, shape_slat_guidance_rescale, shape_slat_sampling_steps, shape_slat_rescale_t,
            tex_slat_guidance_strength, tex_slat_guidance_rescale, tex_slat_sampling_steps, tex_slat_rescale_t, tex_slat_cfg_mp_strength, tex_slat_heun_steps,
            multistep,
            back_image, left_image, right_image, top_image, bottom_image,
            multiview_layout, multiview_mode, texture_multiview_mode, custom_angles,
        ],
        outputs=[output_buf, preview_output],
    )
    
    extract_btn.click(
        lambda: gr.Walkthrough(selected=1), outputs=walkthrough
    ).then(
        extract_glb,
        inputs=[output_buf, decimation_target, texture_size],
        outputs=[glb_output, download_btn],
    )
        

# Launch the Gradio app
if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)

    # Construct ui components
    btn_img_base64_strs = {}
    for i in range(len(MODES)):
        icon = Image.open(MODES[i]['icon'])
        MODES[i]['icon_base64'] = image_to_base64(icon)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()
    
    envmap = {
        'forest': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'sunset': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/sunset.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'courtyard': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/courtyard.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
    }

    import os
    demo.launch(
        css=css,
        head=head,
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860))
    )
