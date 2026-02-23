"""
Post-GLB color correction via histogram matching in chrominance space.

Renders the mesh from the reference camera viewpoint, computes the chrominance
histogram mismatch between the rendered and reference images, and applies a
correction to the UV texture atlas. Works in LAB color space, only modifying
a* and b* channels (chrominance) to preserve the texture's luminance/detail.

Directly targets the V4 A2 metric (HSV histogram + LAB ΔE + mean color proximity).
"""

import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple
import trimesh


def _histogram_specification(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match source histogram to target histogram via CDF inversion.

    Both source and target are 1D arrays of values (e.g. a* channel pixels).
    Returns corrected source values that follow target's distribution.
    """
    # Compute CDFs
    src_sorted = np.sort(source)
    tgt_sorted = np.sort(target)

    # Build CDF lookup: for each source quantile, find target quantile
    n_src = len(src_sorted)
    n_tgt = len(tgt_sorted)

    # Map source values to their quantile positions [0, 1]
    src_ranks = np.searchsorted(src_sorted, source, side='left')
    src_quantiles = src_ranks.astype(np.float64) / n_src

    # Map quantile positions to target values
    tgt_indices = (src_quantiles * (n_tgt - 1)).astype(np.int64)
    tgt_indices = np.clip(tgt_indices, 0, n_tgt - 1)
    corrected = tgt_sorted[tgt_indices]

    return corrected


def _extract_foreground_pixels(image_rgba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract foreground pixels from RGBA image.

    Returns:
        pixels_rgb: [N, 3] uint8 RGB values
        mask: [H, W] bool
    """
    if image_rgba.shape[2] == 4:
        mask = image_rgba[:, :, 3] > 128
    else:
        # No alpha: assume all foreground
        mask = np.ones(image_rgba.shape[:2], dtype=bool)

    pixels_rgb = image_rgba[:, :, :3][mask]
    return pixels_rgb, mask


def compute_color_correction(
    rendered_rgb: np.ndarray,
    rendered_alpha: np.ndarray,
    reference_rgba: np.ndarray,
    blend_strength: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute LAB chrominance correction from rendered vs reference.

    Args:
        rendered_rgb: [H, W, 3] uint8 RGB rendered front view
        rendered_alpha: [H, W] uint8 alpha mask (>128 = foreground)
        reference_rgba: [H, W, 3 or 4] uint8 RGBA reference image
        blend_strength: How strongly to apply correction (0=none, 1=full)

    Returns:
        delta_a: Mean a* shift to apply
        delta_b: Mean b* shift to apply
    """
    rend_mask = rendered_alpha > 128
    if reference_rgba.shape[2] == 4:
        ref_mask = reference_rgba[:, :, 3] > 128
    else:
        ref_mask = np.ones(reference_rgba.shape[:2], dtype=bool)

    # Resize to match if needed
    ref_h, ref_w = reference_rgba.shape[:2]
    rend_h, rend_w = rendered_rgb.shape[:2]
    if (rend_h, rend_w) != (ref_h, ref_w):
        rendered_rgb = cv2.resize(rendered_rgb, (ref_w, ref_h))
        rendered_alpha = cv2.resize(rendered_alpha, (ref_w, ref_h))
        rend_mask = rendered_alpha > 128

    if rend_mask.sum() < 100 or ref_mask.sum() < 100:
        return np.float64(0), np.float64(0)

    # Convert to LAB
    rend_lab = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    ref_lab = cv2.cvtColor(reference_rgba[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float64)

    # Compute mean chrominance in foreground
    rend_a = rend_lab[:, :, 1][rend_mask].mean()
    rend_b = rend_lab[:, :, 2][rend_mask].mean()
    ref_a = ref_lab[:, :, 1][ref_mask].mean()
    ref_b = ref_lab[:, :, 2][ref_mask].mean()

    delta_a = (ref_a - rend_a) * blend_strength
    delta_b = (ref_b - rend_b) * blend_strength

    return delta_a, delta_b


def apply_color_transfer_to_texture(
    glb_mesh: trimesh.Trimesh,
    reference_image,
    rendered_rgb: Optional[np.ndarray] = None,
    rendered_alpha: Optional[np.ndarray] = None,
    mode: str = 'mean_shift',
    blend_strength: float = 0.7,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """Apply color correction to GLB texture atlas.

    Two modes:
    - 'mean_shift': Shift a*/b* channels by mean chrominance difference.
      Fast, safe, minimal artifacts. Best for systematic color drift.
    - 'histogram': Full histogram specification per LAB channel (a*, b*).
      More aggressive, matches distributions exactly. Better for large mismatches.

    Args:
        glb_mesh: trimesh mesh with UV texture
        reference_image: PIL Image (RGBA) or numpy array
        rendered_rgb: Pre-rendered front view [H, W, 3] uint8 (optional)
        rendered_alpha: Pre-rendered alpha [H, W] uint8 (optional)
        mode: 'mean_shift' or 'histogram'
        blend_strength: Blending factor (0=none, 1=full match)
        verbose: Print diagnostics

    Returns:
        Modified glb_mesh with corrected texture
    """
    # Get reference as numpy RGBA
    if isinstance(reference_image, Image.Image):
        ref_rgba = np.array(reference_image.convert('RGBA'))
    else:
        ref_rgba = reference_image

    # Get current texture
    material = glb_mesh.visual.material
    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
        texture_pil = material.baseColorTexture
    elif hasattr(material, 'image') and material.image is not None:
        texture_pil = material.image
    else:
        if verbose:
            print("[ColorTransfer] No texture found, skipping", flush=True)
        return glb_mesh

    texture_np = np.array(texture_pil)
    has_alpha = texture_np.shape[2] == 4
    tex_rgb = texture_np[:, :, :3]
    tex_alpha = texture_np[:, :, 3] if has_alpha else None

    # Convert texture to LAB
    tex_lab = cv2.cvtColor(tex_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)

    # Create texture mask (non-black, non-grey pixels)
    tex_mask = (tex_rgb.max(axis=2) > 10) & (tex_rgb.std(axis=2) > 3 if tex_rgb.shape[2] == 3 else True)

    if mode == 'mean_shift':
        if rendered_rgb is None or rendered_alpha is None:
            if verbose:
                print("[ColorTransfer] mean_shift requires pre-rendered views, skipping", flush=True)
            return glb_mesh

        delta_a, delta_b = compute_color_correction(
            rendered_rgb, rendered_alpha, ref_rgba, blend_strength
        )

        if verbose:
            print(f"[ColorTransfer] mean_shift: delta_a={delta_a:.1f}, delta_b={delta_b:.1f}", flush=True)

        if abs(delta_a) < 0.5 and abs(delta_b) < 0.5:
            if verbose:
                print("[ColorTransfer] Correction too small, skipping", flush=True)
            return glb_mesh

        # Apply shift to a* and b* channels
        tex_lab[:, :, 1] = np.clip(tex_lab[:, :, 1] + delta_a, 0, 255)
        tex_lab[:, :, 2] = np.clip(tex_lab[:, :, 2] + delta_b, 0, 255)

    elif mode == 'histogram':
        # Histogram specification on a* and b* channels
        ref_rgb, ref_mask = _extract_foreground_pixels(ref_rgba)
        if len(ref_rgb) < 100:
            if verbose:
                print("[ColorTransfer] Reference has too few foreground pixels", flush=True)
            return glb_mesh

        ref_lab = cv2.cvtColor(ref_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)

        tex_fg_indices = np.where(tex_mask.ravel())[0]
        if len(tex_fg_indices) < 100:
            if verbose:
                print("[ColorTransfer] Texture has too few valid pixels", flush=True)
            return glb_mesh

        tex_fg_lab = tex_lab.reshape(-1, 3)[tex_fg_indices]

        # Match a* and b* channels
        for ch in [1, 2]:
            ch_name = 'a*' if ch == 1 else 'b*'
            source = tex_fg_lab[:, ch]
            target = ref_lab[:, ch]

            corrected = _histogram_specification(source, target)
            # Blend with original
            blended = source * (1 - blend_strength) + corrected * blend_strength
            tex_fg_lab[:, ch] = blended

            if verbose:
                delta = (blended - source).mean()
                print(f"[ColorTransfer] histogram {ch_name}: mean shift={delta:.1f}", flush=True)

        # Write back
        flat_lab = tex_lab.reshape(-1, 3)
        flat_lab[tex_fg_indices] = tex_fg_lab
        tex_lab = flat_lab.reshape(tex_lab.shape)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Convert back to RGB
    tex_lab_u8 = np.clip(tex_lab, 0, 255).astype(np.uint8)
    tex_rgb_corrected = cv2.cvtColor(tex_lab_u8, cv2.COLOR_LAB2RGB)

    # Reassemble with alpha
    if has_alpha:
        tex_corrected = np.concatenate([tex_rgb_corrected, tex_alpha[:, :, None]], axis=2)
        corrected_pil = Image.fromarray(tex_corrected, mode='RGBA')
    else:
        corrected_pil = Image.fromarray(tex_rgb_corrected, mode='RGB')

    # Apply to material
    if hasattr(material, 'baseColorTexture'):
        material.baseColorTexture = corrected_pil
    elif hasattr(material, 'image'):
        material.image = corrected_pil

    glb_mesh.visual = trimesh.visual.TextureVisuals(
        uv=glb_mesh.visual.uv,
        material=material,
    )

    if verbose:
        print(f"[ColorTransfer] Done ({mode}, strength={blend_strength})", flush=True)

    return glb_mesh
