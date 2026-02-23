"""
Render-and-compare texture refinement via differentiable rasterization.

Uses nvdiffrast to render the GLB mesh from the input camera viewpoint,
then optimizes the UV texture map using chrominance L1 + LPIPS + TV loss.
Based on GTR paper approach: ~50 iterations, ~10 seconds, +1.12 dB PSNR.

Key design: compares in chrominance (YCbCr Cb,Cr channels) not raw RGB,
because the reference image has real-world lighting while renders are unlit
base color. Chrominance is largely lighting-invariant.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, List
import trimesh


def intrinsics_to_projection(intrinsics: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """OpenCV intrinsics to OpenGL perspective matrix."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = (far + near) / (far - near)
    ret[2, 3] = 2 * near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


def total_variation_loss(texture: torch.Tensor) -> torch.Tensor:
    """Compute total variation of a texture [H, W, C]."""
    dx = texture[1:, :, :] - texture[:-1, :, :]
    dy = texture[:, 1:, :] - texture[:, :-1, :]
    return (dx.abs().mean() + dy.abs().mean()) * 0.5


def laplacian_hf_loss(rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Match Laplacian (high-frequency) energy between rendered and target.

    Encourages the rendered image to have the same amount of fine detail
    as the reference image, directly targeting the C3 Laplacian energy metric.
    """
    # Laplacian kernel
    kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=rendered.dtype, device=rendered.device
    ).view(1, 1, 3, 3)

    # Convert to grayscale [1, 1, H, W]
    rend_gray = (0.299 * rendered[..., 0] + 0.587 * rendered[..., 1] + 0.114 * rendered[..., 2])
    tgt_gray = (0.299 * target[..., 0] + 0.587 * target[..., 1] + 0.114 * target[..., 2])
    mask_2d = mask[..., 0]

    rend_gray = rend_gray.unsqueeze(0).unsqueeze(0)
    tgt_gray = tgt_gray.unsqueeze(0).unsqueeze(0)
    mask_2d = mask_2d.unsqueeze(0).unsqueeze(0)

    rend_lap = F.conv2d(rend_gray, kernel, padding=1)
    tgt_lap = F.conv2d(tgt_gray, kernel, padding=1)

    # Match Laplacian energy in masked region
    diff = (rend_lap - tgt_lap) * mask_2d
    return (diff ** 2).mean()


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [H, W, 3] in [0,1] to YCbCr [H, W, 3].

    Y  = 0.299*R + 0.587*G + 0.114*B  (luminance)
    Cb = 0.564*(B - Y) + 0.5           (blue chroma, shifted to [0,1])
    Cr = 0.713*(R - Y) + 0.5           (red chroma, shifted to [0,1])
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y) + 0.5
    cr = 0.713 * (r - y) + 0.5
    return torch.stack([y, cb, cr], dim=-1)


class TextureRefiner:
    """Post-GLB texture refinement via render-and-compare.

    Takes a trimesh GLB, renders it differentiably with nvdiffrast,
    and optimizes the base color texture to match the input reference image.

    Uses chrominance-only L1 (YCbCr Cb,Cr) instead of raw RGB L1 to handle
    the lighting mismatch between unlit renders and lit reference photos.

    Args:
        device: CUDA device string.
    """

    def __init__(self, device: str = 'cuda'):
        import nvdiffrast.torch as dr
        self.device = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self._lpips_fn = None

    def _get_lpips(self):
        if self._lpips_fn is None:
            import lpips
            self._lpips_fn = lpips.LPIPS(net='vgg').to(self.device).eval()
        return self._lpips_fn

    def _extract_mesh_data(self, mesh: trimesh.Trimesh) -> dict:
        """Extract vertices, faces, UVs, and texture from a trimesh object."""
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(mesh.faces, dtype=torch.int32, device=self.device)

        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=self.device)
        else:
            raise ValueError("Mesh has no UV coordinates — cannot refine texture")

        material = mesh.visual.material
        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            tex_img = material.baseColorTexture
        elif hasattr(material, 'image') and material.image is not None:
            tex_img = material.image
        else:
            raise ValueError("Mesh has no base color texture — cannot refine")

        if isinstance(tex_img, Image.Image):
            tex_np = np.array(tex_img.convert('RGBA')).astype(np.float32) / 255.0
        else:
            tex_np = np.array(tex_img).astype(np.float32) / 255.0

        texture = torch.tensor(tex_np, dtype=torch.float32, device=self.device)

        return {
            'vertices': vertices,
            'faces': faces,
            'uvs': uvs,
            'texture': texture,  # [H, W, 4] RGBA in [0,1]
        }

    def _setup_camera(
        self,
        yaw: float = 0.0,
        pitch: float = 0.25,
        r: float = 2.0,
        fov: float = 40.0,
        resolution: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Set up camera extrinsics and projection matrix.

        The GLB mesh has Y-up coordinate system (postprocess applies Y/Z swap:
        (x,y,z) -> (x,z,-y)). Camera must match this convention.

        Returns:
            (extrinsics, perspective, intrinsics)
        """
        import utils3d

        fov_rad = torch.deg2rad(torch.tensor(float(fov))).to(self.device)
        yaw_t = torch.tensor(float(yaw)).to(self.device)
        pitch_t = torch.tensor(float(pitch)).to(self.device)

        # render_utils Z-up convention:
        #   orig = (sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch)) * r
        # GLB Y-up convention (postprocess Y/Z swap: x,z,-y):
        orig_x = torch.sin(yaw_t) * torch.cos(pitch_t)
        orig_y = torch.cos(yaw_t) * torch.cos(pitch_t)
        orig_z = torch.sin(pitch_t)
        orig = torch.tensor([
            orig_x,       # X stays
            orig_z,       # Z -> Y (up)
            -orig_y,      # -Y -> Z (backward)
        ], device=self.device) * r

        extr = utils3d.torch.extrinsics_look_at(
            orig,
            torch.zeros(3, device=self.device),
            torch.tensor([0, 1, 0], dtype=torch.float32, device=self.device)
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov_rad, fov_rad)

        near, far = 0.1, 100.0
        perspective = intrinsics_to_projection(intr, near, far)

        return extr, perspective, intr

    def _render(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        uvs: torch.Tensor,
        texture: torch.Tensor,
        extrinsics: torch.Tensor,
        perspective: torch.Tensor,
        resolution: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable render using nvdiffrast.

        Returns:
            (rendered_image [H, W, 3], mask [H, W, 1])
        """
        import nvdiffrast.torch as dr

        verts_homo = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
        full_proj = perspective @ extrinsics
        verts_clip = (verts_homo @ full_proj.T).unsqueeze(0)  # [1, V, 4]

        rast, rast_db = dr.rasterize(
            self.glctx, verts_clip, faces, (resolution, resolution)
        )

        texc, texd = dr.interpolate(
            uvs.unsqueeze(0), rast, faces, rast_db=rast_db, diff_attrs='all'
        )

        color = dr.texture(
            texture[:, :, :3].unsqueeze(0).contiguous(),
            texc,
            texd,
            filter_mode='linear-mipmap-linear',
            boundary_mode='clamp',
        )[0]  # [H, W, 3]

        mask = (rast[0, :, :, 3:4] > 0).float()
        rendered = color * mask

        return rendered, mask

    def _prepare_reference(
        self,
        reference_image,
        resolution: int = 512,
    ) -> torch.Tensor:
        """Prepare reference image as tensor [H, W, 3] in [0,1]."""
        if isinstance(reference_image, Image.Image):
            if reference_image.mode == 'RGBA':
                bg = Image.new('RGB', reference_image.size, (0, 0, 0))
                bg.paste(reference_image, mask=reference_image.split()[3])
                reference_image = bg
            reference_image = reference_image.convert('RGB')
            reference_image = reference_image.resize((resolution, resolution), Image.LANCZOS)
            arr = np.array(reference_image).astype(np.float32) / 255.0
        elif isinstance(reference_image, np.ndarray):
            arr = reference_image.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            pil = pil.resize((resolution, resolution), Image.LANCZOS)
            arr = np.array(pil).astype(np.float32) / 255.0
        else:
            arr = reference_image

        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def _compute_losses(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        texture_rgb: torch.Tensor,
        lpips_fn,
        chroma_weight: float,
        lpips_weight: float,
        tv_weight: float,
        sat_weight: float,
        texture_orig_rgb: torch.Tensor,
        proximity_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss with chrominance L1, LPIPS, TV, proximity, and saturation.

        Returns:
            (total_loss, {name: value} dict for logging)
        """
        losses = {}

        # Chrominance L1: compare only Cb, Cr (lighting-invariant color)
        if chroma_weight > 0:
            rend_ycbcr = rgb_to_ycbcr(rendered)
            tgt_ycbcr = rgb_to_ycbcr(target)
            # Compare Cb and Cr channels only (index 1, 2), skip Y (luminance)
            chroma_loss = F.l1_loss(
                rend_ycbcr[..., 1:] * mask,
                tgt_ycbcr[..., 1:] * mask
            )
            losses['chroma'] = chroma_loss.item()
        else:
            chroma_loss = torch.tensor(0.0, device=rendered.device)

        # LPIPS perceptual loss
        if lpips_fn is not None and lpips_weight > 0:
            rendered_lpips = (rendered.permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)
            target_lpips = (target.permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)
            perceptual_loss = lpips_fn(rendered_lpips, target_lpips).mean()
            losses['lpips'] = perceptual_loss.item()
        else:
            perceptual_loss = torch.tensor(0.0, device=rendered.device)

        # Total variation (WARNING: smooths texture detail, use sparingly or set to 0)
        if tv_weight > 0:
            tv_loss = total_variation_loss(texture_rgb)
            losses['tv'] = tv_loss.item()
        else:
            tv_loss = torch.tensor(0.0, device=rendered.device)

        # Proximity to original: prevent over-modification of texture
        # This replaces TV as the regularizer — preserves original detail while
        # allowing color corrections only where the render-and-compare loss demands it.
        if proximity_weight > 0 and texture_orig_rgb is not None:
            prox_loss = F.l1_loss(texture_rgb, texture_orig_rgb)
            losses['prox'] = prox_loss.item()
        else:
            prox_loss = torch.tensor(0.0, device=rendered.device)

        # Saturation preservation: penalize desaturation relative to original
        if sat_weight > 0 and texture_orig_rgb is not None:
            orig_ycbcr = rgb_to_ycbcr(texture_orig_rgb)
            curr_ycbcr = rgb_to_ycbcr(texture_rgb)
            orig_sat = torch.sqrt(
                (orig_ycbcr[..., 1] - 0.5)**2 + (orig_ycbcr[..., 2] - 0.5)**2 + 1e-8
            )
            curr_sat = torch.sqrt(
                (curr_ycbcr[..., 1] - 0.5)**2 + (curr_ycbcr[..., 2] - 0.5)**2 + 1e-8
            )
            sat_deficit = F.relu(orig_sat - curr_sat)
            sat_loss = sat_deficit.mean()
            losses['sat'] = sat_loss.item()
        else:
            sat_loss = torch.tensor(0.0, device=rendered.device)

        total = (chroma_weight * chroma_loss
                 + lpips_weight * perceptual_loss
                 + tv_weight * tv_loss
                 + proximity_weight * prox_loss
                 + sat_weight * sat_loss)
        losses['total'] = total.item()

        return total, losses

    def refine(
        self,
        glb_mesh: trimesh.Trimesh,
        reference_image,
        num_iters: int = 20,
        lr: float = 0.002,
        chroma_weight: float = 1.0,
        lpips_weight: float = 0.3,
        tv_weight: float = 0.0,
        proximity_weight: float = 0.5,
        sat_weight: float = 0.1,
        hf_weight: float = 0.3,
        render_resolution: int = 768,
        camera_yaw: float = 0.0,
        camera_pitch: float = 0.25,
        camera_r: float = 2.0,
        camera_fov: float = 40.0,
        verbose: bool = True,
    ) -> trimesh.Trimesh:
        """Refine the GLB texture via render-and-compare optimization.

        Args:
            glb_mesh: Input trimesh with UV-mapped PBR texture.
            reference_image: Reference input image (PIL RGBA or numpy).
            num_iters: Optimization iterations (20 recommended; 50 causes over-smoothing).
            lr: Learning rate for Adam (0.002 recommended; 0.005 over-modifies).
            chroma_weight: Weight for chrominance L1 loss (YCbCr Cb,Cr only).
            lpips_weight: Weight for LPIPS perceptual loss.
            tv_weight: TV regularization (0.0 recommended — TV smooths detail, kills C3).
            proximity_weight: Penalize deviation from original texture (preserves detail).
            sat_weight: Weight for saturation preservation.
            hf_weight: Weight for Laplacian high-frequency matching loss (targets C3 metric).
            render_resolution: Resolution for differentiable rendering.
            camera_yaw/pitch/r/fov: Camera parameters matching input viewpoint.
            verbose: Print progress.

        Returns:
            Updated trimesh with refined texture.
        """
        if verbose:
            print(f"[TextureRefiner] Starting refinement ({num_iters} iters, lr={lr}, "
                  f"tv={tv_weight}, prox={proximity_weight})")

        mesh_data = self._extract_mesh_data(glb_mesh)
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        uvs = mesh_data['uvs']
        texture_orig = mesh_data['texture']  # [H, W, 4] RGBA

        texture_rgb = texture_orig[:, :, :3].clone().detach().requires_grad_(True)
        texture_alpha = texture_orig[:, :, 3:4].clone().detach()
        texture_orig_rgb = texture_orig[:, :, :3].clone().detach()

        extr, perspective, _ = self._setup_camera(
            yaw=camera_yaw, pitch=camera_pitch,
            r=camera_r, fov=camera_fov, resolution=render_resolution
        )

        target = self._prepare_reference(reference_image, render_resolution)

        optimizer = torch.optim.Adam([texture_rgb], lr=lr)
        lpips_fn = self._get_lpips() if lpips_weight > 0 else None

        best_loss = float('inf')
        best_texture = texture_rgb.data.clone()

        for step in range(num_iters):
            optimizer.zero_grad()

            rendered, mask = self._render(
                vertices, faces, uvs,
                texture_rgb, extr, perspective, render_resolution
            )

            loss, loss_dict = self._compute_losses(
                rendered, target, mask, texture_rgb,
                lpips_fn, chroma_weight, lpips_weight, tv_weight, sat_weight,
                texture_orig_rgb, proximity_weight=proximity_weight,
            )

            # Laplacian high-frequency matching: encourages texture detail
            if hf_weight > 0:
                hf_loss = laplacian_hf_loss(rendered, target, mask)
                loss = loss + hf_weight * hf_loss
                loss_dict['hf'] = hf_loss.item()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                texture_rgb.data.clamp_(0, 1)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_texture = texture_rgb.data.clone()

            if verbose and (step % 10 == 0 or step == num_iters - 1):
                parts = ' '.join(f'{k}={v:.4f}' for k, v in loss_dict.items()
                                 if k != 'total')
                print(f"  Step {step+1}/{num_iters}: loss={loss.item():.4f} ({parts})")

        refined_texture = torch.cat([best_texture, texture_alpha], dim=-1)
        refined_np = (refined_texture.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        refined_pil = Image.fromarray(refined_np, mode='RGBA')

        material = glb_mesh.visual.material
        if hasattr(material, 'baseColorTexture'):
            material.baseColorTexture = refined_pil
        elif hasattr(material, 'image'):
            material.image = refined_pil

        glb_mesh.visual = trimesh.visual.TextureVisuals(
            uv=glb_mesh.visual.uv,
            material=material,
        )

        if verbose:
            print(f"[TextureRefiner] Done. Best loss: {best_loss:.4f}")

        return glb_mesh

    def refine_multiview(
        self,
        glb_mesh: trimesh.Trimesh,
        reference_images: list,
        camera_params: list,
        num_iters: int = 50,
        lr: float = 0.005,
        chroma_weight: float = 1.0,
        lpips_weight: float = 0.3,
        tv_weight: float = 0.01,
        sat_weight: float = 0.1,
        render_resolution: int = 512,
        verbose: bool = True,
    ) -> trimesh.Trimesh:
        """Multi-view texture refinement.

        Optimizes texture to match multiple reference views simultaneously.
        """
        if verbose:
            print(f"[TextureRefiner] Multi-view refinement ({len(reference_images)} views, "
                  f"{num_iters} iters)")

        mesh_data = self._extract_mesh_data(glb_mesh)
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        uvs = mesh_data['uvs']
        texture_orig = mesh_data['texture']

        texture_rgb = texture_orig[:, :, :3].clone().detach().requires_grad_(True)
        texture_alpha = texture_orig[:, :, 3:4].clone().detach()
        texture_orig_rgb = texture_orig[:, :, :3].clone().detach()

        cameras = []
        targets = []
        for params, ref_img in zip(camera_params, reference_images):
            extr, perspective, _ = self._setup_camera(
                yaw=params.get('yaw', 0.0),
                pitch=params.get('pitch', 0.25),
                r=2.0, fov=40.0, resolution=render_resolution
            )
            cameras.append((extr, perspective))
            targets.append(self._prepare_reference(ref_img, render_resolution))

        optimizer = torch.optim.Adam([texture_rgb], lr=lr)
        lpips_fn = self._get_lpips() if lpips_weight > 0 else None

        best_loss = float('inf')
        best_texture = texture_rgb.data.clone()

        for step in range(num_iters):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for (extr, perspective), target in zip(cameras, targets):
                rendered, mask = self._render(
                    vertices, faces, uvs,
                    texture_rgb, extr, perspective, render_resolution
                )

                view_loss, _ = self._compute_losses(
                    rendered, target, mask, texture_rgb,
                    lpips_fn, chroma_weight, lpips_weight, tv_weight=0,
                    sat_weight=0, texture_orig_rgb=None,
                )
                total_loss = total_loss + view_loss

            total_loss = total_loss / len(cameras)

            # Add TV and saturation only once (not per-view)
            tv_loss = total_variation_loss(texture_rgb)
            total_loss = total_loss + tv_weight * tv_loss

            if sat_weight > 0:
                orig_ycbcr = rgb_to_ycbcr(texture_orig_rgb)
                curr_ycbcr = rgb_to_ycbcr(texture_rgb)
                orig_sat = torch.sqrt(
                    (orig_ycbcr[..., 1] - 0.5)**2 + (orig_ycbcr[..., 2] - 0.5)**2 + 1e-8
                )
                curr_sat = torch.sqrt(
                    (curr_ycbcr[..., 1] - 0.5)**2 + (curr_ycbcr[..., 2] - 0.5)**2 + 1e-8
                )
                sat_loss = F.relu(orig_sat - curr_sat).mean()
                total_loss = total_loss + sat_weight * sat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                texture_rgb.data.clamp_(0, 1)

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_texture = texture_rgb.data.clone()

            if verbose and (step % 10 == 0 or step == num_iters - 1):
                print(f"  Step {step+1}/{num_iters}: loss={total_loss.item():.4f}")

        refined_texture = torch.cat([best_texture, texture_alpha], dim=-1)
        refined_np = (refined_texture.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        refined_pil = Image.fromarray(refined_np, mode='RGBA')

        material = glb_mesh.visual.material
        if hasattr(material, 'baseColorTexture'):
            material.baseColorTexture = refined_pil
        elif hasattr(material, 'image'):
            material.image = refined_pil

        glb_mesh.visual = trimesh.visual.TextureVisuals(
            uv=glb_mesh.visual.uv,
            material=material,
        )

        if verbose:
            print(f"[TextureRefiner] Done. Best loss: {best_loss:.4f}")

        return glb_mesh
