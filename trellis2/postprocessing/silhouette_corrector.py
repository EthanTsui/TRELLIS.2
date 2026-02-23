"""
Post-GLB silhouette correction via differentiable rasterization.

Renders the mesh from the input camera viewpoint, compares the rendered
silhouette against the reference image alpha mask, and deforms vertex
positions to minimize silhouette mismatch.

Uses scale-invariant comparison: both masks are cropped to their bounding
boxes and centered in a padded frame before computing loss. The camera
distance is auto-calibrated to make the mesh fill the frame.

Uses nvdiffrast dr.antialias() for differentiable silhouette gradients,
BCE + distance transform loss, and Laplacian regularization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Tuple
import trimesh
import cv2


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


def _crop_to_bbox_tensor(mask: torch.Tensor, pad_frac: float = 0.1,
                         out_size: int = 256) -> torch.Tensor:
    """Crop mask to its bounding box with padding, resize to out_size.

    Non-differentiable — used for target preparation and Dice evaluation only.

    Args:
        mask: [H, W] float tensor (0 or 1)
        pad_frac: Padding as fraction of bounding box size
        out_size: Output square size

    Returns:
        [out_size, out_size] float tensor
    """
    coords = torch.nonzero(mask > 0.5, as_tuple=False)
    if len(coords) < 10:
        return torch.zeros(out_size, out_size, device=mask.device)

    y0, x0 = coords.min(dim=0).values
    y1, x1 = coords.max(dim=0).values
    h = (y1 - y0 + 1).item()
    w = (x1 - x0 + 1).item()
    side = max(h, w)
    pad = int(side * pad_frac)
    cy = ((y0 + y1) // 2).item()
    cx = ((x0 + x1) // 2).item()
    half = side // 2 + pad

    H, W = mask.shape
    sy = max(0, cy - half)
    ey = min(H, cy + half)
    sx = max(0, cx - half)
    ex = min(W, cx + half)

    crop = mask[sy:ey, sx:ex]
    if crop.numel() == 0:
        return torch.zeros(out_size, out_size, device=mask.device)

    # Resize using interpolate
    crop_4d = crop.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
    resized = F.interpolate(crop_4d, size=(out_size, out_size), mode='bilinear',
                            align_corners=False)
    return resized[0, 0]


class SilhouetteCorrector:
    """Post-GLB silhouette correction via differentiable rasterization.

    Uses scale-invariant comparison (crop-to-bbox) matching the V4
    evaluator's A1 methodology. Camera distance is auto-calibrated.
    """

    def __init__(self, device: str = 'cuda'):
        import nvdiffrast.torch as dr
        self.device = device
        self.glctx = dr.RasterizeCudaContext(device=device)

    def _extract_mesh_data(self, mesh: trimesh.Trimesh) -> dict:
        """Extract vertices and faces from a trimesh object."""
        vertices = torch.tensor(
            mesh.vertices, dtype=torch.float32, device=self.device
        )
        faces = torch.tensor(
            mesh.faces, dtype=torch.int32, device=self.device
        )
        return {'vertices': vertices, 'faces': faces}

    def _setup_camera(
        self,
        yaw: float = 0.0,
        pitch: float = 0.25,
        r: float = 2.0,
        fov: float = 40.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Set up camera extrinsics and projection matrix.

        GLB mesh has Y-up coordinate system (postprocess applies Y/Z swap).
        """
        import utils3d

        fov_rad = torch.deg2rad(torch.tensor(float(fov))).to(self.device)
        yaw_t = torch.tensor(float(yaw)).to(self.device)
        pitch_t = torch.tensor(float(pitch)).to(self.device)

        orig_x = torch.sin(yaw_t) * torch.cos(pitch_t)
        orig_y = torch.cos(yaw_t) * torch.cos(pitch_t)
        orig_z = torch.sin(pitch_t)
        orig = torch.tensor([
            orig_x.item(),
            orig_z.item(),
            -orig_y.item(),
        ], device=self.device) * r

        extr = utils3d.torch.extrinsics_look_at(
            orig,
            torch.zeros(3, device=self.device),
            torch.tensor([0, 1, 0], dtype=torch.float32, device=self.device)
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov_rad, fov_rad)
        perspective = intrinsics_to_projection(intr, near=0.1, far=100.0)

        return extr, perspective

    def _render_silhouette(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        extrinsics: torch.Tensor,
        perspective: torch.Tensor,
        resolution: int = 512,
    ) -> torch.Tensor:
        """Differentiable silhouette render using nvdiffrast.

        Returns:
            mask: [1, H, W, 1] soft antialiased mask
        """
        import nvdiffrast.torch as dr

        verts_homo = torch.cat(
            [vertices, torch.ones_like(vertices[:, :1])], dim=-1
        )
        full_proj = perspective @ extrinsics
        verts_clip = (verts_homo @ full_proj.T).unsqueeze(0)

        rast, _ = dr.rasterize(
            self.glctx, verts_clip, faces, (resolution, resolution)
        )

        mask = (rast[..., -1:] > 0).float()
        mask = dr.antialias(mask, rast, verts_clip, faces, pos_gradient_boost=1.0)

        return mask

    def _auto_calibrate_r(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        yaw: float,
        pitch: float,
        fov: float,
        target_coverage: float,
        resolution: int = 256,
    ) -> float:
        """Binary search for camera distance r that gives target screen coverage.

        Args:
            target_coverage: Desired fraction of pixels covered (0-1)

        Returns:
            Optimal camera distance r
        """
        r_low, r_high = 0.3, 5.0
        for _ in range(15):
            r_mid = (r_low + r_high) / 2
            extr, perspective = self._setup_camera(yaw, pitch, r_mid, fov)
            with torch.no_grad():
                mask = self._render_silhouette(
                    vertices, faces, extr, perspective, resolution
                )
                coverage = mask.sum().item() / (resolution * resolution)
            if coverage > target_coverage:
                r_low = r_mid
            else:
                r_high = r_mid

        return (r_low + r_high) / 2

    def _precompute_target(
        self,
        reference_image,
        resolution: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Extract alpha mask and compute distance transform.

        Returns:
            (target_mask [resolution, resolution],
             dt_weights [resolution, resolution],
             coverage fraction)
        """
        from scipy.ndimage import distance_transform_edt

        if isinstance(reference_image, Image.Image):
            if reference_image.mode != 'RGBA':
                raise ValueError("Reference image must be RGBA for silhouette correction")
            ref_resized = reference_image.resize(
                (resolution, resolution), Image.LANCZOS
            )
            alpha = np.array(ref_resized)[:, :, 3].astype(np.float64)
        elif isinstance(reference_image, np.ndarray):
            if reference_image.ndim == 3 and reference_image.shape[2] >= 4:
                alpha = reference_image[:, :, 3].astype(np.float64)
            else:
                alpha = reference_image.astype(np.float64)
            pil_alpha = Image.fromarray(alpha.astype(np.uint8))
            pil_alpha = pil_alpha.resize((resolution, resolution), Image.LANCZOS)
            alpha = np.array(pil_alpha).astype(np.float64)
        else:
            raise ValueError(f"Unsupported reference image type: {type(reference_image)}")

        target_binary = (alpha > 128).astype(np.float64)
        coverage = target_binary.sum() / (resolution * resolution)

        # Distance transform for loss weighting
        dt_outside = distance_transform_edt(1 - target_binary)
        dt_inside = distance_transform_edt(target_binary)
        target_dt = dt_outside + dt_inside

        dt_max = target_dt.max()
        if dt_max > 0:
            dt_weights = 1.0 + 4.0 * target_dt / dt_max
        else:
            dt_weights = np.ones_like(target_dt)

        target_mask = torch.tensor(
            target_binary, dtype=torch.float32, device=self.device
        )
        dt_weights_t = torch.tensor(
            dt_weights, dtype=torch.float32, device=self.device
        )

        return target_mask, dt_weights_t, float(coverage)

    def _compute_laplacian_target(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """Compute uniform Laplacian coordinates for original vertices."""
        V = vertices.shape[0]
        faces_long = faces.long()

        neighbor_sum = torch.zeros_like(vertices)
        neighbor_count = torch.zeros(V, 1, device=vertices.device)
        ones = torch.ones(faces_long.shape[0], 1, device=vertices.device)

        for i in range(3):
            j = (i + 1) % 3
            src = faces_long[:, i]
            dst = faces_long[:, j]
            neighbor_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), vertices[src])
            neighbor_count.scatter_add_(0, dst.unsqueeze(1), ones)
            neighbor_sum.scatter_add_(0, src.unsqueeze(1).expand(-1, 3), vertices[dst])
            neighbor_count.scatter_add_(0, src.unsqueeze(1), ones)

        centroid = neighbor_sum / neighbor_count.clamp(min=1)
        return vertices - centroid

    def _laplacian_loss(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        original_lap: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Laplacian regularization loss."""
        current_lap = self._compute_laplacian_target(vertices, faces)
        return F.mse_loss(current_lap, original_lap)

    def _compute_face_normals(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """Compute face normals."""
        faces_long = faces.long()
        v0 = vertices[faces_long[:, 0]]
        v1 = vertices[faces_long[:, 1]]
        v2 = vertices[faces_long[:, 2]]
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        return F.normalize(normals, dim=-1)

    def _normal_consistency_loss(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        original_normals: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize face normal flips relative to original."""
        new_normals = self._compute_face_normals(vertices, faces)
        dot = (new_normals * original_normals).sum(dim=-1)
        return F.relu(-dot).mean()

    def _silhouette_loss(
        self,
        rendered_mask: torch.Tensor,
        target_mask: torch.Tensor,
        dt_weights: torch.Tensor,
    ) -> torch.Tensor:
        """DT-weighted BCE loss between rendered and target silhouettes."""
        p = rendered_mask[0, :, :, 0].clamp(1e-6, 1 - 1e-6)
        bce = -(target_mask * torch.log(p) + (1 - target_mask) * torch.log(1 - p))
        return (bce * dt_weights).mean()

    def _soft_dice_loss(
        self,
        rendered_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Soft Dice loss — directly optimizes the A1 evaluation metric.

        Dice = 2*|A∩B| / (|A|+|B|). Differentiable via soft mask.
        Loss = 1 - Dice so that minimizing loss maximizes Dice.
        """
        p = rendered_mask[0, :, :, 0]
        intersection = (p * target_mask).sum()
        total = p.sum() + target_mask.sum()
        dice = (2 * intersection + 1) / (total + 1)  # +1 for numerical stability
        return 1.0 - dice

    def _compute_scale_invariant_dice(
        self,
        rendered_mask: torch.Tensor,
        target_mask: torch.Tensor,
        out_size: int = 256,
    ) -> float:
        """Compute scale-invariant Dice (matching V4 evaluator methodology)."""
        rend_hard = (rendered_mask[0, :, :, 0] > 0.5).float()
        rend_crop = _crop_to_bbox_tensor(rend_hard, pad_frac=0.05, out_size=out_size)
        ref_crop = _crop_to_bbox_tensor(target_mask, pad_frac=0.05, out_size=out_size)

        rend_bin = (rend_crop > 0.5).float()
        ref_bin = (ref_crop > 0.5).float()

        intersection = (rend_bin * ref_bin).sum()
        total = rend_bin.sum() + ref_bin.sum()
        return (2 * intersection / total.clamp(min=1)).item()

    def correct(
        self,
        glb_mesh: trimesh.Trimesh,
        reference_image,
        yaw: float = 0.0,
        pitch: float = 0.25,
        r: float = None,
        fov: float = 40.0,
        num_steps: int = 100,
        lr: float = 1e-3,
        w_silhouette: float = 1.0,
        w_laplacian: float = 10.0,
        w_normal: float = 3.0,
        max_displacement: float = 0.06,
        resolution: int = 512,
        use_dice_loss: bool = True,
        multi_resolution: bool = True,
        verbose: bool = True,
    ) -> trimesh.Trimesh:
        """Correct mesh silhouette to match reference image alpha mask.

        Uses auto-calibrated camera distance for scale-invariant comparison.
        The camera `r` is automatically found so the mesh fills the frame
        proportionally to the reference image.

        Args:
            glb_mesh: Input trimesh with UV-mapped PBR texture.
            reference_image: Reference RGBA image (PIL or numpy).
            yaw/pitch: Camera viewing angles.
            r: Camera distance (None = auto-calibrate to match reference coverage).
            fov: Field of view in degrees.
            num_steps: Number of optimization steps.
            lr: Learning rate for Adam optimizer.
            w_silhouette: Weight for silhouette BCE loss.
            w_laplacian: Weight for Laplacian regularization.
            w_normal: Weight for normal consistency loss.
            max_displacement: Max vertex displacement (fraction of unit cube).
            resolution: Render resolution for silhouette comparison.
            use_dice_loss: Use Soft Dice loss instead of BCE (directly optimizes A1).
            multi_resolution: Two-stage: coarse pass at 256px then fine at resolution.
            verbose: Print progress.

        Returns:
            Modified trimesh with deformed vertices (UVs/textures preserved).
        """
        if verbose:
            print(f"[SilhouetteCorrector] Starting ({num_steps} steps, lr={lr})")

        # Extract mesh data
        data = self._extract_mesh_data(glb_mesh)
        vertices_orig = data['vertices'].clone()
        faces = data['faces']

        # Precompute target mask and coverage
        target_mask, dt_weights, ref_coverage = self._precompute_target(
            reference_image, resolution
        )
        if verbose:
            print(f"  Reference coverage: {ref_coverage*100:.1f}%")

        # Auto-calibrate camera distance if not specified
        if r is None:
            r = self._auto_calibrate_r(
                vertices_orig, faces, yaw, pitch, fov,
                target_coverage=ref_coverage, resolution=256
            )
            if verbose:
                print(f"  Auto-calibrated r: {r:.3f}")

        # Setup camera with calibrated r
        extr, perspective = self._setup_camera(yaw, pitch, r, fov)

        # Verify coverage match
        with torch.no_grad():
            check_mask = self._render_silhouette(
                vertices_orig, faces, extr, perspective, resolution
            )
            actual_coverage = check_mask.sum().item() / (resolution * resolution)
            if verbose:
                print(f"  Rendered coverage: {actual_coverage*100:.1f}%")

        # Create optimizable vertex parameter
        vertices_opt = vertices_orig.clone().detach().requires_grad_(True)

        # Precompute Laplacian coordinates
        original_lap = self._compute_laplacian_target(vertices_orig, faces).detach()

        # Precompute original face normals
        original_normals = self._compute_face_normals(vertices_orig, faces).detach()

        # Compute initial scale-invariant Dice
        with torch.no_grad():
            init_mask = self._render_silhouette(
                vertices_orig, faces, extr, perspective, resolution
            )
            init_dice = self._compute_scale_invariant_dice(init_mask, target_mask)
            if verbose:
                print(f"  Initial scale-invariant Dice: {init_dice:.4f}")

        # Build optimization stages
        if multi_resolution and resolution > 256:
            # Stage 1: coarse (256px, half steps) for large movements
            # Stage 2: fine (full resolution, remaining steps) for boundary refinement
            coarse_steps = num_steps // 3
            fine_steps = num_steps - coarse_steps
            stages = [
                (256, coarse_steps, lr, max_displacement),
                (resolution, fine_steps, lr * 0.5, max_displacement),
            ]
        else:
            stages = [(resolution, num_steps, lr, max_displacement)]

        # Optimize
        best_dice = init_dice
        best_vertices = vertices_orig.clone()
        global_step = 0

        for stage_idx, (stage_res, stage_steps, stage_lr, stage_max_disp) in enumerate(stages):
            if verbose and len(stages) > 1:
                print(f"  Stage {stage_idx+1}/{len(stages)}: {stage_res}px, {stage_steps} steps")

            # Prepare target at stage resolution
            stage_target, stage_dt, _ = self._precompute_target(
                reference_image, stage_res
            )

            optimizer = torch.optim.Adam([vertices_opt], lr=stage_lr)

            for step in range(stage_steps):
                optimizer.zero_grad()

                mask = self._render_silhouette(
                    vertices_opt, faces, extr, perspective, stage_res
                )

                if use_dice_loss:
                    l_sil = self._soft_dice_loss(mask, stage_target)
                else:
                    l_sil = self._silhouette_loss(mask, stage_target, stage_dt)
                l_lap = self._laplacian_loss(vertices_opt, faces, original_lap)
                l_norm = self._normal_consistency_loss(vertices_opt, faces, original_normals)

                loss = w_silhouette * l_sil + w_laplacian * l_lap + w_normal * l_norm
                loss.backward()
                optimizer.step()

                # Clamp displacement
                with torch.no_grad():
                    disp = vertices_opt - vertices_orig
                    mag = disp.norm(dim=-1, keepdim=True)
                    clamped = torch.where(
                        mag > stage_max_disp,
                        disp * stage_max_disp / mag,
                        disp
                    )
                    vertices_opt.copy_(vertices_orig + clamped)

                global_step += 1

                # Track best Dice every 10 steps
                if (step + 1) % 10 == 0 or step == stage_steps - 1:
                    with torch.no_grad():
                        # Always evaluate at full resolution for consistent tracking
                        eval_mask = self._render_silhouette(
                            vertices_opt, faces, extr, perspective, resolution
                        )
                        dice = self._compute_scale_invariant_dice(eval_mask, target_mask)
                        if dice > best_dice:
                            best_dice = dice
                            best_vertices = vertices_opt.data.clone()
                        if verbose:
                            max_disp = (vertices_opt - vertices_orig).norm(dim=-1).max().item()
                            print(
                                f"  Step {global_step}/{num_steps}: "
                                f"sil={l_sil:.4f} lap={l_lap:.6f} "
                                f"norm={l_norm:.6f} dice={dice:.4f} "
                                f"max_disp={max_disp:.5f}"
                            )

        # Apply best vertices to mesh
        improvement = best_dice - init_dice
        if improvement > 0:
            new_verts = best_vertices.detach().cpu().numpy()
            glb_mesh.vertices = new_verts
            glb_mesh._cache.clear()
            if verbose:
                print(
                    f"[SilhouetteCorrector] Done. Dice: {init_dice:.4f} -> "
                    f"{best_dice:.4f} (+{improvement:.4f})"
                )
        else:
            if verbose:
                print(
                    f"[SilhouetteCorrector] No improvement (init={init_dice:.4f}, "
                    f"best={best_dice:.4f}). Keeping original mesh."
                )

        return glb_mesh
