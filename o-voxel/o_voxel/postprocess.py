from typing import *
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F_torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual
from flex_gemm.ops.grid_sample import grid_sample_3d
import nvdiffrast.torch as dr
import cumesh


def _push_pull_padding(texture, mask, levels=6):
    """Push-pull pyramid padding for UV texture maps.

    Fills invalid texels by propagating colors from valid regions without
    cross-island contamination. Much better than cv2.inpaint for UV maps.

    Args:
        texture: (H, W, C) uint8 array
        mask: (H, W) bool array of valid texels
        levels: Number of pyramid levels

    Returns:
        (H, W, C) uint8 array with all texels filled
    """
    h, w = texture.shape[:2]
    tex_f = texture.astype(np.float32)
    weight = mask.astype(np.float32)

    # Push phase: build pyramid by downsampling
    pyramid_tex = [tex_f]
    pyramid_w = [weight]
    for _ in range(levels):
        # Weighted average downsampling
        tex_weighted = pyramid_tex[-1] * pyramid_w[-1][..., None]
        down_tw = cv2.resize(tex_weighted, (max(1, pyramid_tex[-1].shape[1] // 2),
                                             max(1, pyramid_tex[-1].shape[0] // 2)),
                             interpolation=cv2.INTER_AREA)
        down_w = cv2.resize(pyramid_w[-1], (down_tw.shape[1], down_tw.shape[0]),
                            interpolation=cv2.INTER_AREA)
        safe_w = np.maximum(down_w, 1e-6)
        down_tex = down_tw / safe_w[..., None]
        pyramid_tex.append(down_tex)
        pyramid_w.append(down_w)

    # Pull phase: reconstruct from coarse to fine
    result = pyramid_tex[-1]
    result_w = (pyramid_w[-1] > 0.01).astype(np.float32)

    for lev in range(levels - 1, -1, -1):
        # Upsample coarse result to fine resolution
        fh, fw = pyramid_tex[lev].shape[:2]
        up_result = cv2.resize(result, (fw, fh), interpolation=cv2.INTER_LINEAR)
        up_w = cv2.resize(result_w, (fw, fh), interpolation=cv2.INTER_LINEAR)

        # Use fine-level data where available, coarse fill elsewhere
        fine_w = pyramid_w[lev]
        has_fine = fine_w > 0.01
        result = np.where(has_fine[..., None], pyramid_tex[lev], up_result)
        result_w = np.where(has_fine, 1.0, up_w)

    return np.clip(result, 0, 255).astype(np.uint8)


def _detect_uv_seams(mask, uvs, faces, texture_size, width=3):
    """Detect UV seam boundaries in texture space.

    UV seams occur at chart boundaries where neighboring texels in the texture
    come from different UV charts. These cause visible discontinuities.

    Args:
        mask: [H, W] bool, valid texel mask.
        uvs: [V, 2] tensor, UV coordinates.
        faces: [F, 3] tensor, face indices.
        texture_size: int, texture resolution.
        width: int, seam band width in pixels.

    Returns:
        [H, W] bool mask of seam pixels.
    """
    # Simple approach: detect seams via mask boundary erosion.
    # UV chart edges are where valid texels border invalid ones.
    mask_u8 = mask.astype(np.uint8) * 255

    # Erode mask to find interior
    kernel = np.ones((width * 2 + 1, width * 2 + 1), np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)

    # Seam band = valid texels that are near the edge of the valid region
    seam_mask = mask & (eroded < 128)

    return seam_mask


def _laplacian_seam_smooth(texture, seam_mask, levels=3):
    """Multi-scale Laplacian blending at UV seams.

    Builds a Gaussian pyramid, smooths the seam regions at each level,
    then reconstructs. This preserves detail away from seams while
    reducing discontinuities at seam boundaries.

    Args:
        texture: [H, W, 3] uint8 base color texture.
        seam_mask: [H, W] bool, seam pixel mask.
        levels: int, number of pyramid levels.

    Returns:
        [H, W, 3] uint8 smoothed texture.
    """
    tex_float = texture.astype(np.float32)
    result = tex_float.copy()

    # Build Gaussian pyramid
    pyramid = [tex_float]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))

    # Build mask pyramid
    mask_float = seam_mask.astype(np.float32)
    mask_pyramid = [mask_float]
    for _ in range(levels):
        mask_pyramid.append(cv2.pyrDown(mask_pyramid[-1]))

    # Smooth from coarse to fine
    for level in reversed(range(levels)):
        # Get the smoothed version at this level
        sigma = 3.0 * (level + 1)
        smoothed = cv2.GaussianBlur(pyramid[level], (0, 0), sigmaX=sigma)

        # Resize mask to match this level
        h, w = pyramid[level].shape[:2]
        level_mask = cv2.resize(mask_pyramid[level], (w, h))

        # Blend factor: stronger smoothing at coarser levels
        blend_strength = np.clip(level_mask, 0, 1)[..., None]
        alpha = blend_strength * (0.3 + 0.2 * level)  # 0.3-0.7 blend range
        pyramid[level] = pyramid[level] * (1 - alpha) + smoothed * alpha

    # Reconstruct: only apply changes at seam pixels
    seam_3d = seam_mask[..., None].astype(np.float32)
    result = result * (1 - seam_3d) + pyramid[0] * seam_3d

    return np.clip(result, 0, 255).astype(np.uint8)


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    max_metallic: float = None,
    min_roughness: float = None,
    normal_strength: float = 1.0,
    normal_blur_sigma: float = 0.0,
    enable_grey_recovery: bool = True,
    enable_normal_map: bool = True,
    enable_ao: bool = True,
    enable_seam_smoothing: bool = True,
    seam_smooth_levels: int = 3,
    seam_smooth_width: int = 3,
    grey_recovery_rounds: int = 2,
    grey_recovery_sigma: float = 25.0,
    sharpen_texture: bool = False,
    enable_mesh_smooth: bool = True,
    mesh_smooth_iterations: int = 3,
    mesh_smooth_lamb: float = 0.5,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Convert an extracted mesh to a GLB file.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.

    Safe enhancements over upstream:
    - Normal map baking (tangent-space, from high-res to simplified mesh)
    - Ambient Occlusion map (cuBVH ray-traced)
    - Metallic clamp (prevents unreasonable metallic values)
    - Min roughness floor (prevents overly shiny surfaces)

    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sparse tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        max_metallic: optional clamp for metallic channel (0.0-1.0)
        min_roughness: optional floor for roughness channel (0.0-1.0), prevents overly shiny surfaces
        normal_strength: scale factor for normal map perturbation (1.0=full detail, 0.0=flat), default 1.0
        normal_blur_sigma: Gaussian blur sigma for normal map (0=no blur), smooths high-freq noise
        sharpen_texture: unused, kept for API compatibility
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
    """
    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3

    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Move data to GPU
    vertices = vertices.cuda()
    faces = faces.cuda()

    # Initialize CUDA mesh handler
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)

    # --- Initial Mesh Cleaning ---
    # Fills holes as much as we can before processing
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    vertices, faces = mesh.read()
    if use_tqdm:
        pbar.update(1)

    # Build BVH for the current mesh to guide remeshing
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose:
        print(f"Building BVH for current mesh...", end='', flush=True)
    bvh = cumesh.cuBVH(vertices, faces)
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    if verbose:
        print("Cleaning mesh...")

    # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
    if not remesh:
        # Step 1: Aggressive simplification (3x target)
        mesh.simplify(decimation_target * 3, verbose=verbose)
        if verbose:
            print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 3: Final simplification to target count
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 4: Final Cleanup loop
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 5: Unify face orientations
        mesh.unify_face_orientations()

    # --- Branch 2: Remeshing Pipeline ---
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()

        # Perform Dual Contouring remeshing (rebuilds topology)
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = verbose,
            bvh = bvh,
        ))
        if verbose:
            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Simplify and clean the remeshed result
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Cleanup after remesh + simplify (fill holes, repair topology)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=5e-2)
        mesh.unify_face_orientations()
        if verbose:
            print(f"After remesh cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    # --- Taubin Mesh Smoothing ---
    # Reduces voxel staircase artifacts from marching cubes without shrinking.
    # Applied after simplification/cleanup, before UV unwrapping.
    if enable_mesh_smooth and mesh_smooth_iterations > 0:
        if verbose:
            print(f"Taubin smoothing ({mesh_smooth_iterations} iters, lambda={mesh_smooth_lamb})...")
        verts_np, faces_np = mesh.read()
        verts_np = verts_np.cpu().numpy()
        faces_np = faces_np.cpu().numpy()
        tm = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        trimesh.smoothing.filter_taubin(
            tm,
            lamb=mesh_smooth_lamb,
            nu=mesh_smooth_lamb + 0.01,  # positive nu for correct shrink-dilate alternation
            iterations=mesh_smooth_iterations,
        )
        smoothed_verts = torch.tensor(tm.vertices, dtype=torch.float32, device=vertices.device)
        mesh.init(smoothed_verts, torch.tensor(tm.faces, dtype=torch.int32, device=faces.device))
        if verbose:
            print(f"After smoothing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

    # --- UV Parameterization ---
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")

    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    # --- Texture Baking (Attribute Sampling) ---
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)

    # Setup differentiable rasterizer context
    ctx = dr.RasterizeCudaContext()
    # Prepare UV coordinates for rasterization (rendering in UV space)
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)

    # Rasterize in chunks to save memory
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i+100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i # Store face ID in alpha channel
        rast = torch.where(mask_chunk, rast_chunk, rast)

    # Mask of valid pixels in texture
    mask = rast[0, ..., 3] > 0

    # Interpolate 3D positions in UV space (finding 3D coord for every texel)
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]

    # BVH query: always needed for normal map baking and AO
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)

    # BVH reprojection: snap positions back to original mesh surface for texture sampling.
    # DIAGNOSTIC: Toggle to test if BVH causes grey patches on thin-walled objects.
    USE_BVH_REPROJECTION = True  # TEST 1 result: BVH HELPS (41.5% grey without vs 13.8% with)
    if USE_BVH_REPROJECTION:
        orig_tri_verts = vertices[faces[face_id.long()]]
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    # Sample attributes from sparse voxel grid using trilinear interpolation
    # (same mode as the preview renderer for consistent results)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    sparse_coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)
    sparse_shape = torch.Size([1, attr_volume.shape[1], *grid_size.tolist()])
    grid_pts = ((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3)

    sampled_attrs = grid_sample_3d(
        attr_volume, sparse_coords, shape=sparse_shape, grid=grid_pts, mode='trilinear',
    )
    attrs[mask] = sampled_attrs
    del sampled_attrs

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    # --- Normal Map Baking ---
    normal_map_np = None
    if enable_normal_map:
        # Bake tangent-space normal map from original high-res mesh onto simplified mesh.
        # Captures fine geometric detail lost during simplification/remeshing.
        if verbose:
            print("Baking normal map...", end='', flush=True)

        # Compute area-weighted vertex normals for original mesh
        _fv = vertices[faces]  # (F_orig, 3, 3)
        _fn = torch.cross(_fv[:, 1] - _fv[:, 0], _fv[:, 2] - _fv[:, 0], dim=-1)
        vert_normals_orig = torch.zeros_like(vertices)
        vert_normals_orig.scatter_add_(
            0, faces.reshape(-1, 1).expand(-1, 3),
            _fn.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)
        )
        vert_normals_orig = F_torch.normalize(vert_normals_orig, dim=-1)
        del _fv, _fn

        # Interpolate original normals at BVH hit points using barycentric coords
        _tri_n = vert_normals_orig[faces[face_id.long()]]  # (N_valid, 3, 3)
        N_high = F_torch.normalize(
            (_tri_n * uvw.unsqueeze(-1)).sum(dim=1), dim=-1
        )
        del vert_normals_orig, _tri_n

        # Compute per-face tangent frame for simplified mesh from positions and UVs
        _fp = out_vertices[out_faces]  # (F_simp, 3, 3)
        _fuv = out_uvs[out_faces]  # (F_simp, 3, 2)
        _dP1 = _fp[:, 1] - _fp[:, 0]
        _dP2 = _fp[:, 2] - _fp[:, 0]
        _dUV1 = _fuv[:, 1] - _fuv[:, 0]
        _dUV2 = _fuv[:, 2] - _fuv[:, 0]
        _det = _dUV1[:, 0] * _dUV2[:, 1] - _dUV1[:, 1] * _dUV2[:, 0]
        _det = torch.where(_det.abs() < 1e-8, torch.ones_like(_det), _det)
        _inv = 1.0 / _det
        T_face = _inv.unsqueeze(-1) * (_dUV2[:, 1:2] * _dP1 - _dUV1[:, 1:2] * _dP2)
        B_face = _inv.unsqueeze(-1) * (_dUV1[:, 0:1] * _dP2 - _dUV2[:, 0:1] * _dP1)
        N_face = F_torch.normalize(torch.cross(_dP1, _dP2, dim=-1), dim=-1)
        # Gram-Schmidt orthogonalize
        T_face = T_face - N_face * (T_face * N_face).sum(-1, keepdim=True)
        T_face = F_torch.normalize(T_face, dim=-1)
        B_face = torch.cross(N_face, T_face, dim=-1)
        del _fp, _fuv, _dP1, _dP2, _dUV1, _dUV2, _det, _inv

        # Get simplified mesh face ID at each valid texel and lookup tangent frame
        _rfi = (rast[0, ..., 3].long() - 1)[mask]
        _T = T_face[_rfi]; _B = B_face[_rfi]; _N = N_face[_rfi]
        del T_face, B_face, N_face, _rfi

        # Transform original normals to tangent space
        _tn = F_torch.normalize(torch.stack([
            (N_high * _T).sum(-1), (N_high * _B).sum(-1), (N_high * _N).sum(-1),
        ], dim=-1), dim=-1)
        del N_high, _T, _B, _N

        # Attenuate normal map strength (scale XY toward flat, re-normalize)
        if normal_strength < 1.0:
            _tn[:, :2] *= normal_strength
            _tn = F_torch.normalize(_tn, dim=-1)

        # Build normal map (default: flat tangent-space normal (0, 0, 1))
        normal_map_gpu = torch.zeros(texture_size, texture_size, 3, device='cuda')
        normal_map_gpu[..., 2] = 1.0
        normal_map_gpu[mask] = _tn
        normal_map_np = ((normal_map_gpu * 0.5 + 0.5) * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        del normal_map_gpu, _tn

        # Gaussian blur to smooth high-frequency normal map noise
        if normal_blur_sigma > 0:
            ksize = int(np.ceil(normal_blur_sigma * 3)) * 2 + 1
            normal_map_np = cv2.GaussianBlur(normal_map_np, (ksize, ksize), normal_blur_sigma)

        if verbose:
            print("Done")

    # --- Ambient Occlusion Map ---
    ao_np = None
    if enable_ao:
        # Cast rays in hemisphere above each texel to detect nearby occluders.
        if verbose:
            print("Computing AO map...", end='', flush=True)

        ao_radius = 0.03
        ao_min_dist = 0.005
        ao_num_rays = 16
        ao_strength = 0.2
        _e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        _e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        _face_n = F_torch.normalize(torch.cross(_e1, _e2, dim=-1), dim=-1)
        ao_normals = _face_n[face_id.long()]
        del _e1, _e2, _face_n

        N_valid = ao_normals.shape[0]
        ao_values = torch.ones(N_valid, device='cuda')
        ao_batch = 500000

        for bi in range(0, N_valid, ao_batch):
            bj = min(bi + ao_batch, N_valid)
            bn = bj - bi
            b_pos = valid_pos[bi:bj]
            b_n = ao_normals[bi:bj]

            # Build tangent frame from face normals
            up = torch.zeros_like(b_n)
            up[:, 1] = 1.0
            par = (b_n * up).sum(-1).abs() > 0.99
            up[par] = torch.tensor([1.0, 0.0, 0.0], device=b_n.device)
            T = F_torch.normalize(torch.cross(up, b_n, dim=-1), dim=-1)
            B = torch.cross(b_n, T, dim=-1)

            # Cosine-weighted hemisphere sampling
            u1 = torch.rand(bn, ao_num_rays, device='cuda')
            u2 = torch.rand(bn, ao_num_rays, device='cuda')
            sin_t = torch.sqrt(u1)
            cos_t = torch.sqrt(1.0 - u1)
            phi = 6.2831853 * u2
            dx = sin_t * torch.cos(phi)
            dy = sin_t * torch.sin(phi)
            dz = cos_t
            dirs = (dx.unsqueeze(-1) * T.unsqueeze(1) +
                    dy.unsqueeze(-1) * B.unsqueeze(1) +
                    dz.unsqueeze(-1) * b_n.unsqueeze(1))

            origins = (b_pos + b_n * 0.003).unsqueeze(1).expand_as(dirs)
            _, _, hit_d = bvh.ray_trace(origins.reshape(-1, 3), dirs.reshape(-1, 3))
            hit_d = hit_d.reshape(bn, ao_num_rays)
            hit_mask = (hit_d > ao_min_dist) & (hit_d < ao_radius)
            falloff = torch.where(hit_mask, 1.0 - hit_d / ao_radius, torch.zeros_like(hit_d))
            ao_values[bi:bj] = 1.0 - ao_strength * falloff.mean(dim=1)

        ao_tex = torch.ones(texture_size, texture_size, device='cuda')
        ao_tex[mask] = ao_values
        ao_np = (ao_tex * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        ao_np = cv2.GaussianBlur(ao_np, (3, 3), 0.5)
        del ao_tex, ao_values, ao_normals

        if verbose:
            print("Done")

    # --- Texture Post-Processing & Material Construction ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)

    mask = mask.cpu().numpy()

    # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha_mode = 'OPAQUE'

    # --- Grey/desaturated texel recovery (multi-round with chroma-aware weights) ---
    if enable_grey_recovery:
        # The voxel grid produces low-saturation (grey) values for surfaces not well-visible
        # in the input image (back, inside, etc.). Use multi-round Gaussian-weighted diffusion
        # from nearby colorful texels to fill grey areas with plausible colors.
        for grey_round in range(grey_recovery_rounds):
            bc_float = base_color.astype(np.float32)
            max_c = np.maximum(np.maximum(bc_float[..., 0], bc_float[..., 1]), bc_float[..., 2])
            min_c = np.minimum(np.minimum(bc_float[..., 0], bc_float[..., 1]), bc_float[..., 2])
            chroma = max_c - min_c
            lum = (max_c + min_c) / 2
            # Grey texels: valid, very low chroma, mid brightness (not black/white)
            grey_texels = mask & (chroma < 18) & (lum > 80) & (lum < 200)
            grey_pct = grey_texels.sum() / max(1, mask.sum()) * 100
            if verbose:
                print(f"\n  Grey recovery round {grey_round+1}/{grey_recovery_rounds}: {grey_pct:.1f}% grey", end='', flush=True)
            if grey_texels.sum() == 0:
                break
            # Gaussian-weighted diffusion: spread color ONLY from colorful (high-chroma) texels
            colorful_mask = mask & (chroma >= 18)
            # Chroma-aware weights: more colorful texels contribute more
            chroma_weight = np.clip(chroma / 255.0, 0, 1)
            colorful_weight = colorful_mask.astype(np.float32) * (0.3 + 0.7 * chroma_weight)
            colorful_color = bc_float.copy()
            colorful_color[~colorful_mask] = 0.0
            colorful_color *= colorful_weight[..., None]
            # Increasing sigma per round for broader coverage
            sigma = grey_recovery_sigma + grey_round * 10.0
            blurred_color = cv2.GaussianBlur(colorful_color, (0, 0), sigmaX=sigma)
            blurred_weight = cv2.GaussianBlur(colorful_weight, (0, 0), sigmaX=sigma)
            safe_weight = np.maximum(blurred_weight, 1e-6)
            avg_color = blurred_color / safe_weight[..., None]
            # Blend: decreasing original weight each round
            blend = 0.7 + 0.1 * grey_round
            blend = min(blend, 0.9)
            blended = blend * avg_color[grey_texels] + (1 - blend) * bc_float[grey_texels]
            base_color[grey_texels] = np.clip(blended, 0, 255).astype(np.uint8)

    # Texture padding: fill gaps to prevent black seams at UV boundaries
    # Push-pull padding for base_color (prevents cross-island contamination)
    base_color = _push_pull_padding(base_color, mask, levels=6)
    mask_inv = (~mask).astype(np.uint8)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness if roughness.ndim == 2 else roughness[..., 0], mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    if normal_map_np is not None:
        normal_map_np = cv2.inpaint(normal_map_np, mask_inv, 3, cv2.INPAINT_TELEA)
    if ao_np is not None:
        ao_np = cv2.inpaint(ao_np, mask_inv, 1, cv2.INPAINT_TELEA)

    # --- UV Seam Smoothing (Laplacian multi-scale blend) ---
    if enable_seam_smoothing:
        # Detect UV seam boundaries: pixels where the UV gradient is discontinuous.
        # Seams appear at UV chart edges — neighboring pixels in texture space may be
        # far apart in UV space, causing visible color discontinuities.
        seam_mask = _detect_uv_seams(mask, out_uvs, out_faces, texture_size, seam_smooth_width)
        seam_pct = seam_mask.sum() / max(1, mask.sum()) * 100
        if verbose:
            print(f"\n  UV seam pixels: {seam_pct:.1f}%", end='', flush=True)
        if seam_mask.sum() > 0:
            base_color = _laplacian_seam_smooth(base_color, seam_mask, seam_smooth_levels)
            if verbose:
                print(" smoothed", end='', flush=True)

    # Optional: clamp metallic to prevent unreasonable values
    if max_metallic is not None:
        max_val = int(np.clip(max_metallic * 255, 0, 255))
        metallic = np.clip(metallic, 0, max_val)

    # Optional: enforce minimum roughness to prevent overly shiny surfaces
    if min_roughness is not None:
        min_val = int(np.clip(min_roughness * 255, 0, 255))
        roughness = np.clip(roughness, min_val, 255)

    # Create PBR material (conditionally include normal map and AO)
    mat_kwargs = dict(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True if not remesh else False,
    )
    if normal_map_np is not None:
        mat_kwargs['normalTexture'] = Image.fromarray(normal_map_np)
    if ao_np is not None:
        mat_kwargs['occlusionTexture'] = Image.fromarray(ao_np)
    material = trimesh.visual.material.PBRMaterial(**mat_kwargs)

    # --- Coordinate System Conversion & Final Object ---
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()

    # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate

    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
    )

    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")

    return textured_mesh
