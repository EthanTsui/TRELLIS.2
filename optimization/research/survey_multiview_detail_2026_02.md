# Multi-View Detail Quality Survey: Latest Methods for TRELLIS.2

**Date**: 2026-02-21
**Author**: Research Optimizer Agent
**Scope**: Papers and methods from late 2024 through early 2026 focused on improving multi-view 3D generation detail quality
**Builds on**: Previous survey `improvement_methods.md` (2026-02-19, 44 papers)

---

## Executive Summary

This survey covers 35+ new papers and methods not previously catalogued, discovered through web searches of arXiv, CVPR 2025, ICCV 2025, ICLR 2025-2026, SIGGRAPH 2025, NeurIPS 2025, and AAAI 2025 proceedings. The focus is on five areas directly relevant to TRELLIS.2's quality bottlenecks:

1. **Multi-view consistency and fusion** -- how to better combine information from multiple views
2. **High-resolution texture generation** -- methods for sharper, more detailed textures
3. **Geometry detail enhancement** -- recovering fine details lost in voxel-to-mesh conversion
4. **Post-generation refinement** -- upscaling and polishing 3D models after initial generation
5. **Flow matching improvements** -- latest advances in guidance and sampling for flow-based models

### Top New Findings for TRELLIS.2

| Rank | Method | Area | Impact | Effort | Key Insight |
|------|--------|------|--------|--------|-------------|
| 1 | **SpArSe-Up** | Texture | Very High | Medium | Learnable sparse upsampling constrains voxels to surface, 70% memory reduction, enables high-res texture training |
| 2 | **NaTex / LaFiTe / TEXTRIX** | Texture | Very High | Hard | Native 3D texture as color point cloud / latent field / attribute grid -- bypasses UV entirely |
| 3 | **GTR** | Post-process | High | Medium | Per-instance texture refinement via mesh rendering in 4 seconds (ICLR 2025) |
| 4 | **SparseFlex (TripoSF)** | Geometry | Very High | Hard | 1024^3 isosurface extraction with sparse voxels, 82% CD reduction vs FlexiCubes |
| 5 | **MVPainter** | Texture | High | Medium | ControlNet geometric conditioning + PBR extraction from generated views |
| 6 | **Cycle3D** | Multi-view | High | Hard | Generation-reconstruction cycle enforces consistency during diffusion |
| 7 | **OC-Flow** | Flow Match | Medium | Medium | Optimal control framework for training-free guided flow matching with convergence proof |
| 8 | **MaterialMVP** | Texture | High | Medium | Illumination-invariant PBR via dual-channel generation + consistency regularization |
| 9 | **TetWeave** | Geometry | High | Hard | Adaptive Delaunay tetrahedral mesh extraction, guaranteed manifold, SIGGRAPH 2025 |
| 10 | **Seamless3D** | Multi-view | Medium | Medium | Cross-view dynamic attention fusion + triplane texture fusion for seamless UV |

---

## 1. Multi-View Consistency and Fusion

### 1.1 Cycle3D: Generation-Reconstruction Cycle (AAAI 2025)

**Paper**: "Cycle3D: High-quality and Consistent Image-to-3D Generation via Generation-Reconstruction Cycle"
**Venue**: AAAI 2025
**ArXiv**: https://arxiv.org/abs/2407.19548
**Code**: https://github.com/PKU-YuanGroup/Cycle3D

**Key Technique**: Cyclically alternates between a 2D diffusion generation module and a feed-forward 3D reconstruction module during multi-step diffusion. The 2D model generates high-quality texture, while the reconstruction model guarantees multi-view consistency. Each denoising step produces geometry-consistent outputs.

**Relevance to TRELLIS.2**: TRELLIS.2 generates texture in a single forward pass through its texture SLAT stage. Cycle3D's insight -- that cycling between 2D quality and 3D consistency during generation produces better results -- could inform a post-generation refinement loop: render TRELLIS.2 output from multiple views, enhance with a 2D diffusion model, reconstruct back to 3D, repeat.

**Key Advantage**: The cyclical approach inherently enforces multi-view consistency without explicit multi-view attention mechanisms. This is architecturally different from TRELLIS.2's single-pass approach.

**Feasibility**: Hard (requires full render-reconstruct pipeline)
**Expected Impact**: High (+3-5 points on texture coherence and detail)
**Code Available**: Yes

### 1.2 Seamless3D: Cross-View Attention + Triplane UV (2025)

**Paper**: "Seamless3D: Structured Multi-View Texture Generation with Cross-View Attention and Triplane-UV Optimization"
**Venue**: 3rd International Workshop on Multimedia Content Generation and Evaluation (2025)
**Link**: https://dl.acm.org/doi/10.1145/3746278.3759391

**Key Technique**: Three-stage framework:
1. Cross-View Dynamic Attention Fusion (CVDAF) -- aggregates features from multiple views using depth and normal maps as geometric priors
2. Dual-branch ControlNet with geometry-aware conditions for multi-view-consistent image synthesis
3. Triplane Texture Fusion -- transforms sparse 2D views into a continuous, seamless texture representation, significantly mitigating UV projection artifacts

**Relevance to TRELLIS.2**: The triplane texture fusion (stage 3) is particularly relevant. Instead of directly projecting multi-view images onto UV maps (which causes seam artifacts), Seamless3D fuses views through a triplane intermediary that naturally handles view transitions. This could replace TRELLIS.2's current UV baking step.

**Key Insight**: The triplane acts as a 3D-consistent intermediary between 2D views and UV space, smoothing transitions that would otherwise create visible seams.

**Feasibility**: Medium (triplane fusion can be extracted as a module)
**Expected Impact**: Medium (+2-3 points on smoothness and fragmentation)
**Code Available**: Not yet

### 1.3 3D-Adapter: Geometry Feedback Augmentation (ICLR 2025)

**Paper**: "3D-Adapter: Geometry-Consistent Multi-View Diffusion for High-Quality 3D Generation"
**Venue**: ICLR 2025
**ArXiv**: https://arxiv.org/abs/2410.18974
**Code**: https://github.com/Lakonik/MVEdit

**Key Technique**: Plug-in module that infuses 3D geometry awareness into pretrained image diffusion models via "3D feedback augmentation." At each denoising step:
1. Decode intermediate multi-view features into a coherent 3D representation (Gaussian splatting or neural field)
2. Re-render RGBD views from this 3D representation
3. Feed rendered views back to augment the diffusion model via feature addition

Two variants: fast feed-forward (Gaussian splatting) and training-free (neural fields + meshes).

**Relevance to TRELLIS.2**: While TRELLIS.2 operates in 3D voxel space natively (so doesn't need 2D-to-3D feedback), the concept of periodically "consolidating" intermediate results into a consistent 3D representation during generation is applicable. For multi-view texture generation specifically, 3D-Adapter's approach of rendering from a 3D intermediary at each step could improve TRELLIS.2's TAPA multi-view texture mode.

**Feasibility**: Medium-Hard (requires integration with texture generation stage)
**Expected Impact**: Medium (+2-3 points on multi-view consistency)
**Code Available**: Yes

### 1.4 MV-Adapter: Unified Multi-View Image Generation (ICLR 2025)

**Paper**: "MV-Adapter: Multi-view Consistent Image Generation Made Easy"
**Venue**: ICLR 2025
**Link**: https://openreview.net/forum?id=kcmK2utDhu

**Key Technique**: Unified implementation for generating multi-view images from various conditions (text, image, geometry). Provides a standardized framework for multi-view generation that can be adapted to different base diffusion models.

**Relevance to TRELLIS.2**: Could serve as a better multi-view image generator for TRELLIS.2's multi-view input pipeline. Currently, TRELLIS.2 uses Zero123++ or similar models for generating auxiliary views. MV-Adapter's unified approach might produce more consistent multi-view images.

**Feasibility**: Medium (drop-in replacement for multi-view generation)
**Expected Impact**: Medium (+1-3 points on input quality)
**Code Available**: Limited

### 1.5 Summary Table: Multi-View Methods

| Method | Year | Venue | Technique | Feasibility | Impact | Code |
|--------|------|-------|-----------|-------------|--------|------|
| Cycle3D | 2025 | AAAI | Gen-recon cycle | Hard | High | Yes |
| Seamless3D | 2025 | Workshop | CVDAF + triplane UV | Medium | Medium | No |
| 3D-Adapter | 2025 | ICLR | 3D feedback augmentation | Med-Hard | Medium | Yes |
| MV-Adapter | 2025 | ICLR | Unified MV generation | Medium | Medium | Limited |

---

## 2. High-Resolution Texture Generation

### 2.1 NaTex: Native 3D Texture as Latent Color Diffusion (Nov 2025)

**Paper**: "NaTex: Seamless Texture Generation as Latent Color Diffusion"
**ArXiv**: https://arxiv.org/abs/2511.16317
**Code**: https://github.com/Zeqiang-Lai/NaTex

**Key Technique**: Predicts texture color directly in 3D space by treating texture as a dense color point cloud. Architecture:
1. Geometry-aware color point cloud VAE encodes appearance
2. Multi-control diffusion transformer (DiT) generates in the latent space
3. Native geometry control via positional embeddings and geometry latents

**Key Advantages Over Multi-View Fusion**:
- No occluded regions requiring inpainting
- No mesh-texture boundary alignment issues
- No cross-view consistency/coherence problems
- Seamless by construction (no UV seams)

**Relevance to TRELLIS.2**: NaTex's approach is philosophically similar to TRELLIS.2's texture SLAT (both operate on 3D volumetric representations). NaTex's insight of conditioning on geometry via a dedicated geometry branch could improve TRELLIS.2's texture stage, which currently receives geometry implicitly through the sparse structure. NaTex could also serve as a post-generation texture refinement: take TRELLIS.2's geometry, re-texture it with NaTex using the input image as conditioning.

**Feasibility**: Medium (pre-trained models available; need to integrate with TRELLIS.2's mesh output)
**Expected Impact**: High (+3-5 points on texture quality, especially smoothness and fragmentation)
**Code Available**: Yes

### 2.2 LaFiTe: Latent Color Field for 3D Native Texturing (Dec 2025)

**Paper**: "LaFiTe: A Generative Latent Field for 3D Native Texturing"
**ArXiv**: https://arxiv.org/abs/2512.04786

**Key Technique**: Models 3D texture as a sparse latent color field using a novel VAE. Key innovations:
1. Encodes colored point clouds into sparse, structured latent features concentrated near the object surface
2. Local and continuous representation -- queryable at any 3D point
3. Geometry conditioning via encoding an untextured mesh to extract geometry latents
4. Conditional rectified-flow model (not diffusion) for generation

**Quantitative**: Exceeds state-of-the-art by **>10 dB PSNR in reconstruction** -- a massive improvement.

**Relevance to TRELLIS.2**: LaFiTe uses rectified flow (like TRELLIS.2!) for its generative model. The latent color field is conceptually similar to TRELLIS.2's texture SLAT but with a more principled continuous formulation. The >10 dB PSNR improvement suggests significantly better texture fidelity. Downstream applications include texture super-resolution, directly applicable as a post-processor.

**Key Insight for TRELLIS.2**: LaFiTe's decoupling of texture from mesh connectivity and UV coordinates addresses TRELLIS.2's UV fragmentation bottleneck (currently 65.7/100). If texture is defined continuously in 3D and only discretized at the final UV baking step, many UV artifacts vanish.

**Feasibility**: Medium-Hard (need to adapt to TRELLIS.2's output format)
**Expected Impact**: Very High (+5-8 points, especially on fragmentation)
**Code Available**: Not yet

### 2.3 TEXTRIX: Latent Attribute Grid for Native Texture (Dec 2025)

**Paper**: "TEXTRIX: Latent Attribute Grid for Native Texture Generation and Beyond"
**ArXiv**: https://arxiv.org/abs/2512.02993

**Key Technique**: Constructs a latent 3D attribute grid and uses a Diffusion Transformer with sparse attention for direct volumetric coloring. Avoids multi-view fusion entirely.

**Relevance to TRELLIS.2**: TEXTRIX's architecture is strikingly similar to TRELLIS.2's texture SLAT stage -- both use sparse 3D grids with transformer-based generation. The key difference is TEXTRIX's sparse attention mechanism, which may be more efficient than TRELLIS.2's full attention. Also extended to 3D segmentation, suggesting the representation is rich enough for semantic understanding.

**Key Connection**: TEXTRIX powers DreamTech's Neural4D-2.5 (announced Feb 2026), which delivers "industrial-grade PBR textures." This validates the approach at production scale.

**Feasibility**: Hard (requires architecture modifications or separate pipeline)
**Expected Impact**: High (+3-5 points)
**Code Available**: Not yet

### 2.4 UniTEX: Universal Texturing via Texture Functions (May 2025)

**Paper**: "UniTEX: Universal High Fidelity Generative Texturing for 3D Shapes"
**ArXiv**: https://arxiv.org/abs/2505.23253
**Code**: https://github.com/lightillusions/UniTEX

**Key Technique**: Bypasses UV mapping by operating in 3D functional space via Texture Functions (TFs) -- continuous, volumetric representations mapping any 3D point to texture values based on surface proximity. Pipeline:
1. Generate high-fidelity multi-view images (RGB generation, delighting, super-resolution) using fine-tuned DiTs
2. Reproject to partial textured mesh
3. Large Texturing Model predicts complete texture functions
4. Final texture synthesized by blending predicted TFs with partial geometry

**Key Innovation**: Mixture-of-LoRA strategy for efficiently adapting large-scale DiTs for multi-view texture synthesis across modalities.

**Relevance to TRELLIS.2**: UniTEX's delighting step (removing lighting effects from generated views) is directly applicable. TRELLIS.2's textures sometimes bake in lighting from the input image. UniTEX's 3-step image processing (RGB + delight + super-res) could be applied to multi-view renders of TRELLIS.2 output before reprojection.

**Feasibility**: Medium (modular pipeline, each step can be tested independently)
**Expected Impact**: High (+3-5 points on color accuracy and detail)
**Code Available**: Yes

### 2.5 MVPainter: Geometric Control + PBR (May 2025)

**Paper**: "MVPainter: Accurate and Detailed 3D Texture Generation via Multi-View Diffusion with Geometric Control"
**ArXiv**: https://arxiv.org/abs/2505.12635
**Code**: https://github.com/amap-cvlab/MV-Painter

**Key Technique**: Systematically addresses three dimensions of texture quality:
1. **Reference-texture alignment** -- data filtering and augmentation strategies
2. **Geometry-texture consistency** -- ControlNet-based geometric conditioning
3. **Local texture quality** -- enhanced local detail generation

Extracts PBR attributes from generated views to produce physically-based meshes.

**Relevance to TRELLIS.2**: MVPainter's ControlNet geometric conditioning (using depth/normal maps from the mesh) ensures textures conform to geometry. This could be used as a post-processing step: render TRELLIS.2's untextured mesh from multiple views, generate textures with MVPainter, project back. The PBR extraction is also valuable since TRELLIS.2's PBR quality is limited.

**Feasibility**: Medium (open-source, modular pipeline)
**Expected Impact**: High (+3-5 points on texture quality and PBR)
**Code Available**: Yes

### 2.6 MaterialMVP: Illumination-Invariant PBR (ICCV 2025)

**Paper**: "MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion"
**Venue**: ICCV 2025
**ArXiv**: https://arxiv.org/abs/2503.10289
**Code**: https://github.com/ZebinHe/MaterialMVP

**Key Technique**: Three innovations for PBR texture generation:
1. **Reference Attention** -- extracts and encodes informative latent from input images for controllable texture
2. **Consistency-Regularized Training** -- enforces stability across viewpoints and illumination conditions
3. **Dual-Channel Material Generation** -- separately optimizes albedo and metallic-roughness with Multi-Channel Aligned Attention for spatial consistency

**Relevance to TRELLIS.2**: TRELLIS.2's biggest PBR issue is that albedo textures bake in environment lighting from the input image (violating the PBR assumption that albedo should be lighting-independent). MaterialMVP's illumination-invariant design directly addresses this. As a post-processor, it could take TRELLIS.2's geometry and input image, and generate clean PBR materials with proper albedo/roughness/metallic separation.

**Key Insight**: The dual-channel approach (separate albedo vs material) is important because albedo and roughness/metallic have very different frequency characteristics and should not be generated by the same process.

**Feasibility**: Medium (open-source with weights)
**Expected Impact**: High (+3-5 points on color accuracy, +2-3 on material quality)
**Code Available**: Yes

### 2.7 TEXGen: Feed-Forward UV Texture Diffusion (SIGGRAPH Asia 2024, Best Paper HM)

**Paper**: "TEXGen: a Generative Diffusion Model for Mesh Textures"
**Venue**: SIGGRAPH Asia 2024 (Best Paper Honorable Mention)
**ArXiv**: https://arxiv.org/abs/2411.14740
**Code**: https://github.com/CVMI-Lab/TEXGen

**Key Technique**: 700M parameter diffusion model that generates UV texture maps directly in UV domain. Architecture interleaves convolutions on UV maps with attention layers on point clouds. Supports text-guided inpainting, sparse-view completion, and text-driven synthesis.

**Relevance to TRELLIS.2**: TEXGen operates on UV maps directly, complementary to TRELLIS.2's voxel-based approach. Could serve as a texture refinement step: take TRELLIS.2's initial UV texture map, refine it with TEXGen using the input image as conditioning. The UV-native design avoids multi-view projection artifacts.

**Feasibility**: Medium (pre-trained model, straightforward integration)
**Expected Impact**: Medium-High (+2-4 points on texture detail)
**Code Available**: Yes

### 2.8 Make-A-Texture: Fast 6-View Texturing (WACV 2025)

**Paper**: "Make-A-Texture: Fast Shape-Aware Texture Generation in 3 Seconds"
**Venue**: WACV 2025
**ArXiv**: https://arxiv.org/abs/2412.07766

**Key Technique**: Generates textures using only 6 automatically-selected optimal viewpoints (vs 10-30 in previous methods). Uses depth-aware inpainting diffusion model with selective masking of non-frontal and internal faces. End-to-end runtime of 3.07 seconds on H100.

**Relevance to TRELLIS.2**: The automatic viewpoint selection algorithm is applicable to TRELLIS.2's multi-view texture projection. Currently, TRELLIS.2 uses fixed viewpoints for multi-view rendering. Make-A-Texture's approach of selecting object-specific optimal viewpoints could improve coverage and reduce texture artifacts in occluded regions.

**Feasibility**: Easy-Medium (viewpoint selection is independent of generation method)
**Expected Impact**: Low-Medium (+1-2 points from better view coverage)
**Code Available**: Not yet

### 2.9 2.5D Latents: Unified RGB-Normal-Coordinate Generation (May 2025)

**Paper**: "Advancing high-fidelity 3D and Texture Generation with 2.5D latents"
**ArXiv**: https://arxiv.org/abs/2505.21050

**Key Technique**: Creates 2.5D latents that integrate multiview RGB, normal, and coordinate images into a unified representation. A lightweight 2.5D-to-3D refiner-decoder converts these to detailed 3D. Uses Mixture-of-LoRA architecture for multi-modality generation. Enables partial generation for geometry-conditioned texture generation.

**Relevance to TRELLIS.2**: The 2.5D latent concept bridges the 2D-3D gap differently from TRELLIS.2's pure 3D voxel approach. For texture refinement, the approach of jointly generating RGB + normals + coordinates ensures geometric consistency in the generated textures. Could be used to generate high-quality multi-view reference images for TRELLIS.2's texture optimization.

**Feasibility**: Medium-Hard (requires adapted DiT models)
**Expected Impact**: Medium-High (+2-4 points)
**Code Available**: Yes (stated in paper)

### 2.10 Hunyuan3D 2.5 / LATTICE (June 2025)

**Paper**: "Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details"
**ArXiv**: https://arxiv.org/abs/2506.16504
**Code**: https://github.com/Tencent-Hunyuan/Hunyuan3D-2

**Key Improvements over 2.0/2.1**:
1. **LATTICE shape model**: 10B parameter shape foundation model with VoxSet representation (sparse voxels + fixed-length latent bottleneck)
2. **Geometric resolution 1024** (up from 512), effective facet count increased 10x
3. **PBR Paint model**: Extended multi-view architecture for PBR texture generation
4. **Experimental normal map module**: Improved surface detail rendering
5. **Performance**: CLIP score 0.821 (up from 0.809), 25% latency reduction

**Relevance to TRELLIS.2**: Hunyuan3D 2.5's VoxSet representation is instructive -- it uses sparse voxels with a fixed-length latent bottleneck, avoiding TRELLIS.2's variable-length latent issue that requires a two-stage pipeline. The 1024 geometric resolution shows the frontier of what's achievable with scaled sparse voxel models. The normal map module is also relevant for TRELLIS.2's normal baking.

**Feasibility**: Informational / Hard (architectural changes)
**Expected Impact**: Benchmark comparison only
**Code Available**: Yes (models available)

### 2.11 TexVerse: High-Resolution Texture Dataset (Aug 2025)

**Paper**: "TexVerse: A Universe of 3D Objects with High-Resolution Textures"
**ArXiv**: https://arxiv.org/abs/2508.10868
**Code**: https://github.com/yiboz2001/TexVerse

**Key Contribution**: 858K unique high-resolution 3D models (1.6M instances including variants) with textures at 1024+ resolution, 158K with PBR materials. Preserves 4096 and 8192 texture variants that Objaverse downsample.

**Relevance to TRELLIS.2**: Valuable as training data for fine-tuning texture quality metrics, texture super-resolution models, or PBR estimation networks. The PBR subset (158K) could train a TRELLIS.2 texture quality estimator.

**Feasibility**: Easy (data download for training/evaluation)
**Expected Impact**: Indirect (enables training of enhancement models)
**Code Available**: Yes (dataset + tools)

### 2.12 Summary Table: Texture Methods

| Method | Year | Venue | Technique | Feasibility | Impact | Code |
|--------|------|-------|-----------|-------------|--------|------|
| NaTex | 2025 | arXiv | 3D color point cloud + DiT | Medium | High | Yes |
| LaFiTe | 2025 | arXiv | Latent color field + rectified flow | Med-Hard | Very High | No |
| TEXTRIX | 2025 | arXiv | Latent attribute grid + sparse DiT | Hard | High | No |
| UniTEX | 2025 | arXiv | Texture Functions + LoRA DiT | Medium | High | Yes |
| MVPainter | 2025 | arXiv | ControlNet geom conditioning | Medium | High | Yes |
| MaterialMVP | 2025 | ICCV | Illumination-invariant dual PBR | Medium | High | Yes |
| TEXGen | 2024 | SIG Asia | UV-native 700M diffusion | Medium | Med-High | Yes |
| Make-A-Texture | 2025 | WACV | 6-view optimal selection | Easy-Med | Low-Med | No |
| 2.5D Latents | 2025 | arXiv | RGB+Normal+Coord unified | Med-Hard | Med-High | Yes |
| Hunyuan3D 2.5 | 2025 | arXiv | LATTICE 10B + PBR Paint | Informational | Benchmark | Yes |
| TexVerse | 2025 | arXiv | 858K hi-res 3D dataset | Easy | Indirect | Yes |

---

## 3. Geometry Detail Enhancement

### 3.1 SparseFlex (TripoSF): High-Resolution Sparse Isosurface (ICCV 2025)

**Paper**: "SparseFlex: High-Resolution and Arbitrary-Topology 3D Shape Modeling"
**Venue**: ICCV 2025
**ArXiv**: https://arxiv.org/abs/2503.21732
**Code**: https://github.com/VAST-AI-Research/TripoSF

**Key Technique**: Novel sparse-structured isosurface representation enabling differentiable mesh reconstruction at resolutions up to 1024^3. Combines FlexiCubes accuracy with sparse voxel structure. Features:
1. Sparse voxel focus on surface-adjacent regions only
2. Frustum-aware sectional voxel training activates only relevant voxels during rendering
3. Self-pruning upsampling for progressive resolution increase
4. Natively supports arbitrary topology (open surfaces, closed solids)

**Quantitative Results**: 82% reduction in Chamfer Distance, 88% increase in F-score vs previous methods.

**Relevance to TRELLIS.2**: TRELLIS.2 uses FlexiDualGrid (FDG) for mesh extraction from its sparse voxel representation. SparseFlex/TripoSF is a direct upgrade path -- it uses the same sparse voxel foundation but achieves dramatically better mesh quality at higher resolution. The frustum-aware training is particularly relevant for TRELLIS.2's memory constraints on DGX Spark.

**Key Insight**: SparseFlex's VAE takes point clouds, voxelizes them, applies sparse transformer encoder-decoder, then self-pruning upsampling to higher resolution, and finally decodes to SparseFlex representation. This progressive upsampling approach directly addresses TRELLIS.2's resolution bottleneck.

**Feasibility**: Hard (requires replacing FDG mesh extraction; model retraining)
**Expected Impact**: Very High (+5-8 points on geometry, potentially transformative)
**Code Available**: Yes

### 3.2 TetWeave: Adaptive Delaunay Mesh Extraction (SIGGRAPH 2025)

**Paper**: "TetWeave: Isosurface Extraction using On-The-Fly Delaunay Tetrahedral Grids for Gradient-Based Mesh Optimization"
**Venue**: SIGGRAPH 2025 (Journal Paper)
**ArXiv**: https://arxiv.org/abs/2505.04590
**Code**: https://github.com/AlexandreBinninger/TetWeave

**Key Technique**: Novel isosurface extraction that jointly optimizes:
1. Placement of tetrahedral grid via Delaunay triangulation (on-the-fly, not predefined)
2. Directional signed distance at each point
3. Adaptive resampling -- places new points where reconstruction error is high

**Guarantees**: Watertight, two-manifold, intersection-free meshes.
**Memory**: Near-linear scaling relative to output vertex count (major improvement over fixed grids).

**Relevance to TRELLIS.2**: TetWeave could replace TRELLIS.2's current FDG mesh extraction as a differentiable isosurface extractor. The adaptive resampling is critical -- it allocates mesh resolution where geometric detail is highest (e.g., facial features) rather than uniformly. The manifold guarantees eliminate the topology artifacts that FDG sometimes produces.

**Key Advantage over FDG**: FDG uses a predefined dual grid; TetWeave's on-the-fly Delaunay construction adapts to the actual geometry, producing higher quality meshes with fewer parameters.

**Feasibility**: Hard (requires replacing mesh extraction pipeline)
**Expected Impact**: High (+3-5 points on geometry quality)
**Code Available**: Yes

### 3.3 UltraShape 1.0: Scalable Geometric Refinement (Dec 2025)

**Paper**: "UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement"
**ArXiv**: https://arxiv.org/abs/2512.21185
**Code**: https://github.com/PKU-YuanGroup/UltraShape-1.0

**Key Technique**: Two-stage generation with voxel-based refinement:
1. Coarse global structure synthesis
2. Fine-grained geometry refinement at fixed spatial locations using voxel queries with RoPE positional encoding

Data processing pipeline includes watertight processing, hole filling, thin structure thickening, and quality filtering.

**Relevance to TRELLIS.2**: UltraShape's two-stage approach mirrors TRELLIS.2's sparse structure -> shape SLAT pipeline. The key difference is UltraShape's explicit geometric refinement stage with RoPE-encoded voxel queries. This refinement approach could be added as a Stage 2.5 between TRELLIS.2's shape SLAT and texture SLAT, using the shape output as coarse geometry and refining with a small transformer.

**Feasibility**: Medium-Hard (requires training a refinement module)
**Expected Impact**: High (+3-5 points on geometry detail)
**Code Available**: Yes

### 3.4 SpArSe-Up: Learnable Sparse Upsampling (Sep 2025)

**Paper**: "Sparse-Up: Learnable Sparse Upsampling for 3D Generation with High-Fidelity Textures"
**ArXiv**: https://arxiv.org/abs/2509.23646

**Key Technique**: Two strategies for breaking voxel resolution limits:
1. **Surface Anchoring**: Learnable upsampling that constrains new voxels to the mesh surface, eliminating 70%+ of redundant off-surface voxels that traditional upsampling creates
2. **View-Domain Partitioning**: Image patch-guided voxel partitioning scheme that supervises only visible local patches, reducing memory

**Key Finding**: Texture quality exhibits "significant positive growth" with voxel resolution increase, reaching optimal at 512-resolution. Higher resolution directly captures more detailed texture features.

**Relevance to TRELLIS.2**: This is directly applicable to TRELLIS.2's architecture. TRELLIS.2 currently operates at fixed voxel resolution in each stage. SpArSe-Up's surface anchoring could be applied after TRELLIS.2's shape SLAT to upsample the sparse voxel representation before texture generation, concentrating new voxels at the surface where texture detail matters most.

**Critical Insight**: The 70% voxel reduction from surface anchoring means TRELLIS.2 could potentially double its effective texture resolution within the same GPU memory budget.

**Feasibility**: Medium (requires adapting upsampling to TRELLIS.2's sparse structure format)
**Expected Impact**: Very High (+5-8 points on texture detail and fragmentation)
**Code Available**: Not yet

### 3.5 HVPUNet: Hybrid Voxel Point Cloud Upsampling (ICCV 2025)

**Paper**: "HVPUNet: Hybrid-Voxel Point-cloud Upsampling Network"
**Venue**: ICCV 2025

**Key Technique**: Two-module framework:
1. Shape Completion Module -- fills empty voxels to restore missing geometry
2. Super-Resolution Module -- enhances spatial resolution for finer surface details

Uses "Hybrid Voxels" combining voxel occupancy with continuous point offsets to resolve the efficiency-precision trade-off. Progressive refinement with operational voxel expansion and implicit learning.

**Relevance to TRELLIS.2**: The shape completion module could fill holes in TRELLIS.2's sparse structure output. The super-resolution module could add geometric detail to the simplified mesh. The hybrid voxel concept (discrete occupancy + continuous offset) is directly applicable to TRELLIS.2's sparse voxel representation.

**Feasibility**: Medium (pre-trained model, adapt to TRELLIS.2 output format)
**Expected Impact**: Medium (+2-3 points on geometry)
**Code Available**: Limited

### 3.6 VertexRegen: Progressive Mesh Generation (ICCV 2025)

**Paper**: "VertexRegen: Mesh Generation with Continuous Level of Detail"
**Venue**: ICCV 2025
**ArXiv**: https://arxiv.org/abs/2508.09062
**Code**: https://github.com/zx1239856/VertexRegen (re-implementation)

**Key Technique**: Reformulates mesh generation as the reversal of edge collapse (vertex split), modeled by a Transformer with next-token prediction. Produces meshes of varying detail levels -- generation can be stopped at any step to yield valid meshes at any LOD.

**Relevance to TRELLIS.2**: VertexRegen could replace TRELLIS.2's current mesh decimation step. Instead of extracting a high-face-count mesh and then decimating it (losing detail), VertexRegen could directly generate a mesh at the target detail level from TRELLIS.2's voxel output, preserving important features.

**Feasibility**: Medium (replace post-extraction decimation step)
**Expected Impact**: Medium (+1-3 points on geometry preservation)
**Code Available**: Yes (re-implementation)

### 3.7 Meshtron: 64K Faces at 1024 Resolution (Dec 2024)

**Paper**: "Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale"
**ArXiv**: https://arxiv.org/abs/2412.09548

**Key Technique**: Autoregressive model generating meshes up to 64K faces at 1024-level coordinate resolution (10x+ faces and 8x resolution vs previous methods). Uses Hourglass Transformer with three-stage sequence reduction (coordinate -> vertex -> face), achieving 2.5x faster throughput and 50% memory savings. Accepts inputs: point clouds, face count, quad ratio, creativity level.

**Relevance to TRELLIS.2**: Meshtron could serve as a post-generation retopology step, converting TRELLIS.2's FDG mesh output to clean artist-quality topology. The 64K face limit is practical for most objects. The controllable face count and quad ratio are valuable for production pipelines.

**Key Advantage over MeshAnything V2**: 40x higher face count (64K vs 1600), 8x higher coordinate resolution.

**Feasibility**: Medium (pre-trained model, post-processing integration)
**Expected Impact**: Medium-High (+2-4 points on geometry quality)
**Code Available**: Not yet

### 3.8 MeshCone: Convex Optimization Mesh Refinement (Dec 2024)

**Paper**: "MeshCone: Second-Order Cone Programming for Geometrically-Constrained Mesh Enhancement"
**ArXiv**: https://arxiv.org/abs/2412.08484

**Key Technique**: Formulates mesh refinement as second-order cone program (SOCP). Vertex positions optimized to align with reference geometry while enforcing smoothness through convex edge-length regularization. Runtime: ~20 seconds per mesh. Improvements: 7.9-31.5% reconstruction error reduction across test objects.

**Relevance to TRELLIS.2**: MeshCone could be applied as a fast geometry cleanup step after FDG mesh extraction. The convex optimization is guaranteed to converge and runs in seconds, making it practical for a post-processing pipeline. Particularly useful for smoothing irregular triangles from FDG while preserving overall shape.

**Limitation**: Can over-smooth localized high-frequency details due to global optimization nature.

**Feasibility**: Easy (standalone optimizer, 20-second runtime)
**Expected Impact**: Low-Medium (+1-2 points on geometry smoothness)
**Code Available**: Not specified

### 3.9 TetraSDF: Precise Analytic Mesh Extraction (Nov 2025)

**Paper**: "TetraSDF: Precise Mesh Extraction with Multi-resolution Tetrahedral Grid"
**ArXiv**: https://arxiv.org/abs/2511.16273

**Key Technique**: Analytic meshing framework for SDFs using ReLU MLP + multi-resolution tetrahedral positional encoder. Tracks ReLU linear regions for exact isosurface extraction. Produces highly self-consistent meshes faithful to learned isosurfaces.

**Relevance to TRELLIS.2**: Could replace FDG mesh extraction with a more precise analytic approach. The multi-resolution tetrahedral encoding is particularly relevant -- it provides progressive detail refinement similar to MS3D but at the mesh extraction stage rather than the generation stage.

**Feasibility**: Hard (requires SDF formulation of TRELLIS.2's voxel output)
**Expected Impact**: Medium-High (+2-4 points)
**Code Available**: Not specified

### 3.10 SuperPC: Unified Point Cloud Enhancement (CVPR 2025)

**Paper**: "SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization"
**Venue**: CVPR 2025
**ArXiv**: https://arxiv.org/abs/2503.14558

**Key Technique**: Unified diffusion model handling completion, upsampling, denoising, and colorization simultaneously. Three-level conditioned diffusion with spatial-mix-fusion strategy. Outperforms specialized models on each individual task.

**Relevance to TRELLIS.2**: Could be applied to TRELLIS.2's intermediate point cloud representation (between voxels and mesh). Upsampling would add geometric detail, completion would fill holes, denoising would smooth artifacts. The colorization capability is a bonus for texture quality.

**Feasibility**: Medium (adapt to TRELLIS.2's voxel-to-point-cloud conversion)
**Expected Impact**: Medium (+2-3 points across multiple dimensions)
**Code Available**: Yes

### 3.11 Summary Table: Geometry Methods

| Method | Year | Venue | Technique | Feasibility | Impact | Code |
|--------|------|-------|-----------|-------------|--------|------|
| SparseFlex | 2025 | ICCV | 1024^3 sparse isosurface | Hard | Very High | Yes |
| TetWeave | 2025 | SIGGRAPH | Adaptive Delaunay extraction | Hard | High | Yes |
| UltraShape | 2025 | arXiv | Voxel-based geometric refinement | Med-Hard | High | Yes |
| SpArSe-Up | 2025 | arXiv | Surface-anchored upsampling | Medium | Very High | No |
| HVPUNet | 2025 | ICCV | Hybrid voxel upsampling | Medium | Medium | Limited |
| VertexRegen | 2025 | ICCV | Progressive vertex split | Medium | Medium | Yes |
| Meshtron | 2024 | arXiv | 64K-face autoregressive | Medium | Med-High | No |
| MeshCone | 2024 | arXiv | SOCP mesh refinement | Easy | Low-Med | Unknown |
| TetraSDF | 2025 | arXiv | Analytic multi-res extraction | Hard | Med-High | Unknown |
| SuperPC | 2025 | CVPR | Unified PC enhancement | Medium | Medium | Yes |

---

## 4. Post-Generation Refinement

### 4.1 GTR: Geometry and Texture Refinement (ICLR 2025)

**Paper**: "GTR: Improving Large 3D Reconstruction Models through Geometry and Texture Refinement"
**Venue**: ICLR 2025
**ArXiv**: https://arxiv.org/abs/2406.05649
**Code**: https://snap-research.github.io/GTR/

**Key Technique**: Three-component refinement for large reconstruction models:
1. **Architecture fixes** for LRM: improved multi-view image representation, more efficient training
2. **Differentiable mesh extraction** from NeRF + fine-tuning through mesh rendering at full resolution
3. **Per-instance texture refinement**: fine-tunes triplane representation and color estimation on mesh surface using input multi-view images. **Runtime: 4 seconds.**

**Quantitative**: PSNR improves to 29.79 with texture refinement. Faithful reconstruction of complex textures including text and portraits.

**Relevance to TRELLIS.2**: GTR's per-instance texture refinement is the most directly applicable finding. Instead of a generic model, GTR optimizes the texture for each specific input at inference time, taking only 4 seconds. The approach: render the mesh from input viewpoints, compute reconstruction loss against input images, backpropagate to update texture. This is essentially the "render-and-compare" approach (Section 2.6 of previous survey) validated at production scale.

**Key Insight**: GTR validates that per-instance optimization is fast enough for practical use (4 seconds). TRELLIS.2 could add this as a final refinement step after UV baking.

**Feasibility**: Medium (nvdiffrast already available; need camera matrices and optimization loop)
**Expected Impact**: High (+3-5 points on color fidelity and detail)
**Code Available**: Yes

### 4.2 Elevate3D: HFS-SDEdit Co-Refinement (SIGGRAPH 2025)

**Paper**: "Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model"
**Venue**: SIGGRAPH 2025
**ArXiv**: https://arxiv.org/abs/2507.11465

**Key Technique**: High-Frequency Separated SDEdit (HFS-SDEdit):
1. Diffusion freely generates low-frequency content while constraining high-frequency detail from the input
2. Monocular geometry predictors derive normal/depth cues from enhanced images
3. Iterative texture-geometry co-refinement loop

**Relevance to TRELLIS.2**: (Already in previous survey, Section 1.7) Remains highly relevant. The frequency separation idea is key: TRELLIS.2's textures often have correct low-frequency structure but lack high-frequency detail. HFS-SDEdit could inject sharp details from the input image while preserving the 3D-consistent low-frequency content.

**Update since last survey**: Paper accepted at SIGGRAPH 2025, confirming quality. Code expected with publication.

**Feasibility**: Medium-Hard
**Expected Impact**: High (+3-6 points)

### 4.3 Summary Table: Post-Generation Methods

| Method | Year | Venue | Technique | Feasibility | Impact | Code |
|--------|------|-------|-----------|-------------|--------|------|
| GTR | 2025 | ICLR | 4-second per-instance refinement | Medium | High | Yes |
| Elevate3D | 2025 | SIGGRAPH | HFS-SDEdit co-refinement | Med-Hard | High | Expected |
| (Render-and-Compare) | Various | - | nvdiffrast optimization | Medium | Very High | Infrastructure ready |

---

## 5. Flow Matching and Sampling Improvements

### 5.1 OC-Flow: Optimal Control for Guided Flow Matching (ICLR 2025)

**Paper**: "Training Free Guided Flow Matching with Optimal Control"
**Venue**: ICLR 2025
**ArXiv**: https://arxiv.org/abs/2410.18070
**Code**: https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control

**Key Technique**: Formulates guided flow matching as an optimal control problem. Key features:
1. First training-free approach with proven convergence in Euclidean space
2. Extended Method of Successive Approximations (E-MSA) for solving the control problem
3. Existing backprop-through-ODE methods shown to be special cases of OC-Flow
4. Unified theoretical framework for understanding all guided flow matching approaches

**Relevance to TRELLIS.2**: OC-Flow provides a theoretically grounded replacement for the ad-hoc CFG used in TRELLIS.2's flow matching stages. Instead of simple CFG (which has been shown to cause off-manifold drift), OC-Flow uses optimal control theory to compute the guidance direction, with convergence guarantees. This is a more principled approach than CFG-MP (manifold projection), which only approximately corrects drift.

**Key Difference from CFG-MP**: CFG-MP projects back to the manifold post-hoc; OC-Flow computes guidance that stays on the manifold by construction.

**Feasibility**: Medium (requires implementing E-MSA solver in TRELLIS.2's sampling loop)
**Expected Impact**: Medium-High (+2-4 points, especially at high guidance strengths where off-manifold drift is worst)
**Code Available**: Yes

### 5.2 Re-Meanflow: Efficient One-Step Generation (2025)

**Paper**: "Flow Straighter and Faster: Efficient One-Step Generative Modeling via MeanFlow on Rectified Trajectories"
**ArXiv**: https://arxiv.org/abs/2511.23342

**Key Technique**: Combines rectified flow trajectory straightening with Meanflow's one-step generation:
1. Single reflow step straightens trajectories sufficiently
2. Meanflow model trained on straightened trajectories
3. Achieves competitive one-step generation with 90% training cost reduction

**Relevance to TRELLIS.2**: If Re-Meanflow could be applied to TRELLIS.2's flow models, it would enable one-step generation, reducing texture generation from 16 Euler steps to 1. This would free up compute budget for higher resolution or more refinement iterations. However, requires retraining.

**Feasibility**: Hard (requires retraining with Meanflow objective)
**Expected Impact**: Indirect (speed enables more quality-improving refinement)
**Code Available**: Not yet

### 5.3 CFG-MP Update: Validated on More Models (Jan 2025)

**Paper**: "Improving Classifier-Free Guidance of Flow Matching via Manifold Projection"
**ArXiv**: https://arxiv.org/abs/2601.21892

**Status Update**: Previously surveyed. Now validated on DiT-XL-2-256, Flux, and Stable Diffusion 3.5. Our implementation tested at strengths 0.05-0.5, with 0.15 showing marginal +1.0 improvement using the corrected conditional-only pred_x_0.

**Remaining Question**: Our implementation may still not capture the full benefit. The paper uses Anderson Acceleration for efficiency, which we haven't implemented. The acceleration avoids extra NFE while improving convergence.

### 5.4 Summary Table: Flow Matching Methods

| Method | Year | Venue | Technique | Feasibility | Impact | Code |
|--------|------|-------|-----------|-------------|--------|------|
| OC-Flow | 2025 | ICLR | Optimal control guidance | Medium | Med-High | Yes |
| Re-Meanflow | 2025 | arXiv | One-step rectified flow | Hard | Indirect | No |
| CFG-MP | 2025 | arXiv | Manifold projection (updated) | Implemented | Low-Med | Yes |

---

## 6. Cross-Cutting Themes and Patterns

### 6.1 The Native 3D Texture Revolution

A major trend in late 2025 is the shift from multi-view fusion to native 3D texture generation:

| Method | Representation | UV Needed? | Seam-Free? |
|--------|---------------|------------|------------|
| NaTex | Color point cloud | No | Yes |
| LaFiTe | Latent color field | No | Yes |
| TEXTRIX | Latent attribute grid | No | Yes |
| UniTEX | Texture functions | No | Yes |
| TEXGen | UV map (direct) | Yes | Partial |

**Implication for TRELLIS.2**: TRELLIS.2's texture SLAT already operates in 3D voxel space -- it is positioned to benefit from this trend. The bottleneck is the final UV baking step, which introduces the seam artifacts that these new methods avoid. Two actionable paths:

1. **Short-term**: Keep current pipeline but add UV seam smoothing post-processing
2. **Medium-term**: Replace UV baking with a continuous 3D texture representation (like LaFiTe's latent field or NaTex's color point cloud) that can be rendered at arbitrary resolution without UV mapping

### 6.2 Surface-Anchored Sparse Voxels

Multiple papers (SpArSe-Up, SparseFlex, TEXTRIX) converge on the insight that sparse voxels should be concentrated at the object surface, not distributed volumetrically. This eliminates 70%+ of redundant computation and memory.

**Implication for TRELLIS.2**: TRELLIS.2's sparse structure stage already identifies occupied voxels, but the shape/texture SLATs process all of them. A surface-anchored approach would allocate more voxels to surface regions and fewer to the interior, improving texture detail where it matters.

### 6.3 Per-Instance Optimization is Fast Enough

GTR validates that per-instance texture optimization takes only 4 seconds -- fast enough for production use. Combined with nvdiffrast (already installed in TRELLIS.2's Docker container), this makes render-and-compare optimization practical.

**Implication for TRELLIS.2**: Add a 4-second optimization loop after UV baking. Render the mesh from the input viewpoint, compare to input image, backpropagate to texture. This is the single highest-ROI improvement available.

### 6.4 PBR Decomposition is Now a Solved Problem

Multiple methods (MaterialMVP, MVPainter, Material Anything, UniTEX) now achieve high-quality albedo/roughness/metallic decomposition. The key insight is that illumination-invariant training (MaterialMVP) produces far better albedo maps than standard approaches.

**Implication for TRELLIS.2**: TRELLIS.2's PBR output quality is limited by its voxel-based PBR prediction. Using a dedicated PBR decomposition model as a post-processor (MaterialMVP or Material Anything) would significantly improve material quality.

### 6.5 Geometry Resolution is Scaling Up

The frontier has moved from 256^3 to 1024^3 voxel resolution:
- TRELLIS.2: 64^3 sparse structure
- SparseFlex: 1024^3 isosurface
- Hunyuan3D 2.5: 1024 geometric resolution

**Implication for TRELLIS.2**: The 4-16x resolution gap between TRELLIS.2 and current SOTA is the primary geometry quality bottleneck. Addressing this requires either architectural changes (SpArSe-Up-style upsampling) or post-generation refinement (DetailGen3D, UltraShape).

---

## 7. Updated Prioritized Roadmap for TRELLIS.2

### Phase A: Immediate Wins (1-3 days, no retraining)

| Priority | Method | Expected Impact | Effort | Notes |
|----------|--------|----------------|--------|-------|
| A1 | **GTR-style per-instance texture optimization** | +3-5 pts | 2-3 days | nvdiffrast ready, 4-second runtime per object |
| A2 | **OC-Flow optimal control guidance** | +2-4 pts | 1-2 days | ICLR 2025, code available, theoretically grounded |
| A3 | **MeshCone geometry refinement** | +1-2 pts | 0.5 days | SOCP solver, 20-second runtime |

### Phase B: Medium Effort (1-2 weeks, no retraining)

| Priority | Method | Expected Impact | Effort | Notes |
|----------|--------|----------------|--------|-------|
| B1 | **NaTex or MVPainter texture re-generation** | +3-5 pts | 1 week | Replace/augment texture with dedicated model |
| B2 | **MaterialMVP PBR refinement** | +3-5 pts | 1 week | Illumination-invariant albedo decomposition |
| B3 | **UniTEX delighting + super-res pipeline** | +3-5 pts | 1 week | 3-step: delight, enhance, reproject |
| B4 | **SpArSe-Up surface anchoring** | +5-8 pts | 1-2 weeks | Learnable voxel upsampling, 70% memory savings |

### Phase C: Major Integration (2-4 weeks)

| Priority | Method | Expected Impact | Effort | Notes |
|----------|--------|----------------|--------|-------|
| C1 | **SparseFlex mesh extraction** | +5-8 pts | 2-3 weeks | Replace FDG with 1024^3 capable extractor |
| C2 | **Cycle3D generation-reconstruction loop** | +3-5 pts | 2-3 weeks | Multi-round texture refinement |
| C3 | **LaFiTe continuous texture field** | +5-8 pts | 3-4 weeks | Bypass UV entirely |

### Phase D: Architectural Overhaul (months, retraining required)

| Priority | Method | Expected Impact | Effort | Notes |
|----------|--------|----------------|--------|-------|
| D1 | **LATTICE-style VoxSet representation** | +5-10 pts | Months | Fixed-length latent, single-stage pipeline |
| D2 | **TEXTRIX sparse attention DiT** | +5-10 pts | Months | Native 3D texture generation |
| D3 | **Re-Meanflow one-step generation** | Speed | Months | Enables more refinement budget |

---

## 8. Paper Reference List (New Papers Only)

### Multi-View Consistency (Section 1)
1. Cycle3D (AAAI 2025) -- https://arxiv.org/abs/2407.19548
2. Seamless3D (2025) -- https://dl.acm.org/doi/10.1145/3746278.3759391
3. 3D-Adapter (ICLR 2025) -- https://arxiv.org/abs/2410.18974
4. MV-Adapter (ICLR 2025) -- https://openreview.net/forum?id=kcmK2utDhu

### High-Resolution Texture (Section 2)
5. NaTex (Nov 2025) -- https://arxiv.org/abs/2511.16317
6. LaFiTe (Dec 2025) -- https://arxiv.org/abs/2512.04786
7. TEXTRIX (Dec 2025) -- https://arxiv.org/abs/2512.02993
8. UniTEX (May 2025) -- https://arxiv.org/abs/2505.23253
9. MVPainter (May 2025) -- https://arxiv.org/abs/2505.12635
10. MaterialMVP (ICCV 2025) -- https://arxiv.org/abs/2503.10289
11. TEXGen (SIGGRAPH Asia 2024) -- https://arxiv.org/abs/2411.14740
12. Make-A-Texture (WACV 2025) -- https://arxiv.org/abs/2412.07766
13. 2.5D Latents (May 2025) -- https://arxiv.org/abs/2505.21050
14. Hunyuan3D 2.5 (June 2025) -- https://arxiv.org/abs/2506.16504
15. TexVerse (Aug 2025) -- https://arxiv.org/abs/2508.10868

### Geometry Enhancement (Section 3)
16. SparseFlex/TripoSF (ICCV 2025) -- https://arxiv.org/abs/2503.21732
17. TetWeave (SIGGRAPH 2025) -- https://arxiv.org/abs/2505.04590
18. UltraShape 1.0 (Dec 2025) -- https://arxiv.org/abs/2512.21185
19. SpArSe-Up (Sep 2025) -- https://arxiv.org/abs/2509.23646
20. HVPUNet (ICCV 2025) -- ICCV 2025 proceedings
21. VertexRegen (ICCV 2025) -- https://arxiv.org/abs/2508.09062
22. Meshtron (Dec 2024) -- https://arxiv.org/abs/2412.09548
23. MeshCone (Dec 2024) -- https://arxiv.org/abs/2412.08484
24. TetraSDF (Nov 2025) -- https://arxiv.org/abs/2511.16273
25. SuperPC (CVPR 2025) -- https://arxiv.org/abs/2503.14558

### Post-Generation Refinement (Section 4)
26. GTR (ICLR 2025) -- https://arxiv.org/abs/2406.05649
27. Elevate3D (SIGGRAPH 2025) -- https://arxiv.org/abs/2507.11465

### Flow Matching (Section 5)
28. OC-Flow (ICLR 2025) -- https://arxiv.org/abs/2410.18070
29. Re-Meanflow (Nov 2025) -- https://arxiv.org/abs/2511.23342
30. CFG-MP (Jan 2025) -- https://arxiv.org/abs/2601.21892
31. Rectified-CFG++ (CVPR-W 2025) -- https://openreview.net/forum?id=NosdT1FHPv

### Also Noted (Not Fully Surveyed)
32. Sparc3D (2025) -- https://arxiv.org/abs/2505.14521
33. Ultra3D (2025) -- Part attention for efficient 3D generation
34. NeuroDiff3D (2025) -- SDF + deformable tetrahedral refinement
35. DreamComposer++ (2025) -- Multi-view diffusion with conditioning
36. Neural4D-2.5 (Feb 2026) -- DreamTech production system using TEXTRIX

---

## 9. Methodology

### Search Strategy
- Web searches across arXiv, CVPR/ICCV/ICLR/SIGGRAPH/NeurIPS/AAAI 2024-2026 proceedings, GitHub, and HuggingFace
- Focused on papers with open-source implementations or detailed algorithmic descriptions
- Cross-referenced with existing survey (`improvement_methods.md`) to avoid duplication
- Prioritized papers from top venues with quantitative evaluations

### Evaluation Criteria
- **Feasibility**: How much effort to integrate with TRELLIS.2 (Easy < 1 day, Medium 3-7 days, Hard > 1 week)
- **Impact**: Expected quality score improvement based on paper's reported gains, adjusted for TRELLIS.2 context
- **Code**: Whether open-source implementation is available
- **Venue**: Preference for peer-reviewed publications at top venues

### Key Differences from Previous Survey
- Focus on papers published after October 2024 (last survey cutoff)
- Emphasis on methods that have been validated at multiple venues or adopted commercially
- Greater attention to native 3D texture methods (NaTex, LaFiTe, TEXTRIX) -- a paradigm shift not covered in previous survey
- Inclusion of scalable sparse voxel methods (SpArSe-Up, SparseFlex) directly applicable to TRELLIS.2's architecture
- GTR's validation that per-instance optimization is fast enough for production use

### Limitations
- Some papers are very recent (Dec 2025, early 2026) with limited independent validation
- Expected impact estimates extrapolated from papers' own benchmarks, which may not translate to TRELLIS.2's specific outputs
- Some methods require significant GPU memory that may constrain DGX Spark usage when combined
- Native 3D texture methods (NaTex, LaFiTe, TEXTRIX) are architecturally different from TRELLIS.2 and would require substantial integration work
