# Research Changelog

Research diary for TRELLIS.2 3D generation quality optimization.
Uses genetic algorithm principles: keep good parameters, replace bad ones, continuously evolve.

---

## Survey v6: Practical Methods for Texture Coherence (C1) and Color Vitality (C2) (2026-02-22)

### Scope
Targeted survey of actionable methods to improve texture quality in TRELLIS.2's postprocess pipeline. Focused on C1 (texture coherence, 15pts) and C2 (color vitality, 10pts) metrics. Constraint: implementable in 1-2 days with existing infrastructure.

### Key Findings

1. **Root cause of grey patches is trilinear interpolation at sparse voxel boundaries**
   - grid_sample_3d interpolates toward zero when neighbor voxels are empty
   - A surface point with 3/8 empty neighbors gets 37-50% color dilution
   - BVH reprojection helps (41.5% grey without vs 13.8% with) but doesn't eliminate the issue
   - Fix: occupancy-normalized interpolation (divide by occupancy fraction)

2. **cv2.inpaint (Telea, radius=3) is inadequate for UV padding**
   - Averages colors across UV islands, creating muddy seam colors
   - 3px padding insufficient for mipmapping (becomes 1.5px at 2x downsample)
   - Industry standard: 4-8px padding (Substance Painter default: 8px)
   - Fix: push-pull pyramid algorithm that extends each island's color outward

3. **Current grey recovery misses partial desaturation (chroma 18-35)**
   - Threshold of chroma < 18 only catches severe cases
   - Local saturation context needed: a texel with chroma 30 is desaturated if neighbors average 80
   - Fix: local-context chroma deficit detection + adaptive saturation boost

4. **Seam detection by mask erosion is imprecise**
   - Detects ALL mask boundaries, not just UV chart boundaries
   - Over-smooths legitimate texture edges, under-smooths actual seams
   - Fix: use rasterized face ID discontinuities to detect actual seams

5. **TextureRefiner has coordinate system mismatch**
   - Postprocess applies Y/Z swap + Y flip, refiner camera doesn't account for this
   - LPIPS at 256x256 is too low resolution
   - Missing saturation preservation and color histogram losses

### 8 Ranked Methods

| Rank | Method | C1 Impact | C2 Impact | Effort |
|------|--------|-----------|-----------|--------|
| 1 | Push-Pull Texture Padding | +3-5 | +1-2 | 4-6h |
| 2 | Chroma-Aware Sampling/Recovery | +1-2 | +3-5 | 4-8h |
| 3 | UV-Aware Seam Smoothing | +2-4 | 0 | 3-5h |
| 4 | Saturation Boost in HSV | 0 | +2-3 | 1-2h |
| 5 | Enhanced TextureRefiner | +3-5 | +2-4 | 6-10h |
| 6 | Increased Dilation Padding | +1-3 | 0 | 2-3h |
| 7 | Bilateral Denoise | +1-2 | 0 | 2-3h |
| 8 | UV Island Stitching | +2-3 | 0 | 8-12h |

### Recommended Day 1 (highest ROI)
- Saturation Boost (1-2h, C2 +2-3)
- Push-Pull Padding (4-6h, C1 +3-5)
- Chroma-Aware Recovery Solution A (2-3h, C2 +3-5)
- Expected: C1 +4-7pts, C2 +6-10pts

### File
`TRELLIS.2/optimization/research/survey_texture_coherence_vitality_2026_02.md`

---

## Survey v5: Commercial-Grade Mesh Quality Evaluation for 5 Specific Defects (2026-02-22)

### Scope
Targeted survey for detecting 5 specific defects in TRELLIS.2 outputs that our evaluation system currently FAILS to catch: mesh holes, surface peeling/fragmentation, surface roughness on manufactured objects, texture bleeding across UV seams, and grey/desaturated patches. Covers mesh integrity metrics, surface smoothness methods, texture quality analysis, commercial 3D quality standards (TurboSquid CheckMate, game industry), and 10+ new papers.

### Key Findings

1. **Most impactful improvements require ZERO new model downloads**
   - Boundary edge counting, UV fragmentation analysis, dihedral roughness, seam bleeding detection, and spatial grey cluster analysis all use trimesh (already installed) and OpenCV (already installed)
   - Estimated implementation time for all 5: 2-4 hours

2. **Hole detection should use boundary edge analysis, not binary watertight check**
   - Count boundary loops (connected boundary edge chains) to get hole count
   - Measure total boundary length / mesh area for severity
   - Multi-view silhouette flood-fill catches holes visible from each camera angle

3. **Surface peeling is UV fragmentation, not texture noise**
   - Root cause: xatlas creates too many small UV islands
   - Detection: connected component analysis on texture mask reveals island count, size distribution
   - Fragmentation index = n_islands / sqrt(total_texels)
   - Real TRELLIS.2 data: body_count=5631, confirming fragmentation is severe

4. **Surface roughness best measured by dihedral angle distribution**
   - trimesh.face_adjacency_angles provides dihedral angles for all adjacent face pairs
   - Smooth manufactured objects: std(dihedral_angles) < 15 degrees
   - Lavoue (2007) Gaussian curvature roughness: theoretically grounded, available via trimesh vertex_defects
   - Current v4 normal-map Laplacian is view-dependent and misses backside roughness

5. **Texture bleeding detectable via boundary gradient ratio**
   - Compare Sobel gradient magnitude at UV island boundaries vs interior
   - Ratio > 3.0 indicates bleeding; > 5.0 is severe
   - Industry standard: 0.5-1 pixel margin minimum to prevent bleeding

6. **Grey patches need spatial clustering, not just global ratio**
   - Current v4 only counts grey pixels globally (grey_ratio)
   - A single large grey patch is worse than scattered individual grey pixels
   - Connected component analysis on grey pixels gives patch count + largest patch area

7. **TurboSquid CheckMate Pro v2 defines industry topology standards**
   - Grid edge flow, clean subdivision, no unnecessary geometry
   - AI-generated meshes (including TRELLIS.2) fundamentally fail topology requirements
   - UV and texture quality standards ARE achievable and measurable

8. **PCD crack detection (QoMEX 2024) directly addresses peeling**
   - Contrast + Laplacian modules characterize crack artifacts
   - Code available: github.com/arshafiee/crack-detection-VVM
   - Full-reference but crack detection module usable standalone

### New Papers Added (10)
- PCD: Perceptual Crack Detection (QoMEX 2024) -- relevance 8
- Robust Hole Detection (ScienceDirect 2024) -- relevance 7
- HybridMQA: Geometry-Texture QA (arXiv 2024) -- relevance 6
- SeamCrafter: RL UV Seams (arXiv 2025) -- relevance 6
- ArtUV: Visibility-Aware UV (arXiv 2025) -- relevance 6
- FMQM: SDF+Color Field QA (arXiv 2025) -- relevance 5
- GeodesicPSIM: Geodesic Patch Similarity (arXiv 2025) -- relevance 5
- Lavoue Roughness (APGV 2007) -- relevance 7
- FMPD: Fast Mesh Perceptual Distance (CG 2012) -- relevance 6
- Hi3DEval (NeurIPS 2025) -- relevance 8

### New Methods Added (8)
- boundary-edge-hole-detection (easy, high impact)
- uv-fragmentation-analysis (easy, high impact)
- dihedral-roughness-metrics (easy, medium-high impact)
- uv-seam-bleeding-detection (easy, medium-high impact)
- spatial-grey-cluster-detection (easy, medium impact)
- pcd-crack-detection (medium effort, medium impact)
- connected-component-quality (easy, medium impact)
- texel-density-uniformity (medium effort, medium impact)

### Files
- Survey: `TRELLIS.2/optimization/research/survey_mesh_quality_commercial_2026_02.md`
- Papers database updated: v2.0 -> v3.0 (10 new papers, 44 total)
- Methods database updated: v2.0 -> v3.0 (8 new methods, 35 total)

---

## Survey v4: Practical Mesh Quality Tools & Implementation Guide (2026-02-22)

### Scope
Code-level survey of mesh defect detection and quality scoring systems. Covers 6 libraries (trimesh, Open3D, PyVista, PyMeshLab, PyTorch3D, pyiqa), 3 VLM-based evaluation frameworks (GPTEval3D, Hi3DEval, Q-Align), texture/PBR analysis, and industry quality standards.

### Key Findings

1. **trimesh (already installed) provides 80% of mesh quality metrics we need**
   - Topology: is_watertight, is_winding_consistent, euler_number, body_count
   - Face quality: face_angles, area_faces, degenerate detection
   - Edge quality: edges_unique_length, edge length ratio
   - Normal consistency: face_adjacency + face_normals (manual computation)
   - Component analysis: split() for fragmentation detection

2. **Real TRELLIS.2 GLB analysis results (e6_round0_baseline.glb)**
   - NOT watertight (body_count=5631 -- expected for generated meshes)
   - 44 degenerate faces / 490K total (0.01% -- acceptable)
   - 17,937 non-smooth face pairs (2.69% -- moderate, actionable signal)
   - Edge length ratio 120:1 (non-uniform tessellation)
   - Texture: 2048x2048, 98.4% utilization, 1.76% grey texels

3. **pyiqa provides 30+ no-reference image quality metrics via unified API**
   - `pip install pyiqa` then `pyiqa.create_metric('musiq')`
   - Best for our use: MUSIQ, TOPIQ-NR, CLIPIQA+, NIMA
   - All run on GPU, 5-20ms per image

4. **DreamSim (already installed) supersedes LPIPS**
   - Drop-in replacement: `from dreamsim import dreamsim`
   - 96.2% human agreement vs LPIPS 80.2%

5. **Open3D adds self-intersection detection (not in trimesh)**
   - `mesh.is_self_intersecting()` and `mesh.get_self_intersecting_triangles()`
   - Requires `pip install open3d`

6. **Industry game-ready standards provide concrete thresholds**
   - Min angle > 5 deg, max angle < 150 deg
   - No non-manifold edges, consistent normals
   - Textures >= 2048x2048 for hero assets
   - Metallic should be 0 or 1 (not intermediate)

### Recommended 5-Phase Evaluation Pipeline
1. Mesh Integrity (trimesh, binary checks) -- 0 days to implement
2. Geometry Quality (trimesh, continuous scores) -- 0 days
3. Texture Quality (trimesh + numpy + cv2) -- 0 days
4. Rendered View Quality (pyiqa + DreamSim) -- 1 day
5. Input Alignment (open_clip + existing evaluator) -- 1 day

### File
`TRELLIS.2/optimization/research/mesh_quality_tools_survey_2026_02.md`

---

## Survey v3: Quality Evaluation Metrics (2026-02-21)

### Scope
Comprehensive survey of 3D generation quality evaluation metrics, frameworks, and self-improvement methods from 2024-2026. 40+ references across 7 areas: perceptual metrics, multi-view consistency, geometric quality, texture/PBR evaluation, automated evaluation pipelines, human preference alignment, and reward-guided self-improvement.

### Key Findings

1. **Hi3DEval (NeurIPS 2025) is the new SOTA automated evaluator**
   - Hierarchical: object-level (5 dims), part-level (2 dims), material-subject (4 dims)
   - Uses InternVideo2.5 encoder + PartField 3D features
   - Kendall tau 0.774 vs GPTEval3D 0.690 on geometry plausibility
   - Multi-agent annotation pipeline (M2AP) achieves L1 0.257 vs single GPT-4.1 L1 0.702

2. **DreamSim supersedes LPIPS** (tau 0.77 vs 0.58)
   - NVS benchmarking (2025) confirms DreamSim degrades gracefully with corruption while LPIPS/SSIM cliff-edge
   - Drop-in replacement: `pip install dreamsim`, ~10 LOC

3. **Our evaluator's estimated human correlation is tau ~0.3-0.5** (significantly below SOTA)
   - Hand-crafted heuristics vs learned features
   - Missing input alignment entirely (CLIP-S/VQAScore)
   - Missing multi-view consistency (MEt3R)

4. **Self-improvement via quality metrics is real**
   - DreamDPO: +24-48 pts on GPTEval3D via DPO with HPSv2 reward
   - DSO (ICCV 2025): DRO converges faster than DPO for 3D generator alignment
   - VLM3D: Qwen2.5-VL as differentiable reward outperforms CLIP-based SDS

5. **Production systems use multi-metric dashboards + human QA**
   - No fully automated scoring disclosed by Meshy/Tripo/Rodin
   - 3D Arena: 123K votes confirm automated metrics weakly correlate with preference

### Priority Evaluator Upgrades
1. Add CLIP-S (2 hrs) -- most universally reported metric, 5ms compute
2. Replace LPIPS with DreamSim (1 hr) -- 2x better human alignment
3. Add LAION Aesthetic (1 hr) -- cheap aesthetic proxy
4. Add PBR plausibility checks (2 hrs) -- metallic/roughness/albedo histogram checks
5. Add VQAScore or VLM scoring (1 day) -- compositional understanding
6. Add MEt3R consistency (2 days) -- multi-view coherence

### File
`TRELLIS.2/optimization/research/survey_quality_metrics_2026_02.md`

---

## V3.2 Final Evaluation Results (2026-02-19 Evening)

### V3.2 Comparison Table (3 test images × 4 configs)
| Config | shape | frag | smooth | dark | detail | OVERALL |
|--------|-------|------|--------|------|--------|---------|
| Baseline | 58.0 | 65.7 | 28.8 | 87.4 | 99.8 | **68.5** |
| CFG-MP 0.15 (fixed) | 53.1 | 71.6 | 28.0 | 90.7 | 100.0 | **69.5** |
| CFG-MP 0.3 (fixed) | 58.0 | 66.9 | 29.7 | 86.4 | 99.5 | **68.7** |
| texres_4096 | 58.0 | 66.0 | 28.8 | 87.4 | 99.8 | **68.6** |

### Key Findings
1. **CFG-MP 0.15 (fixed) is marginal best** (+1.0 over baseline), driven by fragmentation +5.9
2. **texres_4096 has NO effect** — higher bake resolution doesn't improve quality
3. **Smoothness metric is inherently limited** for textured objects — base_color renders show legitimate texture detail that gets penalized
4. **Fragmentation is the real bottleneck** (65.7/100) — visible in complex objects like the steampunk boy character
5. **Darkness and vibrancy are already excellent** (87+ and 99+)

### Evaluator Evolution
| Version | Baseline Score | Changes |
|---------|---------------|---------|
| v3 | 59.9 | Initial: all thresholds tight, SSIM for detail, all views averaged |
| v3.1 | 59.5 | Fixed: ref resize, postprocess params, ref-only metrics for view 0 |
| v3.2 | 68.5 | Fixed: detail→intrinsic Laplacian, smoothness thresholds relaxed |

---

## V3.1 Evaluation Results & CFG-MP Fix (2026-02-19 Evening)

### V3.1 Evaluator Fixes
- **Fixed**: Reference image resize bug (1024→512 mismatch caused ValueError)
- **Fixed**: Postprocess params (`sharpen_texture=True`→`False`, `normal_strength=0.0`→`0.7`)
- **Fixed**: Smoothness thresholds (too aggressive for base_color renders: lap>4→lap>10, canny 25/70→40/100)
- **Fixed**: Reference-dependent metrics (shape/color_match) now only use front view, not diluted by 8 views
- **Changed**: detail metric from SSIM (unusable for 2D-to-3D) to intrinsic texture richness (Laplacian energy)

### V3.1 Baseline (3 images, scoring_version v3.1)
| Metric | Score | Weight |
|--------|-------|--------|
| shape | 58.0 | 15 |
| color_match | 43.1 | 10 |
| detail | 9.7 | 10 |
| fragmentation | 65.7 | 20 |
| smoothness | 28.8 | 15 |
| darkness | 87.4 | 15 |
| vibrancy | 99.8 | 15 |
| **OVERALL** | **59.5** | |

### CFG-MP Sweep (old implementation, CFG-boosted pred_x_0)
- Tested: cfg_mp_strength = [0.0, 0.15, 0.3, 0.5]
- **Result: ALL IDENTICAL** (59.3-59.5, within noise)
- **Root cause**: `pred_x_0` was derived from CFG-combined velocity, not conditional-only prediction
- The manifold projection target was already off-manifold, making correction ~zero

### CFG-MP Fix (conditional-only pred_x_0)
- **classifier_free_guidance_mixin.py**: Store `self._last_cond_pred_v = pred_pos` before CFG combination
- **flow_euler.py**: Use `self._last_cond_pred_v` for manifold projection instead of CFG-boosted pred_x_0
- **Status**: Implemented, pending re-evaluation

---

## Phase A — Quick Wins Implementation (2026-02-19)

### Deep Research Survey
- Surveyed 49 papers and 26 methods across 5 areas (geometry, texture, guidance, architecture, postprocess)
- Defined 8 quality dimensions: Geometric Fidelity, Appearance/Texture, Multi-View Consistency, Perceptual Quality, Physical Plausibility, Input Alignment, PBR Material, Production Quality
- Prioritized roadmap: Phase A (quick wins) → Phase D (architecture changes)
- Created adjustment tracking system in `adjustments/` folder with rollback support

### A1. UV Seam Smoothing (adj #001)
- **File**: `o-voxel/o_voxel/postprocess.py`
- **Technique**: Erosion-based UV island boundary detection → dilated seam band → Gaussian blur
- **What**: After inpainting, detect UV island boundaries via morphological erosion, create 5-pixel seam band, apply GaussianBlur(7x7, σ=1.5) to base_color/normal_map, GaussianBlur(5x5, σ=1.0) to metallic/roughness
- **Expected**: +1-2 points on smoothness/coherence
- **Risk**: Very low (only affects narrow band at UV island edges)
- **Status**: Implemented, pending evaluation

### A3. CFG-MP Manifold Projection (adj #002)
- **File**: `trellis2/pipelines/samplers/flow_euler.py`
- **Paper**: arXiv 2601.21892 (Jan 2025) — validated on DiT-XL-2, Flux, SD3.5
- **Technique**: After each Euler step, project sample back toward data manifold using noise-space re-interpolation from pred_x_0
- **Formula**: `x_manifold = (σ_prev/σ_t)*x_t + ((1-t_prev)-(1-t)*σ_prev/σ_t)*pred_x_0`; `x_new = x_euler + λ*(x_manifold - x_euler)`
- **Parameter**: `cfg_mp_strength` (0=off, recommend 0.1-0.3), added as UI slider in app.py Stage 3
- **Different from**: CFG-Zero* (scale), APG (direction) — CFG-MP constrains trajectory
- **Expected**: +2-4 points (addresses off-manifold drift at high guidance)
- **Risk**: Low (default off, backward compatible)
- **Status**: Implemented, pending evaluation with sweep [0.05, 0.1, 0.15, 0.2, 0.3]

---

## Generation 0 — Baseline & Phase 2-3 Sweep (2026-02-16)

### Established Baseline
- **Config**: tex_guidance=5.0, tex_rescale=0.85, shape_guidance=7.5, shape_rescale=0.5
- **Score**: avg 46.37/100 across 3 test examples
- **Breakdown**: Ex4(shoes)=48.9, Ex6(chest)=52.0, Ex7(mickey)=38.2
- **Weakness**: Low silhouette IoU (22/40/16), low detail scores (~28), high artifacts on Ex6

### Phase 2: Texture Guidance Sweep (iter 0-24)
- Tested: tex_guidance={3,4,5,6,7,8} x tex_rescale={0.5,0.7,0.85,0.95}
- **Key Finding**: `tex_rescale=0.95` consistently best (r=+0.822 with overall score)
- **Key Finding**: Low rescale (0.5) with high guidance is catastrophic (avg 33.5)
- **Best Combo**: tex_guidance=8.0, tex_rescale=0.95 → avg 50.0 (+3.6 over baseline)
- **Pattern**: rescale importance increases with guidance level (non-linear interaction)

### Phase 3: Shape Guidance Sweep (iter 25-44)
- Tested: shape_guidance={5,7.5,10,12} x shape_rescale={0.3,0.5,0.7,0.9} x steps={8,12,16}
- **Key Finding**: Shape guidance has NEGLIGIBLE impact (r<0.1, all within ±1.5 of baseline)
- **Root cause**: Shape is determined by Stage 1 (sparse structure), not Stage 2 (shape SLAT)
- **Conclusion**: Stop tuning shape_slat parameters, focus on SS and tex_slat

### Phase 4: Multi-view Mode Sweep (iter 45-56) — IN PROGRESS
- Tested 8 combos so far: multiview_mode={concat,view_weighted} x texture_mode={single,concat,tapa,view_weighted}
- **Best combo**: `concat + single` → avg 48.6 (+2.2 over baseline)
  - Ex4 (shoes) hit **50.8** — best single-example score across all iters
  - `single` texture mode avoids TAPA blending artifacts, better for simpler geometries
- **Surprise**: `concat + tapa` (our default) only avg 46.4 — **below concat+single**
- **view_weighted** modes clustered at 45.7-46.4, slightly worse than concat
- **Conclusion**: texture_multiview_mode matters more than multiview_mode

### Population Seeded
- 2 elite individuals from Phase 2 results
- 14 hypothesis-driven candidates queued for GA Generation 1
- 9 new candidates for CFG-Zero*, APG, and beta schedule guidance modes

---

## Mid-Run Review — 53 Iterations Analysis (2026-02-16)

### Overall Statistics
- **53 iterations** completed (Phase 1-4 partial)
- Score range: 33.5 — 50.0 (mean=45.3, stdev=3.8)
- Best: iter 24 (tex_g=8.0, r=0.95) = **50.0**
- Cherry-pick ceiling (best per-example): **51.5**

### Parameter Impact Ranking (Pearson correlation with overall score)

| Parameter | r(overall) | r(silhouette) | r(detail) | r(artifacts) | r(color) |
|-----------|-----------|---------------|-----------|-------------|---------|
| **tex_rescale** | **+0.824** | +0.821 | -0.424 | +0.791 | +0.667 |
| tex_guidance | -0.305 | -0.439 | +0.272 | +0.001 | -0.346 |
| shape_rescale | +0.101 | +0.135 | -0.061 | +0.015 | +0.055 |
| shape_steps | -0.094 | -0.091 | +0.040 | -0.094 | -0.079 |
| shape_guidance | +0.050 | +0.061 | -0.029 | -0.005 | +0.040 |

**Key insight**: `tex_rescale` is the DOMINANT parameter (r=+0.824). All shape parameters are noise (|r|<0.1).

### Rescale Value Impact (clear step function)

| Rescale | N | Mean Score | Range |
|---------|---|-----------|-------|
| 0.50 | 6 | 38.8 | 33.5 — 43.0 |
| 0.70 | 6 | 40.3 | 35.0 — 45.5 |
| 0.85 | 35 | 46.7 | 44.8 — 48.6 |
| 0.95 | 6 | **48.3** | 47.3 — 50.0 |

### Dimension Weakness Analysis

| Dimension | Avg | Best Achievable | Status |
|-----------|-----|-----------------|--------|
| **Silhouette** | 23.8/100 | 27.8 | CRITICAL bottleneck |
| **Detail** | 26.3/100 | 43.8 | Major bottleneck |
| Contour | 48.6/100 | 56.4 | Medium |
| Color | 48.5/100 | 64.1 | Medium |
| Artifacts | 73.5/100 | 88.1 | OK |

- Silhouette is capped by Stage 1 sparse structure — tuning Stage 2/3 cannot fix this
- Detail has high variance (stdev=5.4), most room for improvement via texture params
- Ex7 (mickey) is hardest: silhouette avg=14.4, detail avg=21.9

### Paradoxical tex_guidance Finding
At rescale=0.95, guidance shows monotonic improvement 3→8:
`3.0(48.0) → 4.0(47.6) → 5.0(47.3) → 6.0(48.2) → 7.0(48.8) → 8.0(50.0)`
But overall correlation is **negative** (-0.305) because low-rescale + high-guidance is catastrophic.
**Takeaway**: guidance and rescale MUST be tuned together. High guidance only works with high rescale.

### Recommendations for GA
1. Fix `tex_rescale=0.95` (or 1.0) — stop wasting evals on low rescale
2. Push `tex_guidance` higher (10, 12, 15) — monotonic trend at r=0.95 hasn't plateaued
3. Focus on SS parameters for silhouette improvement — the real bottleneck
4. Test APG/CFG-Zero* — may improve detail and artifacts without increasing guidance
5. Beta schedule — may smooth artifacts at boundaries

---

## Survey Completed — 2D-to-3D Quality Improvement (2026-02-16)

### Full survey: `survey_2d_to_3d_quality.md` (25+ papers)

### Quick Wins Implemented (no retraining)
1. **CFG-Zero*** (arXiv 2503.18886): s* projection + zero-init — `cfg_utils.py`
2. **APG** (ICLR 2025): Orthogonal decomposition — `cfg_utils.py`
3. Both integrated into all 4 CFG sites (mixin, multidiffusion, TAPA, VW-concat)
4. New params: `cfg_mode`, `apg_alpha`, `zero_init_steps`

### Quick Wins Implemented (cont.)
5. **Beta-shaped guidance schedule**: Smooth beta distribution curve replaces binary on/off — `guidance_interval_mixin.py`
   - New params: `guidance_schedule` ('binary'|'beta'), `guidance_beta_a`, `guidance_beta_b`
   - 3 candidates queued: early-peak (a=2,b=5), symmetric (a=2,b=2), late-peak (a=3,b=2)
   - Hypothesis H17 added

### Quick Wins Queued
- UV seam Laplacian blending (pure post-processing)

### Medium-Effort Methods Identified
- PBR-SR (NeurIPS'25): 4x zero-shot PBR texture SR via DiffBIR + nvdiffrast
- 3DEnhancer (CVPR'25): DiT-based multi-view enhancement
- Render-and-compare: iterative texture optimization via nvdiffrast (already have infra)

### Hypotheses Active
- H7: Higher tex_guidance (10, 12) — queued
- H8: More SS steps (16, 20) — queued
- H9: Higher SS guidance (10, 12) — queued
- H10: More tex steps (16, 20) — queued
- H11: Higher rescale_t (5, 7) — queued
- H12: Seed diversity — queued
- H13: Combined optimization — queued
- H14: APG guidance (alpha=0.0, 0.3, 0.5) — **NEW, queued**
- H15: CFG-Zero* (zero_init=1, 2) — **NEW, queued**
- H16: APG + CFG-Zero* combined — **NEW, queued**
- H17: Beta-shaped guidance schedule (early/symmetric/late) — **NEW, queued**

---

## Research Cycle Status

| Phase | Status | Progress |
|-------|--------|----------|
| Survey | COMPLETE | 25+ papers, 10 ranked findings |
| Hypothesize | COMPLETE | 17 hypotheses (H1-H17), 3 confirmed |
| Experiment Design | COMPLETE | 25 candidates in population.json |
| Run (auto_optimize) | IN PROGRESS | iter 53/100 (Phase 4 multiview, ETA ~6h) |
| Run (GA) | WAITING | GA runner queued, starts after auto_optimize |
| Mid-Run Review | COMPLETE | 53-iter analysis, 5 recommendations |
| Review & Evolve | PENDING | After GA Generation 1 completes |


## Generation 2 — Evolved (2026-02-16 23:28)

### Selection Summary
- Previous generation: 1 (25 evaluated)
- Best fitness: 51.8 (gen1-h13-combined)
- Elites preserved: 6
- New offspring: 19

### New Population
| ID | Origin | Status |
|----|--------|--------|
| gen1-h13-combined | elite_from_gen1 | evaluated (51.8) |
| gen1-h12-seed789 | elite_from_gen1 | evaluated (51.6) |
| gen1-h17-beta-late | elite_from_gen1 | evaluated (51.5) |
| gen1-h17-beta-sym | elite_from_gen1 | evaluated (51.4) |
| gen1-h7-tg12.0 | elite_from_gen1 | evaluated (51.3) |
| gen1-h7-tg10.0 | elite_from_gen1 | evaluated (51.2) |
| gen2-001 | crossover(gen1-h17-beta-sym,gen1-h7-tg10.0) | queued |
| gen2-002 | mutation(gen1-h17-beta-late) | queued |
| gen2-003 | crossover(gen1-h13-combined,gen1-h15-czs-z1) | queued |
| gen2-004 | crossover(gen1-h14-apg-a0.0,gen1-h17-beta-sym) | queued |
| gen2-005 | crossover(gen1-h17-beta-late,gen1-h16-apg-czs) | queued |
| gen2-006 | mutation(gen1-h15-czs-z1) | queued |
| gen2-007 | mutation(gen1-h7-tg12.0) | queued |
| gen2-008 | crossover(gen1-h17-beta-sym,gen1-h7-tg10.0) | queued |
| gen2-009 | crossover(gen1-h7-tg12.0,gen1-h8-ss20) | queued |
| gen2-010 | mutation(gen1-h11-rt5.0) | queued |
| gen2-011 | crossover(gen1-h14-apg-a0.0,gen1-h16-apg-czs) | queued |
| gen2-012 | crossover(gen1-h7-tg10.0,gen1-h17-beta-sym) | queued |
| gen2-013 | crossover(gen1-h9-ssg10.0,gen1-h9-ssg12.0) | queued |
| gen2-014 | crossover(gen1-h14-apg-a0.0,gen1-h13-combined) | queued |
| gen2-015 | mutation(gen1-h13-combined) | queued |
| gen2-016 | mutation(gen1-h9-ssg12.0) | queued |
| gen2-017 | crossover(gen1-h12-seed789,gen1-h17-beta-sym) | queued |
| gen2-018 | crossover(gen1-h17-beta-sym,gen1-h13-combined) | queued |
| gen2-019 | crossover(gen1-h7-tg12.0,gen1-h15-czs-z1) | queued |


## Generation 3 — Evolved (2026-02-17 02:29)

### Selection Summary
- Previous generation: 2 (25 evaluated)
- Best fitness: 52.2 (gen2-004)
- Elites preserved: 6
- New offspring: 19

### New Population
| ID | Origin | Status |
|----|--------|--------|
| gen2-004 | elite_from_gen2 | evaluated (52.2) |
| gen1-h13-combined | elite_from_gen2 | evaluated (51.8) |
| gen1-h12-seed789 | elite_from_gen2 | evaluated (51.6) |
| gen2-006 | elite_from_gen2 | evaluated (51.6) |
| gen2-002 | elite_from_gen2 | evaluated (51.6) |
| gen1-h17-beta-late | elite_from_gen2 | evaluated (51.5) |
| gen3-001 | mutation(gen1-h12-seed789) | queued |
| gen3-002 | mutation(gen1-h17-beta-sym) | queued |
| gen3-003 | mutation(gen2-004) | queued |
| gen3-004 | mutation(gen2-005) | queued |
| gen3-005 | crossover(gen1-h13-combined,gen1-h17-beta-sym) | queued |
| gen3-006 | crossover(gen2-007,gen2-006) | queued |
| gen3-007 | crossover(gen1-h13-combined,gen1-h12-seed789) | queued |
| gen3-008 | crossover(gen2-002,gen1-h17-beta-late) | queued |
| gen3-009 | mutation(gen2-004) | queued |
| gen3-010 | crossover(gen1-h7-tg12.0,gen1-h7-tg10.0) | queued |
| gen3-011 | crossover(gen2-002,gen2-003) | queued |
| gen3-012 | mutation(gen1-h13-combined) | queued |
| gen3-013 | mutation(gen1-h17-beta-late) | queued |
| gen3-014 | crossover(gen1-h12-seed789,gen1-h7-tg10.0) | queued |
| gen3-015 | crossover(gen2-013,gen2-016) | queued |
| gen3-016 | crossover(gen2-008,gen2-002) | queued |
| gen3-017 | crossover(gen1-h12-seed789,gen2-003) | queued |
| gen3-018 | crossover(gen1-h13-combined,gen2-006) | queued |
| gen3-019 | crossover(gen1-h12-seed789,gen1-h7-tg10.0) | queued |


## Generation 4 — Evolved (2026-02-17 03:49)

### Selection Summary
- Previous generation: 3 (25 evaluated)
- Best fitness: 52.2 (gen2-004)
- Elites preserved: 6
- New offspring: 19

### New Population
| ID | Origin | Status |
|----|--------|--------|
| gen2-004 | elite_from_gen3 | evaluated (52.2) |
| gen1-h13-combined | elite_from_gen3 | evaluated (51.8) |
| gen1-h12-seed789 | elite_from_gen3 | evaluated (51.6) |
| gen2-006 | elite_from_gen3 | evaluated (51.6) |
| gen2-002 | elite_from_gen3 | evaluated (51.6) |
| gen1-h17-beta-late | elite_from_gen3 | evaluated (51.5) |
| gen4-001 | crossover(gen2-002,gen1-h12-seed789) | queued |
| gen4-002 | crossover(gen1-h12-seed789,gen3-011) | queued |
| gen4-003 | crossover(gen1-h17-beta-late,gen3-011) | queued |
| gen4-004 | crossover(gen1-h12-seed789,gen1-h13-combined) | queued |
| gen4-005 | mutation(gen2-002) | queued |
| gen4-006 | crossover(gen3-015,gen2-002) | queued |
| gen4-007 | crossover(gen3-011,gen3-014) | queued |
| gen4-008 | crossover(gen2-004,gen3-015) | queued |
| gen4-009 | crossover(gen1-h13-combined,gen2-004) | queued |
| gen4-010 | mutation(gen3-007) | queued |
| gen4-011 | mutation(gen3-005) | queued |
| gen4-012 | crossover(gen1-h13-combined,gen2-002) | queued |
| gen4-013 | crossover(gen3-019,gen2-006) | queued |
| gen4-014 | crossover(gen1-h13-combined,gen2-004) | queued |
| gen4-015 | crossover(gen3-003,gen3-017) | queued |
| gen4-016 | mutation(gen3-003) | queued |
| gen4-017 | crossover(gen3-003,gen2-004) | queued |
| gen4-018 | crossover(gen3-005,gen3-014) | queued |
| gen4-019 | crossover(gen3-005,gen1-h13-combined) | queued |


## Generation 1 — Evolved (2026-02-18 11:07)

### Selection Summary
- Previous generation: 0 (12 evaluated)
- Best fitness: 37.8 (v2-high-tex-g)
- Elites preserved: 2
- New offspring: 6

### New Population
| ID | Origin | Status |
|----|--------|--------|
| v2-high-tex-g | elite_from_gen0 | evaluated (37.8) |
| v2-high-rescale | elite_from_gen0 | evaluated (37.3) |
| gen1-001 | mutation(v2-high-tex-g) | queued |
| gen1-002 | crossover(v2-random-001,v2-high-tex-g) | queued |
| gen1-003 | crossover(v2-high-rescale,v2-random-001) | queued |
| gen1-004 | crossover(v2-prev-champion,v2-random-003) | queued |
| gen1-005 | mutation(v2-random-001) | queued |
| gen1-006 | mutation(v2-high-rescale) | queued |


## Generation 2 — Evolved (2026-02-18 11:07)

### Selection Summary
- Previous generation: 1 (2 evaluated)
- Best fitness: 37.8 (v2-high-tex-g)
- Elites preserved: 2
- New offspring: 6

### New Population
| ID | Origin | Status |
|----|--------|--------|
| v2-high-tex-g | elite_from_gen1 | evaluated (37.8) |
| v2-high-rescale | elite_from_gen1 | evaluated (37.3) |
| gen2-001 | crossover(v2-high-tex-g,v2-high-tex-g) | queued |
| gen2-002 | crossover(v2-high-tex-g,v2-high-tex-g) | queued |
| gen2-003 | crossover(v2-high-tex-g,v2-high-tex-g) | queued |
| gen2-004 | crossover(v2-high-tex-g,v2-high-tex-g) | queued |
| gen2-005 | mutation(v2-high-tex-g) | queued |
| gen2-006 | crossover(v2-high-tex-g,v2-high-tex-g) | queued |


## Generation 3 — Evolved (2026-02-18 11:07)

### Selection Summary
- Previous generation: 2 (2 evaluated)
- Best fitness: 37.8 (v2-high-tex-g)
- Elites preserved: 2
- New offspring: 6

### New Population
| ID | Origin | Status |
|----|--------|--------|
| v2-high-tex-g | elite_from_gen2 | evaluated (37.8) |
| v2-high-rescale | elite_from_gen2 | evaluated (37.3) |
| gen3-001 | mutation(v2-high-tex-g) | queued |
| gen3-002 | mutation(v2-high-tex-g) | queued |
| gen3-003 | crossover(v2-high-tex-g,v2-high-tex-g) | queued |
| gen3-004 | mutation(v2-high-tex-g) | queued |
| gen3-005 | crossover(v2-high-tex-g,v2-high-tex-g) | queued |
| gen3-006 | mutation(v2-high-tex-g) | queued |


## Generation 1 — Evolved (2026-02-18 11:09)

### Selection Summary
- Previous generation: 0 (12 evaluated)
- Best fitness: 37.8 (v2-high-tex-g)
- Elites preserved: 2
- New offspring: 6

### New Population
| ID | Origin | Status |
|----|--------|--------|
| v2-high-tex-g | elite_from_gen0 | evaluated (37.8) |
| v2-high-rescale | elite_from_gen0 | evaluated (37.3) |
| gen1-001 | mutation(v2-high-tex-g) | queued |
| gen1-002 | crossover(v2-random-001,v2-high-tex-g) | queued |
| gen1-003 | crossover(v2-high-rescale,v2-random-001) | queued |
| gen1-004 | crossover(v2-prev-champion,v2-random-003) | queued |
| gen1-005 | mutation(v2-random-001) | queued |
| gen1-006 | mutation(v2-high-rescale) | queued |


## Generation 2 — Evolved (2026-02-18 11:44)

### Selection Summary
- Previous generation: 1 (8 evaluated)
- Best fitness: 37.8 (v2-high-tex-g)
- Elites preserved: 2
- New offspring: 6

### New Population
| ID | Origin | Status |
|----|--------|--------|
| v2-high-tex-g | elite_from_gen1 | evaluated (37.8) |
| v2-high-rescale | elite_from_gen1 | evaluated (37.3) |
| gen2-001 | crossover(v2-high-rescale,v2-high-tex-g) | queued |
| gen2-002 | crossover(v2-high-tex-g,gen1-006) | queued |
| gen2-003 | mutation(v2-high-rescale) | queued |
| gen2-004 | crossover(v2-high-tex-g,gen1-002) | queued |
| gen2-005 | mutation(v2-high-tex-g) | queued |
| gen2-006 | mutation(v2-high-tex-g) | queued |


## Generation 3 — Evolved (2026-02-18 12:18)

### Selection Summary
- Previous generation: 2 (8 evaluated)
- Best fitness: 37.8 (v2-high-tex-g)
- Elites preserved: 2
- New offspring: 6

### New Population
| ID | Origin | Status |
|----|--------|--------|
| v2-high-tex-g | elite_from_gen2 | evaluated (37.8) |
| gen2-004 | elite_from_gen2 | evaluated (37.8) |
| gen3-001 | mutation(v2-high-rescale) | queued |
| gen3-002 | crossover(gen2-004,gen2-001) | queued |
| gen3-003 | crossover(v2-high-tex-g,gen2-006) | queued |
| gen3-004 | crossover(gen2-006,v2-high-rescale) | queued |
| gen3-005 | crossover(gen2-006,v2-high-tex-g) | queued |
| gen3-006 | mutation(gen2-004) | queued |


## Quality Definitions Survey (2026-02-19)

### Comprehensive quality taxonomy completed

Wrote `quality_definitions.md` -- a thorough survey of how "quality" is defined in the image-to-3D generation literature, covering 30+ papers from CVPR, NeurIPS, ICCV, ICLR, SIGGRAPH (2023-2026).

### Eight Quality Dimensions Identified

1. **Geometric Fidelity**: CD, F-score, V-IoU, HD, EMD, Normal Consistency, Silhouette IoU, Depth Error
2. **Appearance/Texture Quality**: LPIPS, SSIM, PSNR, FID, KID, DreamSim
3. **Multi-View Consistency**: MEt3R (CVPR'25), PPLC, FVD
4. **Perceptual Quality / Human Preference**: GPTEval3D (CVPR'24), 3DGen-Bench, MATE-3D/HyperScore (ICCV'25), 3D Arena, T3Bench, Rank2Score, DreamSim, LAION Aesthetic
5. **Physical Plausibility**: Manifoldness, Watertightness, Self-Intersection, Janus Detection, SRAM
6. **Prompt/Input Alignment**: CLIP-S, ULIP-2, Uni3D, Object Detection-based
7. **Material Quality (PBR)**: PBR attribute PSNR/LPIPS, Relighting Consistency, Albedo Accuracy
8. **Production Quality**: Topology, UV Mapping, Texel Density, TGE, TMQA

### Key Findings

- **No single metric captures quality** -- the field has moved to multi-dimensional evaluation
- **Human preference is gold standard** -- best automated metrics trained on human data
- **GPTEval3D's 5 criteria** (alignment, 3D plausibility, texture detail, geometry detail, tex-geom coherency) are the most widely adopted framework
- **3DGen-Bench** provides the largest human preference dataset (68K expert votes) with 5 overlapping dimensions
- **DreamSim** (NeurIPS'23 Spotlight) bridges LPIPS and CLIP; better human alignment than either alone
- **MEt3R** (CVPR'25) is the first dedicated multi-view consistency metric
- **SRAM** enables no-reference shape realism evaluation via LLM bridge

### Gaps Identified in Our Current Evaluator

Our 7-dimension evaluator (`evaluate.py`) is missing:
1. Input alignment (CLIP-S) -- biggest gap
2. Aesthetic quality (LAION Aesthetic) -- cheap to add
3. Multi-view consistency (MEt3R) -- medium effort
4. PBR material plausibility checks
5. UV quality metrics
6. No-reference realism (SRAM)

### papers.json Updated

Added 16 new papers focused on evaluation and quality metrics:
- GPTEval3D, 3DGen-Bench, MEt3R, DreamSim, SRAM, TGE, MATE-3D/HyperScore
- 3D Arena, TMQA, T3Bench, Rank2Score, Meta 3D AssetGen
- Rethinking FID, TRELLIS.2 (native paper), PBR_Boost, Objectness Similarity

### Recommended Next Steps

**Phase 1 (quick, high impact)**:
- Add CLIP-S to evaluator for input alignment
- Add DreamSim as LPIPS alternative
- Add LAION Aesthetic score

**Phase 2 (medium effort)**:
- Integrate multi-view consistency check
- Add PBR plausibility validation
- UV seam visibility metric

**Phase 3 (research-grade)**:
- Train or integrate HyperScore/3DGen-Score
- VLM-based evaluation (GPTEval3D-style)
- SRAM for no-reference shape realism
