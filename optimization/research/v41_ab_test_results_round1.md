# V4.1 A/B Test Results — Round 1 (2026-02-23)

## Summary

6 completed test presets, 2+ pending. Champion baseline: overall ≈ 92.0-92.7.

### Test 1: guidance_anneal (1 image, T.png)

| Config | Overall | A1 | A2 | C1 | C3 | Change |
|--------|---------|-----|-----|-----|-----|--------|
| ga_baseline | 93.88 | 85.8 | 87.7 | 98.8 | 76.3 | — |
| ga_25pct | 93.87 | 85.8 | 87.4 | 98.8 | 76.4 | anneal min=0.25, start=0.3 |
| ga_50pct | 93.88 | 85.8 | 87.5 | 98.8 | 76.4 | anneal min=0.5, start=0.3 |
| ga_25pct_wide | 93.86 | 85.8 | 87.5 | 98.6 | 76.4 | anneal min=0.25, start=0.5 |

**Verdict: ZERO EFFECT.** Linear guidance strength annealing near t=0 has no impact.

### Test 2: best_combos (3 images)

| Config | Overall | A1 | A2 | C1 | C3 | Change |
|--------|---------|-----|-----|-----|-----|--------|
| champion | 92.51 | 81.4 | 80.9 | 95.4 | 81.0 | baseline |
| **split_sched** | **92.69** | 80.8 | 81.6 | 93.4 | **86.0** | quad SS+shape, uniform tex |
| shape_rt4.5 | 92.46 | 80.7 | 82.3 | 95.4 | 80.1 | rescale_t=4.5 |
| split+rt4.5 | 92.70 | 80.8 | 81.5 | 93.4 | **86.1** | both |

**Verdict: split_sched = C3 +5.0 (!!)** — BIGGEST single improvement found.
Trade-off: C1 -2.0, A1 -0.6. rescale_t=4.5 is negligible.

### Test 3: guidance_interval (3 images)

| Config | Overall | A1 | A2 | C1 | C3 | Change |
|--------|---------|-----|-----|-----|-----|--------|
| gi_champion | 92.02 | 81.4 | 80.7 | 94.7 | 79.1 | baseline |
| gi_tex_narrow | 92.58 | 81.4 | 82.9 | 96.6 | 78.6 | interval=[0.05,0.85] |
| gi_tex_original | 92.51 | 81.4 | 80.9 | 95.4 | 81.0 | interval=[0.6,0.9] |
| gi_tex_beta33 | 92.66 | 81.3 | 82.6 | 96.3 | 79.6 | beta(3,3) schedule |

**Verdict: Narrow interval best for A2/C1.** A2 +2.2, C1 +1.9 but C3 -0.5.

### Test 4: guidance_sched2 (3 images)

| Config | Overall | A1 | A2 | C1 | C3 | Change |
|--------|---------|-----|-----|-----|-----|--------|
| gs2_baseline | 92.03 | 81.4 | 80.7 | 94.7 | 79.3 | baseline |
| gs2_triangular | 92.63 | 81.3 | 82.0 | 96.0 | 80.4 | triangular schedule |
| **gs2_beta42** | **92.49** | 81.4 | 81.3 | 94.2 | **82.2** | beta(4,2) schedule |
| gs2_tri_narrow | 92.62 | 81.3 | 82.8 | 96.7 | 78.4 | triangular + narrow [0.05,0.9] |

**Verdict: beta(4,2) = C3 +2.9** (second biggest C3 lever). tri_narrow = A2 +2.1, C1 +2.0.

### Test 5: sde_sampling (3 images, 5 configs)

All configs ≈ 92.7 overall. **ZERO EFFECT.** Confirmed across 5 alpha/profile combos.

### Test 6: occ_threshold (3 images, 4 configs) — RUNNING

- occ_baseline: T.png done = 93.8 (A1=85.7). Remaining configs processing...

### Test 7: staged_bon (1 image, 3 configs) — PENDING

### Test 8: silcorr (1 image, 4 configs) — PENDING

---

## Key Findings

### Biggest Levers Found (ranked by overall impact)

1. **split_sched** (quadratic SS+shape, uniform tex): **C3 +5.0**, overall +0.2
   - Concentrates ODE steps near t=0 where fine detail forms
   - Trade-off: C1 -2.0 (tex coherence drops slightly)
   - A1 -0.6 (minor silhouette impact)

2. **beta(4,2) guidance schedule**: **C3 +2.9**, overall +0.5
   - Bell-curve guidance peaking at 40% into interval
   - A2 modest +0.6, C1 modest -0.5

3. **Triangular + narrow interval [0.05,0.9]**: **A2 +2.1, C1 +2.0**, overall +0.6
   - Best for color fidelity and texture coherence
   - C3 -0.9 (minor detail trade-off)

4. **Beta(3,3) guidance**: A2 +1.9, C1 +1.6, overall +0.7
   - Best balanced single change

### Zero-Effect Findings (confirmed)

- SDE sampling (5 configs): ±0.0
- Guidance strength annealing near t=0: ±0.0
- rescale_t change (3.0 → 4.5): ±0.0

### Score Ceiling Analysis

| Metric | Current Best | Weight | Gap to 100 | Status |
|--------|-------------|--------|-----------|--------|
| B1 mesh_integrity | 98.0 | 10 | 2.0 | CEILING (watertight penalty) |
| B2 surface_quality | 100.0 | 10 | 0.0 | CEILING |
| C2 color_vitality | 100.0 | 10 | 0.0 | CEILING |
| D1 material | 100.0 | 10 | 0.0 | CEILING |
| E1 multiview | 100.0 | 10 | 0.0 | CEILING |
| C1 tex_coherence | 96.7 | 15 | 3.3 | Near ceiling |
| A1 silhouette | 81.4 | 15 | 18.6 | MAJOR GAP (need 64³/BON) |
| A2 color_dist | 82.9 | 10 | 17.1 | MAJOR GAP |
| C3 detail_richness | 86.1 | 10 | 13.9 | SIGNIFICANT GAP |

### Weighted Score Gap (contribution to overall)

- A1: 18.6 × 0.15 = **2.79 pts** (biggest)
- A2: 17.1 × 0.10 = 1.71 pts
- C3: 13.9 × 0.10 = 1.39 pts
- C1: 3.3 × 0.15 = 0.50 pts
- B1: 2.0 × 0.10 = 0.20 pts

**Total recoverable: ~6.6 pts to reach ~99/100 theoretical max.**
**Realistic target: 95-96/100** (A1 to ~90, A2 to ~90, C3 to ~90).

---

## Round 2 Combo Hypotheses

### r3_split_b42 (quadratic geo + beta(4,2) guidance)
- Expected C3: 87-89 (if split_sched +5.0 and beta(4,2) +2.9 partially additive)
- Expected overall: 93.0-93.5

### r3_split_tri_n (quadratic geo + triangular + narrow)
- Expected: C3 ≈ 85 (split_sched +5.0 minus tri_narrow -0.9)
- Expected A2: 83-84 (narrow +2.1)
- Expected C1: 95-96 (triangular +2.0)

### r3_full_combo (split + beta42 + narrow + BON4)
- Expected: best of all worlds if interactions are positive
- Risk: BON4 adds 4x generation time

### 64³ native cascade
- Expected A1: +3-5 (from 32³→64³ occupancy resolution)
- Expected overall: +0.5-0.8

---

## Pending Questions

1. Are split_sched and beta(4,2) additive or sublinear?
2. Does 64³ cascade improve A1 enough to justify 2x cost?
3. Can BON4 close the A1 gap further (stage-wise shape selection)?
4. Does silhouette correction (Dice loss) actually help A1?
5. What's the optimal combined config for 1536 resolution?
