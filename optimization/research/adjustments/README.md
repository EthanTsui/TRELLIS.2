# Adjustments Tracking Log

This directory records every parameter/method adjustment and its impact on quality scores.
Each adjustment is stored as a separate JSON file for easy rollback and comparison.

## File Naming Convention

```
YYYY-MM-DD_NNN_<short-description>.json
```

Example: `2026-02-19_001_baseline-champion-v2.json`

## JSON Schema

```json
{
  "id": "2026-02-19_001",
  "timestamp": "2026-02-19T12:00:00Z",
  "description": "Brief description of what was changed",
  "hypothesis": "Why we expect this to improve quality",
  "category": "guidance|postprocess|architecture|sampling|texture",
  "parent_id": "previous adjustment ID (for rollback chain)",
  "config": {
    "...": "full parameter configuration"
  },
  "changes_from_parent": {
    "param_name": {"old": "value", "new": "value"}
  },
  "scores": {
    "overall": 0.0,
    "per_dimension": {
      "geometry_accuracy": 0.0,
      "surface_smoothness": 0.0,
      "texture_fidelity": 0.0,
      "texture_sharpness": 0.0,
      "multi_view_consistency": 0.0,
      "pbr_material_quality": 0.0,
      "perceptual_quality": 0.0
    },
    "per_example": {}
  },
  "status": "pending|running|completed|failed",
  "notes": "Any observations about the result"
}
```

## Quality Dimensions (from research)

Based on comprehensive literature survey, quality is evaluated across 7 dimensions:

1. **Geometry Accuracy** - Chamfer Distance, Volume IoU, F-Score
2. **Surface Smoothness** - Normal consistency, Laplacian smoothness, mesh regularity
3. **Texture Fidelity** - LPIPS, SSIM, PSNR vs input image
4. **Texture Sharpness** - Detail preservation, frequency analysis
5. **Multi-View Consistency** - Cross-view texture coherence, PPLC
6. **PBR Material Quality** - Physically plausible materials, metallic/roughness accuracy
7. **Perceptual Quality** - GPTEval3D score, CLIP similarity, human preference alignment

## Rollback Instructions

To rollback to a previous configuration:
1. Find the target adjustment JSON file
2. Copy its `config` section
3. Apply to `app.py` or pipeline parameters
4. The `parent_id` chain allows tracing the full history
