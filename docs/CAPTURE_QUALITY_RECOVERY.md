# Stage 4 capture alignment recovery

This v2.0.4 change is in Stage 4. The saved Stage 2/3 bundle remains compatible and immutable. No upstream stage or model needs regeneration for a Stage 4-only test. The qualified cold six-piece run and fourteen-case regression are recorded in [the validation review](V204_CAPTURE_ALIGNMENT_REVIEW.md).

The published release is v2.0.5, which corrects only a Windows path-alias assertion in a runner test. The qualified solver fingerprint, commands, upstream evidence and reconstruction results are unchanged.

## Recovery

A supported complete candidate can still contain a misplaced or rotated piece. Under the authenticated continuous-rigid shared-scale complete-page profile, Stage 4 now:

1. Ranks single-piece tree cuts by their contribution to the nonrectangular outline. Long observed contour spans suggest parallel/perpendicular border rotations. The spans are hypotheses, not connecting seams.
2. Fits native translations and curved contacts at those angles, then verifies native seam support, interval exclusivity, collision, scale and complete-page geometry.
3. Proposes joint rigid corrections from opposing seams and observed page borders, fixing one anchor and keeping each scale exactly one. Support is re-extracted and independently replayed after poses change.
4. Revisits saved pair modes across supported tree cuts. Vectorized rectangle area ranks hypotheses; fine batches are independently verified. A failed batch does not remove later modes. The last verified incumbent survives failed fitting and timeouts.

Quality work stops at a geometry target derived from raster uncertainty, local-neighborhood exhaustion, or the shared cooperative deadline. A supported page meeting the existing 0.98 stopping target is retained after local repair finishes. Acceptance gates are unchanged. A failed bounded search is not an impossibility claim.

The quarter-turn development route keeps its existing behavior. This recovery uses CPU NumPy/SciPy; it needs no OpenCV, CUDA, original page or previous-result file.

## Cold six-piece benchmark

From the cloned repository root, after setup and after preparing your own capture pipeline's Stage 2/3 outputs:

```powershell
python -B tools\run_stage4_capture.py `
  --upstream outputs\my_saved_capture_pipeline `
  --output outputs\real_six_v204_cold `
  --cpu-threads 4
```

Relative input/output arguments resolve inside the repository root. The upstream folder must contain `stage3/bundle.json` and `stage1/capture_profile.json`. This is the saved pipeline output folder, not the raw captures folder. The runner never reads that folder's old Stage 4 results. Each output directory must be fresh. The six-piece real-capture inputs used for qualification are local user data and are not included in the public source or benchmark downloads.

The default cooperative Stage 4 ceiling is 600 seconds. `--seconds 900` permits a longer development test; geometry can stop work earlier. Validation and publication add overhead.

The runner freezes Stage 4 and SDK source, constrains CPU thread pools, prints progress every 30 seconds, seals predictions, independently assesses support and geometry, and checks that upstream inputs and predictions remain unchanged. No reference image or pose is accepted. Shape/support checks are distinguished from reference accuracy.

Results are under `outputs/real_six_v204_cold/`: `report/index.html`, `review.json`, `stage4.log`, `request.json`, `source_inventory.json` and `predictions_sealed.json`.

Compare the sealed result with the original page only in separate post-run evaluation. The earlier saved-assembly diagnostic was not a cold solve; v2.0.4 subsequently passed the fresh six-piece run using this runner.

## Development regression

After reviewing the real case, run the fourteen established puzzles from their saved banks:

```powershell
python -B tools\pipeline.py benchmark `
  --output outputs\v204_development_full
```

This runner seals every prediction before opening evaluation references. It covers two cases at each of 4, 8, 12, 16, 20, 24 and 30 pieces. Real-capture accuracy is a separate qualification.

## Existing Desktop development workspace

The original development workspace contains this repository under `release/paper-reconstruction-v2.0`. From its outer folder, the corresponding local command is:

```powershell
.\.venv\Scripts\python.exe -B release\paper-reconstruction-v2.0\tools\run_stage4_capture.py `
  --upstream outputs\cpu_only_real_six_v201 `
  --output outputs\real_six_v204_cold_repeat `
  --cpu-threads 4
```

Only that existing workstation has the saved upstream folder named `cpu_only_real_six_v201`. Public clones should use their own saved capture pipeline as shown above.

## Telemetry

`search.native_repair.supported_capture_quality_repair` records the source candidate, border-angle spans, new pose counts, native support replays, fine checks, accepted cut moves, rigid-fit objectives, elapsed time, stopping condition and incumbent preservation. New border modes identify `rotation_seed_source=observed-long-straight-contour-spans.v1` and make no upstream-geometry seed claim.
