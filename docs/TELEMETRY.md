# Telemetry and failure attribution

The portable runner records a request, immediate per-stage progress, thirty-second heartbeats, subprocess logs, return codes, wall times and log hashes. Stage 3 emits `stage3_profile.json`: ranking/input preparation, model setup/scoring, parent retrieval/fusion, native `fit_parent`, pose `generate`, bundle export and JSON writes, plus physical-pair completion timestamps and peak memory. Inclusive timings overlap; do not add nested rows. Frozen source/model/policy hashes identify the actual producer.

For Stage 3 inspect native `parent_selection.json`, retrieval audit, the `pair_*.json` caches/trace artifacts, the native bundle's archive and the final pooled working edges. Pooling records its unchanged `support-pose-strata.v1` configuration and audit. Counts alone do not describe usable coverage. For a missed physical pair distinguish never retrieved, dropped parent, parent searched but no usable pose, usable mode archived but dropped from working pool, and evidence present but globally incompatible. In the current runtime some prearchive observations are bounded; missing trace detail must be called unknown rather than proof that a candidate was absent.

Stage 4 writes `stage4/result.json` and `stage4/preview.png`. `result.search.native_repair` includes generated native proposals with IDs/transforms/support intervals, bank revisions, lane/frontier/DFS/backtracking counts, completion attempts and rejections, scheduler consumption, unsupported cuts, smaller-side member IDs, tested pose deltas, corroboration and independent final verdicts. `result.resources`, `termination`, `rejections` and all candidate layouts give runtime and search outcomes. Stage 4's generated native pose bank is distinct from the Stage 3 archive; it can repair missing upstream pose coverage using observed contours.

`stage4.assessment/assessment.json` gives the observed support/geometry status and reasons. Its `feature_accounting.native_repair` contains additional native-search details. `pose_recovery_request.json` identifies weak/unresolved physical cuts using observable evidence. `manifest.json` authenticates the published result and assessment. Fine page geometry measures rectangle fill, overlap, holes, area and orientation without knowing page format/DPI. Geometry alone does not certify correct content or orientation. Candidate top-k only contains layouts search actually generated; it cannot select a missing global arrangement.

The benchmark seals all prediction files in `predictions_sealed.json` before any reference evaluation. `evaluation.json` then records neighbor precision, strict four-pixel measurements and `quality.common_gauge` per-piece orientation, scale and displacement. All pieces use one shared alignment; disconnected fragments are not independently aligned into apparent success. The supported/connected/neighbor flag is reported separately from `construction_regression_pass`, which also checks complete inventory, ≤48-pixel maximum saved-probe displacement and numerically correct relative orientation/scale. These are explicit offline regression checks, not serving thresholds or a substitute for preview review. The original fourteen baseline pages received independent native replay and full visual review.

## Was a real assembly present in the bank?

Run an explicit **offline** diagnostic only after a benchmark prediction barrier exists:

```powershell
python -B tools/diagnose_bank.py --run outputs/v2_full_regression `
  --case puzzle_000600_139279105ea8 --seconds 60
```

This checks all prediction hashes, opens the bound evaluation reference, filters only already-saved native proposals at several tolerances and attempts an independently verified compatible construction. It never supplies reference placements, creates new pair fits, seeds a primary or changes its result. A successful reference-consistent witness proves that the saved bank could form such an assembly. Failure, graph disconnection at a tolerance or timeout is inconclusive: this bounded search cannot prove no assembly exists. Its diagnostic 48-pixel p95 criterion is not a serving acceptance rule.

An accurate connected pair graph is necessary evidence but not a global compatibility proof. A disconnected strict-eight-pixel graph may still permit approximately correct reconstruction at a looser tolerance. If a positive witness exists but the primary fails, investigate Stage 4 path choice/support/verification and scheduling. If useful parents/modes are missing upstream, improve Stage 3 coverage. If paths exist but collide or reuse incompatible seam intervals, inspect the actual contact witnesses. Never assign the cause solely from the final image or elapsed time.

The fourteen-case original review independently replayed native joins and physical checks. The `active_joins_checked: 0` legacy counter does not mean native support was skipped; the separate native verifier checked 392 joins. Current approximate collision checks allow small overlaps and are not a pixel-perfect certificate.
