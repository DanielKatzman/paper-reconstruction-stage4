# Paper Reconstruction v2.0.6 - Conservative CPU reconstruction baseline

This release builds on the working v2.0.5 search and keeps measured runtime and
portable-tooling improvements. Experimental development scheduling and extra
search/refinement routes are excluded. Independent native support, collision,
complete-page geometry and relative scale 1.0 remain unchanged.

The clean scanned eight-piece page reconstructs as one supported component on
four CPU threads. First supported checkpoint: **965.2 seconds**;
worker time including verified alignment/finalization: **1218.5 seconds**.
All 12 selected joins pass native verification; post-seal
reference evaluation measures at most 17.07
native pixels of whole-contour relative displacement across selected pairs.
Correct construction is the target; these are diagnostic measurements, not
replacement acceptance thresholds.

## Retained changes

- Vectorize the original greedy rigid-mode duplicate test with identical
  selected IDs and ordering, including strict numerical boundaries.
- Recompute retained primary modes only for physical pairs receiving new
  proposals. Preserve global ordering, representative replacement and all
  fallback modes; retain the published search scheduler and stopping policy.
- Forward CPU thread limits to Stage 1 and save the exact pipeline solver config.
  Add `--stage4-seconds` for explicit development allowances.
- Report complete versus unresolved pipeline output separately from successful
  process execution. Save worker PID/native exit status and enable native crash
  diagnostics without installing a watchdog.

## Qualification

- **14/14 synthetic development cases**, two each at 4/8/12/16/20/24/30 pieces,
  with 228 pieces and zero false selected neighbor pairs. Fresh independent
  native-support and physical replay passes every case. Largest saved-probe
  displacement: 30.0 native pixels.
- **Scanned eight:** complete verified page from cached Stage 2/3 inputs, with
  no previous assembly or reference pose supplied to inference.
- **Real six:** complete verified page in 490.9 CPU worker
  seconds; post-seal per-piece contour p95 range
  2.96-8.53 native pixels.
- **365 solver tests and 26 runner tests passed**, with 16 documented historical
  archive-dependent skips. CPU-only Torch 2.7.1 loads both frozen Stage 3 rankers.
- The runner config-path assertion compares resolved paths, preserving Windows
  short/long temporary-directory aliases. The production solver fingerprint
  and qualified reconstruction outputs are unchanged by this test correction.
- Fresh CPU pipeline smoke: Stage 1 generator handoff reused; new Stage 2,
  Stage 3 native inference/pooling and Stage 4 execution completed successfully.
- On 8,566 saved real proposals across 28
  physical pairs, original and optimized mode retention return identical IDs
  and order. Median subroutine time: 9.061 versus
  0.312 seconds. This approximately
  29.0x subroutine result is **not** an
  end-to-end pipeline speedup claim.

On the actual 5,501-proposal scanned-eight pool, the same check measured
5.185 versus 0.188 seconds (about 27.5x), again with identical IDs/order. Whole
Stage 4 solve times remain comparable to v2.0.5: scanned eight was 1218.5 versus
1211.4 seconds, and real six 490.9 versus a historical 486.8 seconds. This is a
maintenance update with no demonstrated whole-solve speedup. The changed-pair
cache passed incremental equivalence tests but recorded zero reuse hits in the
scanned-eight cold solve; its end-to-end benefit is not established here.

Source/SDK fingerprint: `f22e081ffc83f84b1bf78dd7c563f246df5693a05558fbc3f0203d36b5b56d25`.
Full scalar evidence: `provenance/V206_CONSERVATIVE_CPU_VALIDATION_2026-11-10.json`.

## Compatibility and use

CPU remains the default. Stage 1-3 source, shared SDK and all four Git LFS model
weights are unchanged; saved Stage 2/3 evidence remains compatible. Existing
v2.0.0 model and benchmark downloads are reused. All producer package versions
remain pinned in `compatibility.json`; the umbrella compatible release is 2.0.6.

```powershell
python -B tools/pipeline.py benchmark --output outputs/v206_regression
python -B tools/run_stage4_capture.py `
  --upstream path/to/saved_pipeline `
  --output outputs/v206_capture --seconds 1200 --cpu-threads 4
```

The capture runner defaults to a 600-second allowance; the scanned-eight
qualification deliberately used 1200 seconds. The full pipeline also accepts
`--stage4-seconds 1200`. These remain development controls, not a general
30-piece real-scan reliability claim. Raw real images, masks, reference poses
and previews remain local. References were evaluated after predictions were
sealed and never entered inference.

The next milestone is a separate real-scan cohort with two independent pages
per piece count, common scanner DPI, accepted masks and intact-page evaluation
scans stored separately from solver input. See `docs/STAGE3_HANDOFF.md`.
