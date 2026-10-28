# v2.0.4 capture alignment and regression review

## Cold real-capture result

Stage 4 reconstructs the six flat, shared-scale photographed pieces as one supported page. All eight selected joins pass independent native support replay and agree with adjacency observed after independent reference registration. The input bundle is unchanged, with 71 Stage 2 candidates, 1,151 Stage 3 geometries, 2,100 occurrences and 360 working edges. No previous reconstruction or reference pose is supplied to the cold solve.

Stage 4 takes 486.8 seconds on CPU. The first supported complete checkpoint appears after about 304.9 seconds; it is still visibly misaligned, with page geometry 0.9396. The quality pass takes 168.5 seconds, proposes 12 observed-border angle modes and measures 147 new native poses. Placed-seam repair improves the final geometry to 0.9953. Fill is approximately 99.33%, overlap 0.094%, and interior holes 0.095% at the bounded assessment raster. These are geometric measurements rather than probabilities of correctness.

The final incumbent passes the ordinary native, physical and page acceptance gates. The stricter improvement target of residual errors within one raster uncertainty estimate is narrowly missed; the tested local repair neighborhood is exhausted before the shared deadline. A verified high-quality page is retained. This is neither an exhaustive global-search claim nor a four-pixel exactness requirement.

The reference page is used only after prediction sealing. Independent registration has approximately 0.97-1.41 native-pixel feature RMS across the six pieces, with additional mask, capture and extrapolation uncertainty. Using one common rigid comparison frame fitted to the other five pieces, the repaired corner has contour p95 displacement 8.53 native pixels, previously about 211 pixels. The other five pieces have p95 displacement 2.96-5.39 pixels. A balanced six-piece comparison frame gives 2.37-6.08 pixels across the six pieces. All reconstruction transforms remain proper rigid motions with relative scale exactly one.

This validates the new cold Stage 4 recovery on this development page. It does not rerun upstream stages or measure the total four-stage pipeline. Raw photographs, masks, poses, reference registrations and previews remain local. A scalar validation receipt is published in `provenance/V204_ALIGNMENT_AND_REGRESSION_VALIDATION_2026-10-28.json`.

## Why the new route is needed

The corner's only Stage 2 candidate covers its complete perimeter, including two exterior page borders. Post-seal comparison of every saved Stage 3 geometry, every original occurrence transform and every Stage 4 native proposal found no closely aligned corner-to-neighbor pose in the old bank. The closest available angle is approximately 11.7 degrees from the registered orientation. Accurate connecting modes do exist for the other five pieces.

Translation refitting at those finite inherited angles cannot supply the missing rotation. The new Stage 4 route derives parallel/perpendicular hypotheses from long observed contour spans under the complete-rectangle contract, refits translations and curved contacts, then independently verifies the resulting page. Border lines themselves never become invented connecting seams. No individual resizing, reflection, shear, perspective correction or original-page guidance is introduced.

A supported full candidate is now eligible for repair even if its current bridge passes local checks. Tree-cut moves preserve internal poses on either side, re-extract native support and retain only verified geometric improvements. Continuous rigid fitting helps polish a plausible mode; it does not replace discrete pose repair. The last verified incumbent survives rejected proposals or timeouts.

## Fourteen-case development regression

All fourteen cold Stage 4 controls pass: two each at 4, 8, 12, 16, 20, 24 and 30 pieces. They reconstruct all 228 pieces in one supported component per puzzle with zero false selected neighbor pairs. Maximum saved-probe displacement remains 30 pixels; relative orientation and scale checks pass. Each primary is sealed before evaluation, and independent review confirms unchanged output hashes and replays support and physical checks from freshly loaded immutable bundles.

The source fingerprint is `6000dccca2004995b02a6b76f21791e5a85b1daabfd235cb755be4761fb69c37`. The source is identical to the qualified development implementation; promotion changes release metadata and documentation. Stage 1-3 source, the SDK and four frozen model files remain unchanged. The historical baseline measurements are retained in `docs/BASELINE.md` and earlier release receipts.

Local checks: 378 Stage 4 tests discovered, 362 passed, 16 documented historical skips; all 19 portable pipeline/runner tests passed. Compatibility and export inventory verification include all four pinned model weights. GitHub CI additionally checks the portable suite and CPU-only ranker loading on Windows and Linux.

This cohort has exact generator masks and quarter-turn captures. It is development data, not a held-out guarantee for unseen pages, predicted masks or 30-piece real photos. The real six-piece result is recorded separately. Neither the original four-pixel diagnostic nor stricter pixel perfection is substituted for correct construction.

## Next improvements

Stage 4's real-case quality pass examines more than 35,000 cut options and about 1,300 fine geometry candidates. Earlier missing-mode detection, less repeated fine raster work and better mode ordering are promising speed targets after preserving this release baseline. Upstream Stage 3 preparation remains a major scaling cost in the existing 30-piece measurements.

Stage 3 should retrieve coherent subarcs from full-perimeter candidates, preserve distinct angle and translation basins, expose exterior-border ambiguity and report pruning reasons. This can reduce Stage 4's fallback fitting and search cost. Keep native support/collision checks and shared scale intact, and qualify changes with cold downstream runs and post-seal measurements.
