# Stage 3: start here for the v2.0 construction baseline

Stage 4 now reconstructs all fourteen development pages through 30 pieces: two each at 4, 8, 12, 16, 20, 24 and 30, with 228 pieces total. Every primary is supported, connected and has zero false selected neighbor pairs. Independent replay and manual inspection confirm the intended approximate construction. Relative orientations/scales are correct; the largest measured probe displacement is 30 native pixels. **The target is correct fragment placement and a coherent page, not one-to-one or subpixel reproduction.**

The source is now inlined at `stages/stage1` through `stages/stage4`; no sibling source checkout is required. The shared contract is at `contracts/sdk`. Four frozen release models are tracked with Git LFS; benchmark datasets are optional ignored assets with exact receipts. Read `README.md`, `docs/BASELINE.md` and `docs/TELEMETRY.md` before changing inference.

## Current capture integration: v2.0.4

The cold six-piece real-capture test now constructs one supported page in 486.8 seconds on CPU, with all eight selected joins independently verified. Post-seal registration measures approximately 3-9 native pixels of contour p95 displacement per piece. No previous assembly or reference poses enter the solve. The fourteen development controls through 30 pieces also pass, with zero false selected neighbor pairs and a largest saved-probe displacement of 30 pixels. See `docs/V204_CAPTURE_ALIGNMENT_REVIEW.md` and `provenance/V204_ALIGNMENT_AND_REGRESSION_VALIDATION_2026-10-28.json`.

Stage 4 supplies missing rotations from observed border spans and refits native seams, repairs supported complete candidates across tree cuts, and performs fixed-scale rigid refinement with independent support replay. Stage 1-3 source, the SDK and model weights remain unchanged. Existing upstream bundles are compatible; regeneration is unnecessary for Stage 4-only tests. The new Stage 4 source/SDK fingerprint is `6000dccca2004995b02a6b76f21791e5a85b1daabfd235cb755be4761fb69c37`.

Post-seal full-bank review found that the corner's only Stage 2 candidate covers the entire perimeter, including exterior borders. Its needed rotation is absent from all 1,151 Stage 3 archive geometries and 2,100 original occurrence transforms; the closest bank angle is about 11.7 degrees away. Stage 4's observed-border fallback measures 147 new native proposals across 12 angle basins. The other five pieces have useful connecting modes in the existing bank. This is a concrete coverage gap, not an inference from failed search or ordinary graph connectivity.

Priority upstream work: retrieve coherent subarcs from full-perimeter candidates; expose exterior-border ambiguity; preserve distinct angle and translation basins; report rotation-seed provenance and pruning reasons. These can reduce the approximately 168.5-second Stage 4 quality pass and the much larger upstream cost in existing 30-piece preparation measurements. Retain native support, collision and shared-scale checks. Qualify changes with cold downstream runs and separate post-seal reference evaluation.

## Historical capture integration: v2.0.1-v2.0.3

The original construction measurements below belong to v2.0.0. Version 2.0.1 made CPU the installation/execution default and added the audited Stage 1 boundary adapter. Version 2.0.2 corrects the real-capture route to use V21 with explicit predicted-mask provenance, rather than the older general continuation configuration. Stage 3 source and the four model files remain unchanged.

Version 2.0.2 still contained lower-level quarter-turn checks in native graph growth/cut repair and rounded support-cycle rotations. Telemetry replay found 1,612 valid continuous-angle proposals blocked in the real six-piece case. Version 2.0.3 applies the declared rotation family consistently, gives ordinary routes an initial turn, retains independently verified partial support and revisits earlier attachments of stalled partial groups. It also fits targeted angles from the existing archive with freshly measured translations. These are Stage 4 corrections; the failed v2.0.2 run is not evidence of a Stage 3-only coverage failure. See `docs/V203_CAPTURE_RECOVERY_REVIEW.md` for current validation and limits.

The corrected six-piece run now publishes a supported six-piece candidate, but the small corner is still visibly misaligned. It tests 97 additional existing archive angles, retaining 775 derived proposals, and still does not reach the 0.98 geometry improvement target. The corner has a locally valid ~595-pixel contact with p95 9.02 pixels and shape spread 0.0619. A strong local subarc can therefore support a wrong whole-fragment pose. Stage 4 still needs better global outline/orientation handling; Stage 3 should expose useful rotation/translation alternatives and uncertainty rather than treating a short good subarc as a certain full-piece placement. No real-page reference coordinates or full-bank impossibility proof were used. Preserve this page as a separate real-capture test, not another exact-mask success.

The user's “1:1” requirement means common relative physical/native pixel scale, with arbitrary rigid rotation and translation. It does not mean exact masks or pixel-perfect placement. Same-DPI scans or fixed-distance parallel flat phone captures must preserve that common scale. Complete-page shape, format and DPI remain unknown. The exact-mask quarter-turn benchmark is one development acquisition, not the definition of the supported transform model.

For continuous rotations Stage 4 now takes distinct observed working-pool angles, refits translation from native contours and verifies the resulting support/collisions/page geometry. It never inherits an old whole assembly or fits scale/perspective. This is not exhaustive angular coverage: retained observed modes and quarter-turn fallbacks can miss a needed angle. Stage 3 should preserve meaningful rotation and translation alternatives, with uncertainty and parent provenance, especially under predicted-mask boundary error. A failed bounded downstream run alone cannot prove the bank lacks a real construction.

CPU-only real-input timing and the V21 rerun outcome are recorded separately in `docs/REAL_CAPTURE_CPU_REVIEW_2026-10-27.md` and `docs/V202_SHARED_SCALE_PROFILE_REVIEW.md`. Use `compatibility.json` for the current Stage 4 source fingerprint; the fingerprint below identifies the immutable original baseline.

## Exact starting point

- Umbrella release: **2.0.0**. Keep original component version/provenance fields intact when comparing against it.
- Stage 3: qualified **unit-scale-v5**, frozen candidate `6b0ac07074ff3ecf340aa7c30deae6760d8a8cd4807e3dd9b71348d72e86e66a`. Inlined files are from that candidate, not the mutable old root source. Its original manifest is `stages/stage3/provenance/candidate.json`.
- Stage 3 model/config manifest: `stages/stage3/models/alpha.json`, manifest identity `86d547731a0b3b629647d01d657c19170859e47264a79ed1e947c2c1e0b6e59e`. The two frozen rankers are unchanged. Validated interpreter: Python 3.12.10, Torch 2.7.1+cu118 on the existing GPU.
- Stage 4: **V21 revision 4**, source+SDK fingerprint `10270c9ef986d15529e05fd2bdc7b8d39675ba598d59b1876be815445084761e`. Config `stages/stage4/configs/joint_v21_cut_repair_development.json` and profile `shared_scale_complete_rectangle_v1.json`.
- Original SDK/model/component file inventories and final request/barrier commitments live in `provenance`. `baseline/v21r4/review.json` records measured outcomes; `baseline/upstream_profiles` contains six original larger-case Stage 3 profiles.
- Assumptions: all pieces of one complete rectangular page, shared native scale, no perspective/reflection/per-piece resizing, exact generator masks and quarter-turn development captures. Format, dimensions, aspect ratio and DPI are unknown. Do not learn the answer from page IDs or generator coordinates.

The final V21 run reused Stage 2/3 inputs; it was a fresh cold Stage 4 run, not a fresh whole-pipeline timing. The source export additionally checks relocation and fresh small-case inference; its separate packaging receipt makes that scope explicit. The cohort is a regression/development set, not unseen reliability evidence.

## What Stage 4 added

Stage 4 consumes ranked evidence but also fits native contour modes when upstream coverage is insufficient. The current search combines component placement, native bridge creation, page geometry, ambiguity-aware top-k retention and independent support/physical verification. Persistent beam/DFS frontiers and support jobs survive pauses and bank expansion; modes are not discarded just because a scheduling slice ends. Weak near-straight matches require corroboration, preventing outer page borders from serving as convincing false seams. Geometry rejects incomplete/elongated/overlapping layouts once search generates them; it cannot recover an arrangement that search never produced.

V21 adds unsupported-cut repair using **existing observed native modes**. It repositions the smaller side of a selected graph cut as a rigid singleton/group, preserves internal poses and all other pieces, replaces the relevant tree bridge and reruns independent contact/geometry/collision checks. It preserves a verified incumbent while pursuing the existing observed geometry stopping target. The overall development deadline remains a safety limit; the 0.1-second cooperative slices are scheduling granularity, not total search budgets.

The previous remaining failure, puzzle **600** (20 pieces), now passes in **162.63 seconds**. An alternative already present in Stage 4's bank translates piece 015 by **11.05 pixels** through 003/015; this permits corroborating contacts to 002, 009 and 012. The initial seam was excellent locally but ambiguous along a weakly curved span. A 21.19-pixel existing-mode adjustment also completes puzzle 976. These are evidence for preserving meaningful nearby translation alternatives with uncertainty; more numerical precision on the top single seam would not address the ambiguity.

## What Stage 4 most needs from Stage 3

**1. Reduce native pose search and pooling time while preserving useful diversity.** Stage 3 is now the dominant end-to-end bottleneck. Existing measured times are:

| Case | Pieces | Stage 3 native | Pooling | Stage 4 V21 worker |
| --- | ---: | ---: | ---: | ---: |
| 600 | 20 | 1,455.37 s | 141.19 s | 162.63 s |
| 654 | 20 | 1,473.96 s | 158.00 s | 62.73 s |
| 187 | 24 | 1,909.96 s | 199.38 s | 89.83 s |
| 976 | 24 | 2,081.83 s | 180.95 s | 167.27 s |
| 062 | 30 | 2,820.56 s | 313.49 s | 125.03 s |
| 847 | 30 | 2,476.09 s | 254.96 s | 105.77 s |

Stage 2 takes about 15–20 seconds per large page. Stage 4 is below ten minutes on every case, but upstream plus Stage 4 is not. Profile first: reuse prepared contour/image features, batch equivalent fitting, remove repeated physical-pair work, cache immutable receipts/decoded assets within one execution, and avoid serializing/revalidating the same large archive repeatedly. Maintain validation at the trust boundary and prove equivalent outputs for pure execution optimizations. Inspect the high-level inclusive profile before selecting a target; avoid claiming serialization or hashing is the sole cause without measurements.

**2. Deliver useful rigid pose/parent coverage instead of larger raw banks.** Preserve independent physical-pair paths and nearby translation modes on long, nearly straight seams. Use declared common scale throughout fitting. Avoid spending work on scale/perspective degrees of freedom that are forbidden by acquisition. Retain physical-pair/parent diversity before within-parent redundant modes, and measure the effect after pooling. Rank can guide ordering; the solver still needs alternatives when long straight borders or repeated text are ambiguous.

The following are **post-seal direct-pose diagnostics at eight pixels**, not serving thresholds or proof of global impossibility:

| Case | True contact pairs | Retained-parent pairs | Archive pairs with a mode ≤8 px | Accurate archive graph components |
| --- | ---: | ---: | ---: | --- |
| 600 | 44 | 29 | 11 | 5,5,3,2,1,1,1,1,1 |
| 654 | 40 | 36 | 15 | 15,2,1,1,1 |
| 187 | 52 | 39 | 20 | 13,6,3,1,1 |
| 976 | 52 | 43 | 11 | 7,3,3,2,1,1,1,1,1,1,1,1,1 |
| 062 | 66 | 46 | 24 | 14,5,3,2,2,1,1,1,1 |
| 847 | 68 | 50 | 29 | 27,1,1,1 |

Stage 4 compensates for these gaps and still assembles all six. Tiny junction contacts are not all necessary. Improve useful bridges and compatible paths; do not impose retaining every reference contact as a runtime requirement.

For **600**, 002/015, **003/015** and 012/015 were retrieved but had no retained pose-search parent or archived geometry. The 003/015 seam is long (~546 px), with very low shape spread (~0.00268). The best archived 009/015 mode was ~120.6 px off. Stage 2 has one candidate each for 003, 012, 015 and 020: inspect span coverage alongside parent retention rather than automatically blaming segmentation. The Stage 4 bank contained a real construction: a separate reference-filtered existing-pose witness assembled 20 pieces in 3.82 s at page p95 12.21 px. That was diagnostic only; V21 subsequently solved the blind primary without using the reference.

**3. Export actionable uncertainty and provenance.** For every candidate/physical pair expose stable IDs, parent rank, native contact intervals and native-to-native direction, origin stage, score/residual/support length, low-curvature/sliding ambiguity, broad-support/collision outcomes, retained/dropped counts and reason. Distinguish never retrieved, parent dropped, fitted but wrong mode, working-pool pruning and globally incompatible evidence. Record whether scores are calibrated; raw ranker scores are not probabilities. Prefer bounded counters plus targeted traces over writing every primitive fit.

An optional later integration could accept an observable Stage 4 recovery request for an unresolved cut, then generate more modes only for those physical pairs. Treat that as a versioned interface proposal, not functionality the current Stage 3 already implements. Do not pass reference coordinates or an old successful assembly back into serving inference.

## Stage 4 work that remains ours

801 rises from 122.74 to 246.43 seconds with unchanged measured placement accuracy: persistent frontier/support scheduling adds work. 976 rises from 79.79 to 167.27 seconds, mostly a 75-second expanded fitting pass, while alignment improves. Across the thirteen previously successful controls V21 is 26.9% slower. Investigate new-bank root scheduling and cheap existing-mode repair before broad refitting. These are Stage 4 performance issues; upstream coverage improvements may help but should not hide them.

Retain the 600 repair success and all ordinary acceptance checks. Do not solve speed by weakening contact/collision/geometry checks, replacing the approximate-construction goal with a four-pixel rule, or simply increasing timeouts. Full rectangle geometry can accept semantically wrong layouts; retain independent evidence and visual/common-gauge review.

## First experiments and acceptance

1. Authenticate the inlined baseline with `tools/doctor.py`, asset verification and the regression command in `README.md`. Keep the original source inventories, models, cohort and selected config as controls.
2. Start with 600's parent-retention/mode-coverage case plus 062/847's profiles. Separate exact execution/cache changes from altered retrieval/pose policies. Give changed code/configuration new provenance; do not overwrite the baseline receipt or reuse old caches as fresh inference.
3. For an execution-only change, compare ranked parents, native archive/working-pool semantics and inference measurements. For a coverage change, report stage-specific recall/diversity/ambiguity counters and downstream native-refitting demand. Recompute upstream evidence and rerun cold Stage 4 under the unchanged V21 config.
4. Keep all fourteen pages approximately correctly constructed with no false selected neighbor pairs, no relative rotation/scale mistakes and no unsupported/colliding layout. Compare common-gauge probe displacement against recorded controls and review every preview; small offsets are acceptable, large coherent drift is not. Report strict four-pixel numbers separately.
5. Report Stage 2, Stage 3 native, pooling, Stage 4 fitting/search/verification and total wall time separately. Target the largest measured Stage 3 costs first. Preserve regression pages while adding disjoint unseen content-rich 20–30-piece holdouts for reliability; qualify real capture masks afterward.

For future failures use `tools/diagnose_bank.py` after sealing predictions. A positive compatible construction witness proves available evidence; a failed bounded probe cannot prove absence. Keep diagnostic reference use visibly separated from blind success throughout reports and feedback.

## Stage 4 capture quality recovery release

The v2.0.4 release integrates observed-border rotation hypotheses, supported-incumbent tree-cut repair and fixed-scale rigid refinement. This is Stage 4 consumer-side recovery: existing Stage 2/3 bundles remain valid and need no regeneration for the cold capture test. The six-piece capture and fourteen-case regression passed independent review. See [implementation and cold commands](CAPTURE_QUALITY_RECOVERY.md).

The motivating six-piece case exposed a full-perimeter Stage 2 candidate and a missing corner rotation in the entire Stage 3 pose archive. Stage 4 translation refitting at inherited angles could not fix that gap. Long observed border spans can supply new rotation basins, but native curved contact, independent support, collision and page checks remain mandatory. No per-piece scale, reflection, shear or perspective fitting is introduced.

Stage 3 can reduce downstream recovery cost by retrieving coherent subarcs from full-perimeter candidates, preserving angle and translation diversity, and identifying whether a useful mode is absent, pruned from the working pool, or poorly translated. Emit candidate spans, exterior-border ambiguity, rotation-seed provenance and pruning reasons. Coordinate with Stage 2 on perimeter-vs-seam candidate extraction. Stage 4 should still support fallback recovery when upstream evidence misses a pose.

The integration's earlier saved-assembly check was diagnostic. The CPU-only cold runner accepts neither a previous reconstruction nor reference geometry, and subsequently passed the real six-piece test. Keep sealing results before separate reference measurement and preserve the fourteen development controls when changing upstream inference.
