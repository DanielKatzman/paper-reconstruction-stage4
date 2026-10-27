# Stage 3 handoff after the complete V21 validation

Stage 4 now passes all 14 known development cases: two each at 4, 8, 12, 16, 20, 24 and 30 pieces, 228 pieces total. Every primary result is a supported single page with zero false selected neighbor pairs. Fresh independent replay validates native supports, collisions and geometry, and reference checks find no relative rotation/scale errors. The largest evaluated placement displacement across the cohort is 30 pixels. All selected previews were inspected.

This is a passing regression baseline, not a held-out reliability result. The complete-page/common-scale/exact-mask/quarter-turn assumptions remain explicit, with unknown page format and DPI. Approximate correct placement is the goal; subpixel fitting and exact four-pixel whole-page agreement are not required.

All paths below are relative to `C:/Users/danka/Desktop/paper-reconstruction-stage4`:

- Full review: `docs/V21_FULL_BENCHMARK_REVIEW_2026-10-27.md`.
- Original report: `outputs/v21_cut_repair_full_r4/report/index.html`.
- Independent replay: `outputs/v21_full_validation_review_2026-10-27_r2/review.json`.
- Frozen source: `controls/stage4_v21_cut_repair_development_r4`.
- Implementation: `10270c9ef986d15529e05fd2bdc7b8d39675ba598d59b1876be815445084761e`.
- Existing upstream inputs/profiles for the larger cases: `outputs/v20_capacity_inputs/cases/`.

## Stage 4 changes relevant upstream

Stage 4 keeps persistent native beam/DFS frontiers and support jobs across scheduling turns and bank expansion. A geometry-passing page with an unsupported cut can now reopen that specific placement using existing observed native modes. It moves a singleton or group rigidly, preserves the other placements and reruns ordinary independent verification. Scheduling charges actual consumed time so expensive support work does not starve growth. A verified incumbent is preserved while the existing observed geometry target is pursued.

The former failure, 20-piece puzzle 600, now completes in 162.63 worker seconds. It uses an existing 003/015 pose alternative to translate piece 015 by 11.05 pixels. That permits independent corroborating contacts to 002, 009 and 012. No reference pose, old assembly seed or new learned inference is used. A separate 24-piece case, 976, completes after a 21.19-pixel existing-mode correction to piece 001. A seven-piece group alternative is also independently verified during 801's search, although its final selected result comes from further growth with better page geometry.

This confirms that nearby ambiguous translations need to remain available with explicit uncertainty. An excellent single-seam residual does not necessarily resolve sliding. Useful pose coverage and cross-piece consistency are more important than making every local fit numerically exact.

## Main remaining bottleneck

The 30-piece Stage 4 worker times are 125.03 seconds for 062 and 105.77 seconds for 847. The reused original Stage 3 measurements are:

| Case | Stage 3 native inference | Pooling | Stage 4 new cold worker |
| --- | ---: | ---: | ---: |
| 062, 30 pieces | 2,820.56 s | 313.49 s | 125.03 s |
| 847, 30 pieces | 2,476.09 s | 254.96 s | 105.77 s |

Stage 2 takes roughly 15–20 seconds per larger page. No Stage 2/3 inference was rerun in V21; the upstream times above must be treated as reused measurements. They identify a much larger end-to-end opportunity than shaving small placement offsets in Stage 4.

## Useful upstream coverage gaps

These are post-seal reference diagnostics on true contacts. Eight pixels is a diagnostic direct-pose tolerance, not a proposed serving acceptance threshold. A connected accurate-pair graph still would not prove compatible global support.

| Case | True contact pairs | Pairs with retained Stage 3 parents | True pairs with an archived pose within 8 px | Accurate archive graph component sizes at 8 px |
| --- | ---: | ---: | ---: | --- |
| 600, 20 pieces | 44 | 29 | 11 | 5,5,3,2,1,1,1,1,1 |
| 654, 20 pieces | 40 | 36 | 15 | 15,2,1,1,1 |
| 187, 24 pieces | 52 | 39 | 20 | 13,6,3,1,1 |
| 976, 24 pieces | 52 | 43 | 11 | 7,3,3,2,1,1,1,1,1,1,1,1,1 |
| 062, 30 pieces | 66 | 46 | 24 | 14,5,3,2,2,1,1,1,1 |
| 847, 30 pieces | 68 | 50 | 29 | 27,1,1,1 |

Stage 4's additional native contour fitting compensates for these gaps, and all six larger cases ultimately assemble. The table does not imply that every tiny junction contact must be retained, or that the Stage 3 archive alone has no construction at looser tolerances. It motivates profiling and recovering useful bridges rather than merely increasing geometry counts.

The concrete 600 parent misses remain good targets: 002/015, 003/015 and 012/015 are retrieved physical pairs with no retained pose-search parent or archived geometry. The long 003/015 shared seam makes this more consequential than a tiny contact at a junction. The best archived 009/015 pose is about 120.6 pixels off. Boundary candidates for pieces 003, 012, 015 and 020 number one each; inspect span coverage together with parent selection rather than assuming the segmentation itself is defective.

Recommended Stage 3 priorities are to profile native fitting and pooling, avoid repeated equivalent work and redundant serialization/validation, preserve informative physical-pair/translation diversity, and improve provenance for never-retrieved versus dropped-parent versus wrong-mode cases. Retain the existing runtime/bundles as controls and compare a new version separately. Stage 4's 801 and 976 speed regressions are tracked independently; they do not invalidate the successful constructions.
