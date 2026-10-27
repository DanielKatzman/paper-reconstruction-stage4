# V21 full benchmark review

The completed `outputs/v21_cut_repair_full_r4` run reconstructs all 14 development puzzles correctly for the project's approximate-construction goal. This includes 228 pieces in total and both 30-piece pages. Every selected layout contains all its pieces in one component, has zero false selected neighbor pairs, and agrees with the reference arrangement with no relative rotation or scale errors. Remaining placement offsets are at most 30 pixels at the evaluated probes across the entire cohort. All 14 selected previews were inspected manually and form coherent pages.

This validates this development benchmark, under its declared complete-page, shared-scale, quarter-turn and exact-generator-mask assumptions. It does not establish a success rate on unseen pages or real camera captures. Several of these pages were used to diagnose and tune the solver, so this is a regression baseline rather than a held-out reliability evaluation.

## Verification

The review authenticates the request, all bound serving input files, frozen source/configuration, prediction barrier and completed report receipt. It independently reloads every input with the exact frozen reader and replays native contacts, interval exclusivity, weak-bridge corroboration/page-boundary witnesses, final physical feasibility, page geometry and reference evaluation. The independently recalculated reference measurements match the saved report after normalizing JSON pair tuples/lists. The audit does not run a solver or alter any prediction.

Each case has one completed cold serving attempt, with no pre-existing published result. Recorded commands supply no saved assembly or reference coordinates. Stage 2/3 evidence is reused; none of those inference stages was rerun by V21. Reference-aware diagnosis follows the sealed prediction phase, and no reference-assisted diagnostic replaces a primary result. Every primary passes without needing a constructibility probe.

The source fingerprint is `10270c9ef986d15529e05fd2bdc7b8d39675ba598d59b1876be815445084761e`. The unchanged implementation had passed 350 tests before this full run. Its existing acceptance gates remain intact.

The fresh replay verifies 392 selected native joins and all 2,242 within-page piece pairs, with all 14 fine geometry checks passing. No page-boundary bridge exemptions are used by these selected layouts. The report's `active_joins_checked: 0` field counts trimmed legacy active supports only; it does not mean the native joins were skipped. Physical checks use the configured bounded native-interior sampling and permit small overlaps; they are not a claim of pixel-perfect, overlap-free reconstruction. All 507 local report links resolve. The 182 sealed prediction files and 241 report files match their receipts.

## All results

Times below are measured Stage 4 subprocess wall time, including input loading, publication and assessment. The internal solve time is lower; report generation is separate. Page p95 uses one common gauge for the entire page, not per-piece alignment.

| Pieces | Puzzle suffix | Baseline worker s | V21 worker s | Baseline page p95 px | V21 page p95 px |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | 330 | 6.86 | 6.80 | 5.10 | 5.10 |
| 4 | 450 | 6.67 | 6.80 | 3.16 | 9.22 |
| 8 | 678 | 18.49 | 18.59 | 4.12 | 4.12 |
| 8 | 917 | 20.30 | 18.18 | 18.97 | 27.31 |
| 12 | 341 | 58.34 | 58.30 | 21.95 | 21.95 |
| 12 | 362 | 29.58 | 30.96 | 5.39 | 5.39 |
| 16 | 681 | 43.15 | 46.46 | 3.16 | 3.16 |
| 16 | 801 | 122.74 | 246.43 | 17.26 | 17.26 |
| 20 | 600 | 612.05, unresolved | 162.63 | Unavailable: partial page | 30.00 |
| 20 | 654 | 61.04 | 62.73 | 13.15 | 9.06 |
| 24 | 187 | 101.18 | 89.83 | 10.05 | 15.03 |
| 24 | 976 | 79.79 | 167.27 | 31.58 | 18.36 |
| 30 | 062 | 122.70 | 125.03 | 21.26 | 22.83 |
| 30 | 847 | 103.77 | 105.77 | 18.03 | 11.66 |

All 14 V21 rows are supported, with zero false selected neighbor pairs. The worst evaluated per-piece displacement is 30 pixels; there is no hidden substantially misplaced fragment excluded by the page percentile. The reference for 976 is mostly blank with a printed header/border; its blank-looking assembly is consistent with that reference. It is a useful shape/ambiguity case but does not replace testing additional content-rich 24-piece pages.

Total Stage 4 worker time is 1,145.77 seconds, about 19 minutes 6 seconds for all 14 pages. Both 30-piece cases take about two minutes, and the slowest page is 801 at 4 minutes 6 seconds. Every Stage 4 case is comfortably below the ten-minute target.

## What improved, and what did not

Puzzle 600 is the main correctness improvement: its previously unresolved eight-component fallback becomes a supported complete 20-piece page in 162.63 seconds. The first unsupported 015 cut requests more observed modes; the existing 003/015 alternative moves 015 by 11.05 pixels and passes the normal final checks. This full run needs one local alternative, with no reference information used by the solver. It reproduces the approximately correct construction demonstrated during the earlier investigation.

The other 13 cases remain correctly constructed. Pixel alignment improves on 654, 976 and 847; it is unchanged on six cases and slightly worse on four. The largest increase in page p95 among previously successful cases is 917, from 18.97 to 27.31 pixels. Its pieces still occupy the correct places and satisfy the stated construction goal. V21 is not uniformly more precise, and greater subpixel precision is not the user's target.

There are two material speed regressions:

- **801: 122.74 to 246.43 seconds.** Pair fitting costs are almost unchanged, about 50 seconds in both versions. Native growth/support work grows from about 45 seconds to about 170 seconds. V21 preserves an older beam, maintains wider search/support jobs and verifies a seven-piece group alternative before selecting a page with the same measured placement accuracy. This is scheduling/search overhead, not a missing Stage 3 input or new assembly failure.
- **976: 79.79 to 167.27 seconds.** V21 requests expanded native fitting after an unsupported initial cut; that adds 75.22 seconds of fitting that the baseline did not need. Repair then moves piece 001 by 21.19 pixels and completes verification in roughly two seconds. Alignment improves from 31.58 to 18.36 pixels, at a real runtime cost.

The aggregate cohort is about 17.4% faster than the earlier total, but that earlier total includes the 612-second failed 600 run. On the 13 previously successful controls alone, worker time rises from 774.61 to 983.15 seconds, about 26.9%. The defensible conclusion is improved completeness with localized speed costs, not a universal speedup.

## Next steps

1. Preserve V21 revision 4, the 14-case dataset/input commitments and these results as the construction baseline before further algorithm changes. Project cleanup and GitHub publication can now be planned around a known passing baseline; no push or destructive cleanup was performed during this review.
2. Prioritize Stage 3 latency and useful pose coverage. The reused upstream records show Stage 3 native inference plus pooling taking 52.2 minutes for 062 and 45.5 minutes for 847, versus about two minutes for Stage 4. The Stage 4 portion meets the speed target; the complete pipeline still needs substantial upstream acceleration.
3. Keep 801/976 as explicit Stage 4 performance regressions. A future version should interleave new-bank beam roots with the preserved older frontier, and try cheap existing-mode repair before automatically paying for all expanded fits when sufficient evidence is already present. Preserve the accepted incumbent, 600 recovery and normal verification gates. Avoid simply adding more search time or imposing a truth-based accuracy threshold.
4. Validate generalization with additional disjoint, unseen 20–30-piece pages, including content-rich 24-piece examples. Then qualify real segmentation/capture conditions separately. Do not expand the benchmark and change the solver simultaneously when measuring that reliability step.

## Saved evidence

- Original comparison: `outputs/v21_cut_repair_full_r4/report/index.html`.
- Independent replay: `outputs/v21_full_validation_review_2026-10-27_r2/review.json` and per-case `audit.json` files.
- Reproduction script: `research/review_v21_full_run.py`.
- Manual inspection and review receipt: `research/V21_FULL_BENCHMARK_REVIEW_RECEIPT_2026-10-27.json`.
- Cross-stage feedback: `docs/STAGE3_V21_VALIDATED_BASELINE_HANDOFF_2026-10-27.md`.

The first review folder without `_r2` is an incomplete checker attempt: it stopped because the Python evaluator represents some pair IDs as tuples while saved JSON uses equivalent lists. The second attempt compares canonical JSON representations and replays all cases. This was a review-checker issue, not a discrepancy in a prediction or metric. Original reports and predictions were preserved.
