# V21 support-directed repair and persistent native search

This is an opt-in Stage 4 development change. The sealed V20 capacity results, their V19r2 solver snapshot, Stage 2/3 bundles, and all earlier controls remain intact. The final V21 source snapshot is `controls/stage4_v21_cut_repair_development_r4`, with implementation fingerprint `10270c9ef986d15529e05fd2bdc7b8d39675ba598d59b1876be815445084761e`.

## Behavior

`native_cut_repair: true` enables both changes under the existing geometry-driven/deferred-straight policy. It is absent from normalized historical configurations when false. The new preset is `configs/joint_v21_cut_repair_development.json`.

When a full candidate passes page geometry but lacks independent seam corroboration, recovery records the unsupported graph cut. If the cut splits a single bridge of the independent growth tree, it tries existing pair-pose alternatives anchored across the cut. It moves the smaller side as a rigid group, preserving internal relative placements on both sides and retaining the rest of the tree. This works for a singleton or a group; no puzzle/piece identifiers are embedded in the solver.

Same-rotation alternatives within four cycle tolerances receive an early ordering tier; this is 32 native pixels in the current preset. Wider and rotated alternatives remain available. Ranking uses observed native fit and corroboration from distinct placed neighbors. A hypothesis still has to pass native pose/arc replay, collision checks, whole-page geometry, disjoint support assignment and independent serialized-evidence verification. It does not gain acceptance from a rectangular outline alone.

Failed fixed-pose support jobs remain resumable. Successful local changes are independently verified; unsuccessful attempts cannot replace a verified incumbent. A verified but loose page is preserved while search continues toward the already-existing 0.98 observed geometry stopping target. That target is a stopping preference, not a stricter acceptance gate. A verified incumbent survives a deadline or unsuccessful further search.

The native scheduler keeps beam generators, seed cursors, decoded contacts, live DFS frames and their tried alternatives across scheduling turns and pose-bank expansion. Existing beams finish with their original proposal view. Expansion adds new roots and refreshes DFS choices; it does not restart old roots. Beam growth, wider backtracking, unsupported-cut repair and deferred support jobs receive cooperative turns. Scheduling charges each lane for the actual wall time it consumes, including expensive callbacks. The established nonstraight beam receives eight relative scheduling shares, and each other route receives one; the least-consumed normalized active route runs next. These are priorities, not total allowances, and DFS starts before an unsuccessful beam is exhausted. A 0.1-second growth scheduling turn pauses and resumes the same frontier; it is not a total search allowance or a candidate-count cutoff. The overall 600-second cooperative deadline remains configurable through the existing preset.

Duplicate final pose/support histories are avoided across lanes, and their existing support jobs are resumed separately. Cache eviction can repeat work rather than silently declare unexplored branches impossible. Search still does not claim a certified global top-k or exhaustive continuous-pose solution.

New telemetry includes moving cut members, original candidate and proposal IDs, placement displacement, rotation change, native-neighbor corroboration, new contacts and verifier outcomes, proposal-bank revisions, live DFS sessions, actual backtracks, avoided duplicate final histories, per-resume work deltas and the retained verified incumbent.

## Run the full comparison

From the Stage 4 workspace:

```powershell
.\.venv\Scripts\python.exe -B scripts\run_cut_repair_v21_r4.py `
  --output outputs\v21_cut_repair_full_r4 `
  --resume
```

This performs **14 fresh cold Stage 4 solves**, two each at 4, 8, 12, 16, 20, 24 and 30 pieces. Stage 2/3 inference is not rerun. Each child receives only its sealed Stage 3 bundle, frozen configuration and capture/page profiles. No saved assembly layouts, cached Stage 4 proposal banks or reference coordinates seed the solver.

The output is separate from every original run. Final report:
`outputs/v21_cut_repair_full_r4/report/index.html`.

Startup prints progress immediately and verifies the exact serving bundle bytes against their prior seals. It reuses the existing semantic-validation commitments instead of repeatedly decompressing and validating all upstream archives. Each fresh serving worker still validates the input it consumes. The controller binds all input files, frozen source/configuration and its own controller dependencies. Resume rejects changed files and preserves immutable completed/interrupted attempt timings.

After all selected prediction attempts are sealed, a separate frozen reader opens references and compares baseline and V21 previews, contact accuracy, whole-page placement, support replay, native coverage, retained candidates, cut repairs and real backtracking activity. Previous stage diagnoses are reused and labeled; they are not presented as new upstream inference. Unsupported or substantially inaccurate results receive the existing reference-assisted saved-bank diagnostic, separately labeled and excluded from blind success counts. Its failure is inconclusive about impossibility.

For a separate focused run, use a new output folder and add:

```powershell
  --case-id puzzle_000600_139279105ea8
```

Use the same case selection when resuming that folder. A process interruption restarts unpublished in-memory work under the same frozen request; completed publications and attempt timings are preserved. Persistent frontiers refer to uninterrupted solver scheduling and bank expansion, not a claim of saving every Python generator across a terminated process.

## Validation and scope

The final implementation passes 350 tests, including old-policy normalization and frozen-control preservation, straight-cut reconstruction, invalid/uncorroborated seam rejection, expired deadlines, frontier resumption, new-bank reachability, group movement, verified-incumbent preservation, continuation beyond a completed beam when geometry is still marginal, input tampering and the reference-evaluation phase barrier.

The final revision's end-to-end cold smoke run completed:

| Case | Pieces | Worker seconds | Whole-page p95 px | False selected neighbor pairs | Assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| 330 | 4 | 8.65 | 5.10 | 0 | supported |
| 600 | 20 | 313.47 | 30.00 | 0 | supported |

Puzzle 600's internal solver time was 283.46 seconds; normal loading/publication/assessment account for the rest of its worker time. Its final geometry score is 0.985335. The scheduler started 48 DFS seed sessions and performed five backtracks while preserving its beam frontiers. Existing-pose cut repair examined seven alternatives and independently verified two completions; the better verified geometry was retained. The original V20 run spent 612.05 worker seconds and remained unresolved. These are development measurements on a shared host; the user's serial full comparison is the intended performance check.

The selected 20-piece preview was inspected and its serialized support was independently replayed by the frozen post-seal reader. The final smoke report is `outputs/v21_cut_repair_final_smoke_r4/report/index.html`. Earlier development revisions and their isolated smoke folders are preserved for audit; use the revision 4 command above for the final comparison. The full 14-case comparison is deliberately left for the user's run. Do not infer its success from unit tests or a cached diagnostic. A prior revision also checked the difficult eight-piece case 917 and 30-piece case 062 with no false selected neighbor pairs; they still need replay under the final revision in the full comparison.

The inputs retain the development assumptions: all pieces from one complete rectangular page, common pixel scale, unknown format/DPI, exact generator masks and declared quarter turns. Stage 3's measured coverage and latency gaps remain described in `docs/STAGE3_V20_CAPACITY_FEEDBACK_2026-10-27.md`; this change does not alter Stage 3.
