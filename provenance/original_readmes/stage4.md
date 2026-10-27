# Paper reconstruction — Stage 4 alpha

The [shared-scale complete-page review and plan](docs/UNIT_SCALE_PAGE_VALIDATION_REVIEW_AND_PLAN_2026-10-24.md) addresses Stage 3's new rigid `unit-scale-v5` evidence. The user confirmed complete rectangular pages with unknown format/DPI. All four sealed inputs were authenticated and 15,910 matrices independently verified rigid. A substantially better, native-valid recorded four-piece 450 layout exposes a Stage 4 selection opportunity; the bad eight-piece examples still need missing-bridge feedback. The plan adds explicit profile enforcement, mask-area/page/contact evidence, page-aware candidate retention and separate supported/unresolved assessment. This is planning only; frozen v9 and serving code remain unchanged.

The [completed v9 eleven-case review and bridge investigation](docs/V9_FULL_REVIEW_AND_BRIDGE_INVESTIGATION_2026-10-24.md) finds six recognizable eight-piece constructions, four incomplete, and the accepted twelve-piece page preserved. All 76 selected neighbor identities are true. The three user screenshots are 173/979/483; exhaustive saved-pose measurements identify poor bridges and an exact 483 parent-cutoff loss (best fused rank 21, limit 20). More time alone will not change these exhausted expansions. [User-focused findings](outputs/v9_policy_retention_full/review/findings.html) · [Stage 3 actionable handoff](docs/STAGE3_V9_FULL_BRIDGE_HANDOFF_2026-10-24.md). Mean eight-piece consumer time is 98.39 seconds on reused evidence; full-pipeline/thirty-piece capacity remains unqualified.

The [completed v9 guard review](docs/V9_GUARD_REVIEW_AND_NEXT_PLAN_2026-10-24.md) passes all three construction anchors in both variants: recognizable 174, improved 917 and the restored accepted twelve-piece page. Primary geometry and preview bytes match the intended controls exactly. Each route contributes a recorded winning construction. Retained totals 343.89 consumer seconds versus checked 353.17 with identical primaries; native incumbent checks consume about 25% of time. Run the full eleven cases with `--all-cases --variants retained`, then optimize against that broader quality baseline. [Review gallery](outputs/v9_policy_retention_guards/review/index.html) · [Stage 3 results feedback](docs/STAGE3_V9_GUARD_RESULTS_FEEDBACK_2026-10-24.md).

The [v9 retention implementation and run commands](docs/V9_POLICY_RETENTION_IMPLEMENTATION_AND_RUNS_2026-10-24.md) preserve three independent cold search policies and checked feasible incumbents. Both retained and checked cold pilots exactly recover the accepted twelve-piece 341 and improved eight-piece 917 primary geometries and preview bytes on unchanged evidence. This development preset costs extra work (about 123 seconds for twelve pieces, 99–102 for eight); it exposes a larger budget and a separate capped comparison. **195 tests pass** and all eleven cases pass preflight. Run the three-case retained/checked guard, including 174, before the full sweep. [Retained pilot gallery](outputs/v9_retention_pilot_r1/review.html) · [Checked pilot report](outputs/v9_checked_pilot/report/index.html) · [Stage 3 handoff](docs/STAGE3_V9_POLICY_RETENTION_HANDOFF_2026-10-24.md).

The [completed v8 guard review](docs/V8_GUARD_REVIEW_AND_RETENTION_PLAN_2026-10-24.md) confirms real progress on 917 from combined scoring/search: 7/7 true selected neighbors and visibly improved approximate construction. 174 is unchanged. All three variants regress on twelve-piece 341, although the accepted layout remains feasible and scores higher. The next priority is protecting the successful v6 component-search route while retaining the new route; defer the full eleven-case sweep until that regression is fixed. [Review gallery](outputs/v8_support_aware_guards/review/index.html) · [Updated Stage 3 feedback](docs/STAGE3_V8_GUARD_RESULTS_FEEDBACK_2026-10-24.md).

The [v8 scoring/search implementation and commands](docs/V8_IMPLEMENTATION_AND_RUNS_2026-10-23.md) document the **182-test** checkpoint and three independent variants against the frozen faster-v6 consumer. The final frozen source also fixes independent component coordinates after join deletion. All eleven cases passed preflight; the completed guard review above determines the next testing step. [Stage 3 implementation handoff](docs/STAGE3_V8_CONSUMER_IMPLEMENTATION_HANDOFF_2026-10-23.md).

The [completed v7 review and next plan](docs/V7_FULL_REVIEW_AND_JOINT_IMPROVEMENT_PLAN_2026-10-23.md) found no additional recognizable constructions: nine of eleven primary previews/placements are identical, including the accepted twelve-piece page. Case 917 changes to another wrong join. Revalidating an existing better working-pool layout exposes both Stage 4 scoring and search gaps; specific Stage 3 bridge gaps also remain. See the [review gallery](outputs/v7_continuation_full/review/index.html) and [updated Stage 3 handoff](docs/STAGE3_V7_POST_RUN_TARGETED_HANDOFF_2026-10-23.md). The faster v6 consumer remains the working comparison control; no quality default is promoted.

The [v7 implementation and commands](docs/V7_CONTINUATION_IMPLEMENTATION_AND_RUNS_2026-10-23.md) document the separate soft colour/ink cue and its **169-test** implementation checkpoint. The completed cold run is `outputs/v7_continuation_full`. The next proposal separates support-aware search from content-scoring changes and tests both against correct constructions.

The [completed speed comparison](docs/V6_SPEED_REUSE_RESULTS_AND_ACCURACY_PLAN_2026-10-23.md) preserves all eleven primary geometries and preview files exactly, including the user-accepted twelve-piece construction. Recorded mean eight-piece time falls from 49.90 to 38.09 seconds; twelve-piece time falls from 91.91 to 69.22 seconds. Accuracy outcomes are unchanged. The next plan targets 917's ambiguous attachment and the known upstream bridge gaps.

The [execution-speed experiment and commands](docs/V6_SPEED_REUSE_IMPLEMENTATION_AND_RUNS_2026-10-23.md) add exact geometry/proposal reuse, constant-Jacobian checks for affine transforms and phase measurements. The conservative score and search limits are preserved; accuracy changes follow the user's speed comparison. Run `scripts/run_v6_speed_experiments.py --output outputs\v6_speed_reuse_full` against the completed conservative construction baseline.

The [completed conservative full-run review](docs/V6_CONSERVATIVE_FULL_REVIEW_AND_PLAN_2026-10-23.md) restores 174, avoids 483's false attachment and preserves recognizable twelve-piece construction. Eight-piece outcomes remain five recognizable, four incomplete and one gross error; consumer time averages 49.90 seconds. The next plan targets duplicated search and missing fallback evidence. [Stage 3 feedback](docs/STAGE3_V6_CONSERVATIVE_FULL_FEEDBACK_2026-10-23.md) records specific upstream bridge work.

The [cold retention and attachment implementation](docs/V6_RETENTION_IMPLEMENTATION_2026-10-23.md) adds an opt-in shared-budget cold portfolio, component/pose-family exploration, soft conditioned-bridge scoring and bounded join deletion. **139 tests pass.** Use `scripts/run_v3_retention_experiments.py --output outputs\v6_conservative_full` for the next conservative comparison against both previous consumers. The [completed combined experiment](docs/V6_RETENTION_RESULTS_2026-10-23.md) repaired two regressions but lost the twelve-piece construction; its broader configuration is not promoted. The [completed eleven-case v3 review](docs/V3_FULL_BENCHMARK_REVIEW_AND_PLAN_2026-10-23.md) explains the baseline failures; all eleven v3 imports matched their frozen controls. No quality default is promoted.

The [completed ten-puzzle review and twelve-piece probe](docs/VALIDATION_8_TO_12_RESULTS.md) found five recognizable eight-piece constructions, four incomplete assemblies and one gross error. Exact reuse reduced mean Stage 4 time by 33%; an opt-in coarse preset is faster still. The twelve-piece probe exposes relative scale drift, so reliable capacity remains unestablished. The target is recognizable correct construction; small seam errors are acceptable. [VALIDATION_8_TO_12.md](docs/VALIDATION_8_TO_12.md) documents controls, and [JOINT_ASSEMBLY_V3.md](docs/JOINT_ASSEMBLY_V3.md) describes earlier ablations.

The [v3 implementation and run guide](docs/V3_IMPLEMENTATION_AND_RUNS_2026-10-23.md) provides the checked import, separate v5 cold scoring/search ablations and CPU qualification harness. Its earlier implementation checkpoint passed 125 tests. The completed full comparison reproduced all eleven controls; the v5 combined experiment improved the twelve-piece page but regressed eight-piece cases, so it remains opt-in. Existing v1 routes and frozen controls remain available. The [reviewed plan](docs/STAGE4_V3_INTEGRATION_AND_CPU_QUALITY_PLAN_2026-10-23.md) records its design.

Stage 4 consumes a sealed `stage3.assembly-evidence.v2` folder and produces globally consistent candidate layouts, selected geometry/occurrence assignments, unresolved pieces, diagnostics and a preview. It imports the independent contract SDK, NumPy, Pillow, jsonschema and SciPy. It does not import the Stage 3 runtime or Torch.

Version **0.1.0** implements an experimental CPU baseline. **Reliable reconstruction capacity is unknown; 30 pieces remains the milestone.** The current thresholds are development settings, not calibrated confidence or a validated reliability policy.

The [research plan](STAGE4_RESEARCH_AND_PLAN.md) and [paper review](research/ANNOTATED_PAPERS.md) explain the design. See [benchmark and real-piece commands](docs/BENCHMARK.md), [implementation status](docs/IMPLEMENTATION_STATUS.md), [the current upstream map](docs/UPSTREAM_MAP.md), [legacy-control provenance](docs/LEGACY_CONTROL.md) and [the output contract](contracts/README.md).

## Environment

The local `.venv` is already prepared independently of the other stages. To reproduce it on another Python 3.12 installation, from this directory:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -c constraints.txt setuptools numpy Pillow jsonschema scipy
& .\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -r requirements.txt
```

The released contract SDK is vendored byte-for-byte under `vendor/contracts-sdk`, with source receipts in `vendor/ORIGIN.json`. It is a separate installed distribution. No private upstream path is needed to import the solver.

## Small development checks

```powershell
& .\.venv\Scripts\python.exe scripts\test.py
& .\.venv\Scripts\python.exe -m paper_reconstruction_stage4 validate --bundle .\examples\known_grid\evidence
& .\.venv\Scripts\python.exe -m paper_reconstruction_stage4 solve `
  --bundle .\examples\known_grid\evidence `
  --config .\configs\hybrid_development.json `
  --output .\outputs\my_grid_check
& .\.venv\Scripts\python.exe -m paper_reconstruction_stage4 inspect --result .\outputs\my_grid_check
```

The example is an explicit four-piece toy grid with correct and deliberately incorrect alternatives. It is a coordinate/search fixture, not an assembly-capacity benchmark. Green preview lines mark selected native supports. Each disconnected component has its own pose gauge; canvas packing offsets are display-only.

Run its evaluator separately, after the prediction manifest exists:

```powershell
& .\.venv\Scripts\python.exe evaluation\evaluate.py `
  --result .\outputs\my_grid_check `
  --reference .\examples\known_grid\evaluation_only\reference.json `
  --output .\outputs\my_grid_check_evaluation.json
```

## Solver profiles

| Configuration | Behavior |
| --- | --- |
| `configs/legacy_control.json` | Historical join-count-first forest, approximate collision checks and final cycle reranking; occurrence exclusivity is not enforced |
| `configs/partial_baseline.json` | Bounded beam, explicit join/abstention cost, compatible occurrence assignments, contact checks and coarse/final physical checks |
| `configs/hybrid_development.json` | Partial baseline plus bounded triangles/four-cycles, remove/replace repair and optional bounded joint refinement |

One competing transform per physical pair is selected. Correlated occurrences do not become independent votes. The partial objective sums `clip(weight,-1,1) - join_cost` per selected physical pair, then subtracts `0.05 * sum(min((crop-probe pose error / cycle tolerance)^2, 4))`. This uses the source representative's weight for a geometry and a compatible occurrence assignment for support. All score semantics are heuristic.

The current refinement minimizes robust bidirectional proposal-consistency residuals plus a correction prior. It can reconcile redundant selected contacts, but it cannot recover a missing signal from a tree alone. Full homography seeds are preserved through bounded similarity corrections. Every accepted update must pass final collision/warp and support checks and improve the declared partial objective.

Inputs are immutable and outputs must be in a separate directory. `manifest.json` is published only after the result and preview are sealed. A complete candidate is not an accuracy claim. The raw legacy control winner is retained even when stronger final diagnostics reject it; inspect its feasibility flag.

## Budgets and repeating work

The configuration records beam width, proposal/time budgets, input/index limits, assignment/loop/repair caps, refinement limits and collision sampling resolution. Time caps are checked between operations; they are not hard real-time preemption. Input validation, support indexing, final checks and rendering are measured separately from the search budget. Peak memory is observed, not enforced by an operating-system limit.

Use a fresh output path for changed input, code, dependencies or configuration. With unchanged identities, `--resume` verifies a completed publication or replays matching unpublished work from the start. **Search-cursor checkpoint continuation is not implemented.** An interrupted search can publish its last checked candidate with `status=interrupted`; a fresh run is needed to search farther. Interrupted validation/refinement/publication leaves failure/work diagnostics without claiming a ready result. Exit codes: 0 valid published/validated result, 2 failure, 130 interruption. A budget-exhausted candidate is valid output with an explicit termination reason.

## Development benchmark

The accelerated Stage 3 pilot uses **one fixed eight-piece case**:

```powershell
.\.venv\Scripts\python.exe scripts\run_candidate8.py
```

Output: `outputs/candidate8_exact_v1`. This opt-in launcher retains profiling, verifies the candidate and sealed evidence, and collects the pipeline report plus Stage 3 feedback. See [the eight-piece run guide](docs/CANDIDATE8_RUN.md) for preflight, progress and recovery. Preparation did not launch the longer experiment. The commands below retain the frozen Stage 3 control route.

The first run is now complete: 8m42s serving time, but a structurally wrong primary assembly. See [the reviewed Stage 3/4 attribution and improvement plan](docs/CANDIDATE8_REVIEW_AND_PLAN.md). Preserve the completed run; a repeat needs a fresh `--output` directory.

Stage 4 now also has an opt-in structural solver, coordinated repair, observable correction for trees and a bounded archive-expansion ablation. To test the consumer using the completed Stage 3 evidence:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage4_experiment.py
```

This creates a fresh timestamped run and launches only Stage 4, followed by separate diagnostics. It does not repeat Stage 2/3 inference. See [the structural-v2 guide](docs/STRUCTURAL_V2.md) and [the current Stage 3 collaboration handoff](docs/STAGE3_COLLABORATION_HANDOFF.md). The preserved executable consumer control is under `controls/stage4_alpha_0_1_0_review_baseline`.

The prepared cohort contains **32 puzzles / 568 pieces**, four each at **4, 8, 12, 16, 20, 24, 28 and 30** pieces. It reuses existing generated observed images/masks and bypasses Stage 1 segmentation. Each size includes simple/moderate/hard/extreme cuts, selected from the training split before inference. Ground truth stays separate from serving inputs. This is development data, not a holdout capacity measurement.

```powershell
& .\.venv\Scripts\python.exe scripts\run_benchmark.py `
  --output .\outputs\benchmark_development_v2 --stage2-cpu-threads 4
```

Add `--sizes 4` and use a fresh output path for the first four-puzzle pilot. Add `--limit 1` for one case, `--plan-only` to inspect commands, or `--resume` with the same arguments to reuse unchanged work. The runner launches frozen Stages 2–3 and Stage 4, then writes `report/index.html`, aggregate CSV/JSON and per-puzzle diagnostics with images, stage artifacts, checksums and loss localization. See [the detailed guide](docs/BENCHMARK.md) for metrics, limitations and recovery.

## Real-piece full-pipeline testing

A PowerShell orchestrator calls each stage's existing environment and writes into a new run directory. It has been checked with `-PlanOnly`; it has not yet reconstructed the existing real six-piece page.

To run using the six already accepted real pieces:

```powershell
.\scripts\run_full_pipeline.ps1 `
  -RunRoot "$PWD\outputs\real_six_pipeline" `
  -PuzzleId real_six `
  -ExistingStage1 "C:\Users\danka\Desktop\paper-segmentation-stage\stage1_real_puzzle_output"
```

This launches Stages 2–4 using the existing Stage 1 output. Add `-PlanOnly` to inspect commands. To run all four stages on captures belonging to one physical page:

```powershell
.\scripts\run_full_pipeline.ps1 `
  -RunRoot "$PWD\outputs\real_six_full_pipeline" `
  -PuzzleId real_six `
  -Captures "C:\Users\danka\Desktop\paper-segmentation-stage\real_puzzle_captures" `
  -Review flagged
```

Stage 3 defaults to its preserved CUDA runtime. Pass `-Stage3Device cpu` explicitly to choose CPU. The orchestrator does not train, alter frozen models or perform accuracy evaluation. `-Resume` invokes the existing stage-specific resume checks; changed identities require a new run directory. Never put multiple unrelated pages into one assembly input.

Full-archive expansion, new learned/content scorers, general continuous homography optimization, exact global optimization and statistically supported 30-piece capacity remain later research work. They should be chosen from the first real failure breakdown.
