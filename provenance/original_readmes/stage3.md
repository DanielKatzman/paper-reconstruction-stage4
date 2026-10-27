# Paper reconstruction — Stage 3 alpha

Stage 3 turns a page's accepted Stage 1 pieces and frozen Stage 2 boundary predictions into observable neighbor/pose alternatives for a **separate Stage 4 assembly project**. The handoff is a portable folder of JSON, compressed JSONL, PNG and NPZ files. There is no HTTP service or shared database requirement.

This is alpha **0.1.1**, software **0.9.32**. It preserves the v0.9.31 frozen-core inference recipe while removing research orchestration, dataset dependencies and optimizer state. It contains two frozen models, the runtime, an independently installable contract SDK, tests and a synthetic example. Experimental correction scoring is disabled. Reliable solved-piece capacity is unknown; 20 pieces is the baseline target and 30 the milestone.

## Use the existing Windows environment

From the project root:

```powershell
.\scripts\run_stage3_windows.cmd
```

With no arguments this checks the packaged models and policy. It does not train, reconstruct a page or read a dataset. The existing GTX 1070 environment uses Torch 2.7.1+cu118 and is preserved.

Run the small synthetic example through Stage 3:

```powershell
.\scripts\run_stage3_windows.cmd infer --stage1 examples\synthetic_page\input\stage1 --stage2 examples\synthetic_page\input\stage2 --output outputs\synthetic_page --device cuda --puzzle-id synthetic_page
```

For a real page, replace the two input directories with matching Stage 1/2 exports and use a new output directory. `--device cpu` is explicit; CUDA never silently falls back. Repeating an identical interrupted command reuses committed rankings and physical-pair caches. Source/policy/code changes require a new output directory. New runs report the source and implementation identities.

Inspect the result:

```powershell
.\.venv\Scripts\python.exe scripts\stage3.py inspect --bundle outputs\synthetic_page
```

## Stage 4 boundary

The complete input contract, schemas and independent reader are in [contracts/stage3_to_stage4/v2](contracts/stage3_to_stage4/v2/README.md). Stage 4 receives only the completed evidence folder. It needs numpy, Pillow and jsonschema; it does not import Stage 3 or torch.

```powershell
.\.venv\Scripts\python.exe examples\stage4_consumer.py examples\synthetic_page\evidence
```

This validates and reads the handoff. A full Stage 4 solver is **not** implemented in this repository. The earlier forest control remains in the local research archive. For a new development chat, start with the [Stage 4 development handoff](STAGE4_DEVELOPMENT_HANDOFF.md), which includes the project context, first implementation sequence and a starter prompt. [Stage 4 integration](docs/STAGE4_INTEGRATION.md) describes the project boundary.

## Fresh clone and tests

For the Stage 4-driven speed investigation, see [profiling](docs/PROFILING.md). The separate development launcher `scripts/run_stage3_profile_windows.cmd` profiles one physical pair from the saved local smoke case and checks exact evidence equivalence. It leaves the production runtime/model policy unchanged; the longer profile is user-run.

The completed profile identified repeated RGB correspondence work as the main pair cost. The first [exact evidence acceleration candidate](docs/EXACT_EVIDENCE_V1.md) runs in an isolated source snapshot. Its full four-piece qualification preserved all prediction/support/score objects and the working pool exactly, with observed Stage 3 time reduced from 472.6 to 94.6 seconds. It is not promoted as the default producer. The [candidate-run handoff for the existing Stage 4 chat](docs/STAGE4_CANDIDATE_RUN_HANDOFF.md) records the eight-piece case, opt-in integration and feedback protocol. `scripts/run_stage3_exact_windows.cmd` can repeat the four-piece qualification.

The eight-piece feedback and Stage 4 structural-v2 follow-up now motivate a [saved-evidence pool-diversity replay](docs/POOL_DIVERSITY_REPLAY.md). `scripts/run_stage3_pool_replay_windows.cmd` tests a new bounded pool against the unchanged consumer and produces a selection/support audit, reusing upstream evidence without repeating Stage 2/3 inference.

The user's preview review identified a displaced attached group. The [attached-group alignment experiment](docs/ATTACHED_GROUP_ALIGNMENT.md) holds the main assembly fixed, tests placements at one connection, then permits limited cleanup of the moved pieces. Run `scripts/run_stage4_group_alignment_windows.cmd --bridge piece_004 piece_008` to test that focused correction on the completed pool replay.

That rigid-group trial produced only a negligible adjustment. The next [joint attached-group experiment](docs/JOINT_ATTACHED_ALIGNMENT.md) audits collisions/support blockers and tests bridge and internal-join alternatives together, keeping the main assembly fixed. Run `scripts/run_stage4_joint_alignment_windows.cmd --bridge piece_004 piece_008`. Its feedback explicitly distinguishes a numerical adjustment from an evaluated alignment improvement.

The subsequent ten-puzzle Stage 4 validation identified a Stage 3 feature/image cache capacity cliff. The isolated [byte-bounded feature-cache-v2 candidate](docs/BYTE_BOUNDED_CACHE_V2.md) retains the v1 calculations/models and adds bounded reuse and memory counters. Run `scripts/run_stage3_cache_windows.cmd` for full native-evidence qualification on three eight-piece cases and the completed twelve-piece probe, or add `--check` for preflight only. The old candidate and Stage 4 routing remain pinned until the new qualification is reviewed.

The four-case cache qualification preserved complete evidence but saved only 1.6% aggregate inference time. The next isolated [exact visual-reuse-v3 candidate](docs/EXACT_VISUAL_REUSE_V3.md) reuses query-side image preparation and identical zero-offset boundary matches. Run `scripts/run_stage3_visual_windows.cmd` to qualify the same four cases against native v1 evidence, with the completed cache run as a saved timing control. Add `--check` for preflight only. Default producer and Stage 4 selection remain unchanged.

Visual-v3 subsequently passed all ten eight-piece exact-evidence comparisons, with 15.5% less observed aggregate inference time than v1. The [current Stage 4 integration, CPU deployment and quality handoff](docs/STAGE4_QUALIFIED_V3_CPU_AND_QUALITY_HANDOFF_2026-10-23.md) records the qualified identities, reusable evidence, adapter requirements and next experiments. Its companion JSON contains the per-case paths and receipts. Stage 4 replay and full CPU qualification remain pending.

Stage 4 v9 has since reached six recognizable eight-piece development constructions and preserved the accepted twelve-piece anchor. The next [parent-diversity-v4 accuracy experiment](docs/PARENT_DIVERSITY_V4.md) changes parent selection under the same local budgets, records raw pose losses, and compares new evidence with the frozen v9-retained consumer. Run `scripts/run_stage3_parents_windows.cmd` for the seven-case guard cohort; `--check` performs source and parent-selection checks only. This candidate is not promoted; full CPU qualification and consistent ten-of-ten eight-piece reconstruction remain open.

The completed parent-v4 guard run regressed accepted anchors, so v3/v9 remains the clean-capture control. The [shared-scale benchmark](docs/CLEAN_CAPTURE_BENCHMARK_V1.md) contains two cases each at4/8/12/16 pieces, with pixel-exact original content, quarter-turn rotations and exact-mask Stage1 bypass. Run `scripts/run_clean_capture_windows.cmd` for fresh Stage2/v3 and cold v9 on4/8 first; add `--check` for preflight or `--sizes 12 16` for the larger probes. That first run changes capture alone. Its review motivates the separate [unit-scale-v5 geometry comparison](docs/UNIT_SCALE_V5.md): `scripts/run_stage3_unit_scale_windows.cmd` reuses authenticated clean rankings, generates rigid native-scale poses and runs the same cold v9 consumer. Add `--check` for preflight. Review4/8 before running the larger probes.

Use Python 3.12. Create a virtual environment and install `requirements.txt`; it installs the local contract SDK and Stage 3 package. `constraints.txt` records the tested library versions. Install a suitable Torch build for the machine; keep the existing CUDA build when continuing on the GTX 1070.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\test.py
```

The checkout scripts also work directly with an already compatible environment. See [validation](docs/VALIDATION.md), [development](docs/DEVELOPMENT.md) and [cleanup inventory](docs/CLEANUP_INVENTORY.md).

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/paper_reconstruction_stage3/` | Observable inference, frozen architectures, geometry and export |
| `models/` | Two inference-only checkpoints and sealed policy/model identities |
| `contracts/stage3_to_stage4/v2/` | Versioned file contract, JSON Schemas and independent CPU SDK |
| `scripts/` | CLI, Windows launcher and test runner |
| `examples/` | Synthetic Stage 1/2 input, matching evidence and independent Stage 4 reader |
| `tests/` | Contract/geometry/export tests and v0.9.31 synthetic parity references |

Large experiments, datasets, caches, environments and the local research archive are excluded from Git. They are not needed to run or develop this alpha.
