# Paper Reconstruction v2.0.0

Stage 4's qualified V21r4 baseline reconstructs **14/14 development puzzles**, two each at 4, 8, 12, 16, 20, 24 and 30 pieces. All selected pages are supported and connected, with zero false selected neighbor pairs. Correct construction is the target; the largest measured probe displacement is 30 native pixels.

## Included

- Exact Stage 4 V21r4 source, including persistent native search frontiers, independently verified bridges, page geometry, ambiguity-aware candidates and unsupported-cut repair using existing modes.
- Qualified Stage 3 unit-scale-v5 snapshot, production Stage 1/2 source, frozen model manifests and the shared contract SDK.
- Portable CLI pipeline, cold regression runner, progress/timing logs, post-seal evaluation and saved-bank construction diagnosis.
- Original baseline measurements, fourteen previews, upstream profiles, source/asset inventories, Stage 3 improvement handoff and the modular monolith integration plan.
- Portable Windows CI: 334 Stage 4 tests pass, 16 historical/archive-dependent tests are explicitly skipped, and eight orchestration/measurement tests pass.

Both 30-piece Stage 4 runs take about two minutes; the slowest Stage 4 page takes 4m06s. Stage 3's reused upstream preparation measurements remain 45–52 minutes at 30 pieces, so this does not claim a ten-minute complete pipeline.

## Downloads

Extract each archive at the repository root, then run `python -B tools/assets.py verify --groups models bundles inputs references`.

| Asset | Size | SHA-256 |
| --- | ---: | --- |
| `models-v2.0.0.zip` | 79,235,835 bytes | `734b15cc227ac84de98f52100d22ed1a3fd90bdcc4b952a5dfaf32ad393c95b7` |
| `benchmark-v2.0.0.zip` | 171,300,445 bytes | `f2442a0b8944b5102ecfc59d52f38a032e71698ffd32e6c5c691d267b383c037` |

The model archive contains the two ONNX models and two Torch rankers. The benchmark archive contains generator Stage 1 inputs, saved Stage 3 evidence and evaluation-only references. Neither archive supplies an old assembly as a serving seed. Full original prediction archives, training collections, environments and private workstation mappings are excluded.

This is a known development/regression baseline under complete-page, common-scale, exact-mask and quarter-turn assumptions. It is not a held-out success rate or a real-camera 30-piece qualification. See the README and `docs/BASELINE.md` for precise scope and installation instructions.

The planned API/GUI/container orchestrator is a future repository. This release supplies its stage baseline, CLI and documented integration boundaries.
