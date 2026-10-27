# GitHub preparation

This folder is the **2.0.0 source baseline** published to `https://github.com/DanielKatzman/paper-reconstruction-stage4`. The release tag is `v2.0.0`; its README explains the inlined upstream snapshots and the future separate API/GUI orchestrator.

Tracked content contains inlined source, packaging/configuration, tests, schemas/contracts, model manifests, source/asset receipts, compact baseline results/previews/profiles, documentation and CI configuration. Historical Stage 4 diagnostic/controller source is retained for regression tests; its old experiment runs and most snapshots are outside this export. The new portable commands are the supported repository-root entry points.

The original Desktop projects, datasets, frozen controls and sealed outputs were preserved. No recursive cleanup, source reset or original-result modification was performed. The source export is separate so publication does not sweep old experiment trees into the new repository.

Ignored content includes `local/`, `assets/`, `outputs/`, virtual environments, checkpoints and ONNX/Torch weights. `local/runtimes.json` and `local/assets-sources.json` hold this workstation's interpreter/source paths. The model/benchmark archives are separate release assets; local copies and their checksums are in `local/release-assets`, with tracked checksums in `provenance/release_asset_checksums.json`. The source-only repository is approximately 13 MB before Git compression.

`provenance/EXPORT_MANIFEST.json` seals Git-eligible bytes. Verify immediately before staging/pushing:

```powershell
python -B tools/verify_export.py
git status --short
git remote -v
```

`.gitattributes` prevents line-ending conversion in fingerprinted component source, contracts, baseline evidence and provenance. Preserve it when cloning on Windows. Models and input assets are authenticated independently from source; Git alone cannot reproduce an inference run without those assets.

The retained Stage 2 copyright notice is preserved as the top-level license, with its original component copy. No new open-source rights are granted by packaging. Original model-training/source-page provenance stays recorded in component manifests. Source and optional benchmark artifact publication are separate operations.

The original frozen implementation had passed 350 tests. In this portable copy the test runner discovers 350 and passes 334, with 16 documented skips: fifteen require original historical snapshots/output trees or sibling layouts, and one is optional archived Stage 3 parity. Their original test bodies remain in place. `provenance/historical_test_exclusions.json` lists the fifteen explicit portability exclusions. Eight new orchestration/measurement tests verify input/output separation, source-path isolation, the prediction/evaluation barrier, detection of prediction mutation and rejection of disconnected, wrongly oriented, scaled or far-displaced reference measurements. Packaging checks and their scope are recorded in `provenance/PACKAGING_VALIDATION.json`.
