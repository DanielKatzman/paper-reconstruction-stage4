# GitHub preparation

This folder preserves the **2.0.0 construction baseline** and the **2.0.1 CPU-default update** for `https://github.com/DanielKatzman/paper-reconstruction-stage4`. Its README explains the inlined upstream snapshots and the future separate API/GUI orchestrator. The original `v2.0.0` tag remains unchanged by the CPU-default update.

Tracked content contains inlined source, packaging/configuration, tests, schemas/contracts, model manifests, source/asset receipts, compact baseline results/previews/profiles, documentation and CI configuration. Historical Stage 4 diagnostic/controller source is retained for regression tests; its old experiment runs and most snapshots are outside this export. The new portable commands are the supported repository-root entry points.

The original Desktop projects, datasets, frozen controls and sealed outputs were preserved. No recursive cleanup, source reset or original-result modification was performed. The source export is separate so publication does not sweep old experiment trees into the new repository.

Ignored content includes `local/`, `assets/`, `outputs/`, virtual environments and unrelated checkpoints. The four frozen release models are explicitly allowed and tracked with Git LFS (about 85.6 MB); they do not become ordinary Git blobs. `local/runtimes.json` and `local/assets-sources.json` hold workstation paths. Model/benchmark ZIPs are separate release assets; their tracked checksums are in `provenance/release_asset_checksums.json`. Source is approximately 13 MB before compression, plus small LFS pointers. `compatibility.json` pins stage metadata, source inventories and model identities together.

`provenance/EXPORT_MANIFEST.json` seals Git-eligible bytes. Verify immediately before staging/pushing:

```powershell
python -B tools/verify_export.py
git status --short
git remote -v
```

`.gitattributes` prevents line-ending conversion in fingerprinted component source, contracts, baseline evidence and provenance. Preserve it when cloning on Windows. Models and input assets are authenticated independently from source; Git alone cannot reproduce an inference run without those assets.

The retained Stage 2 copyright notice is preserved as the top-level license, with its original component copy. No new open-source rights are granted by packaging. Original model-training/source-page provenance stays recorded in component manifests. Source and optional benchmark artifact publication are separate operations.

The original frozen implementation had passed 350 tests. In this portable copy the test runner discovers 350 and passes 334, with 16 documented skips: fifteen require original historical snapshots/output trees or sibling layouts, and one is optional archived Stage 3 parity. One test comparison resolves a temporary-directory path before comparing it with the already-resolved result, so Windows short/long aliases are treated as the same path; solver source is unchanged. `provenance/historical_test_exclusions.json` lists the fifteen explicit portability exclusions. The v2.0.0 release adds eight orchestration/measurement tests; v2.0.1 adds three CPU-default tests and four mask-boundary tests. CPU ranker loading is also checked on Windows and Linux. The original packaging checks are in `provenance/PACKAGING_VALIDATION.json`; CPU execution and real-capture findings are in `provenance/CPU_VALIDATION_2026-10-27.json`.
