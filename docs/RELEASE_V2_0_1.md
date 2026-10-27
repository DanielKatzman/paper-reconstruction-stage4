# Paper Reconstruction v2.0.1 - CPU-default pipeline

CPU is now the default for installation and the umbrella pipeline. Setup installs CPU-only Torch 2.7.1; CUDA requires explicit opt-in. Frozen stage implementations and the four LFS model files remain unchanged from v2.0.0.

## Verified

- Both frozen Stage 3 rankers load with `torch==2.7.1+cpu`, `torch.version.cuda == None` and CPU tensors. CI checks this on Windows and Linux.
- The four-piece prepared-mask control completes Stage 2, Stage 3 and Stage 4 on CPU and preserves its 5.10-pixel p95 construction accuracy.
- The full pipeline, including fresh segmentation, executes on CPU for a four-piece generator-image test and the user's six real PDF captures.
- The six real captures finish in approximately 9m11s, with all six pieces processed and published in one candidate component.

**Execution is verified; real-photo assembly accuracy is not qualified.** Visual review finds a misplaced/misoriented fragment in the six-piece result. The generator-image test with predicted masks is also incorrectly arranged. The real-capture route uses the earlier general continuation solver; the 14/14 V21 exact-mask result must not be presented as a real-photo success rate.

## Integration fixes

Raw Stage 1 output is preserved in `stage1_native`. A separate audited adapter prepares Stage 2 input, reconciling eight- versus four-connected fringe pixels and retaining a native zero-mask border for contour coordinates. It removes at most 64 pixels and 0.01% of a mask, preserves native scale and crop origins, and requires review when meaningful fragments or capture-edge truncation are found. It does not forge exact-mask capture profiles or perform perspective/scale normalization.

The original v2.0.0 tag and development results remain available. v2.0.1 uses the same model/benchmark assets: [models](https://github.com/DanielKatzman/paper-reconstruction-stage4/releases/download/v2.0.0/models-v2.0.0.zip), [benchmark](https://github.com/DanielKatzman/paper-reconstruction-stage4/releases/download/v2.0.0/benchmark-v2.0.0.zip), [checksums](https://github.com/DanielKatzman/paper-reconstruction-stage4/releases/download/v2.0.0/checksums.json). Git LFS obtains the same four weights when cloning the new tag.

See `docs/CPU_AND_CAPTURE_SCOPE.md`, `docs/REAL_CAPTURE_CPU_REVIEW_2026-10-27.md` and `provenance/CPU_VALIDATION_2026-10-27.json`. Raw real captures and their image results remain local.
