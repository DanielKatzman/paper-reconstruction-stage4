# CPU execution and real-capture review

This records the v2.0.1 execution test and its older Stage 4 route. Version 2.0.2 corrects that route to V21 with truthful predicted-mask provenance and observed rigid angles. See `V202_SHARED_SCALE_PROFILE_REVIEW.md` for the separate rerun; the original results and timings below remain historical evidence.

The user confirmed that the six existing PDF inputs represent phone photographs taken at fixed distance with pieces flat. All six were run through fresh Stage 1 segmentation, the audited boundary adapter, Stage 2 inference/verification, CPU-only Stage 3 inference/pooling and Stage 4. No GPU inference was used. The Stage 3 environment was newly created with the CPU-only Torch wheel; `CUDA_VISIBLE_DEVICES=-1` was also set.

| Step | Four-piece prepared masks | Four-piece fresh predicted masks | Six real PDF captures |
| --- | ---: | ---: | ---: |
| Stage 1 | Bypassed | 12.42 s | 26.88 s |
| Boundary adapter | Bypassed | 1.45 s | 4.47 s |
| Stage 2 | 9.47 s | 7.48 s | 16.39 s |
| Stage 2 verification | 1.44 s | 1.19 s | 1.34 s |
| Stage 3 native | 149.98 s | 334.22 s | 469.33 s |
| Stage 3 pool | 5.38 s | 5.20 s | 13.09 s |
| Stage 4 | 6.91 s | 7.59 s | 19.20 s |
| Total subprocess wall time | 173.17 s | 369.56 s | 550.70 s |

The two fresh-segmentation runs overlapped on the same workstation, so these are execution measurements under that load, not isolated latency benchmarks. Stage 1/2 used their validated CPU ONNX interpreters; Stage 4 used its CPU numerical environment; Stage 3 used Torch 2.7.1+cpu with no CUDA build. The CPU prepared-mask control has reference-verified p95 displacement 5.10 pixels and correct construction.

## Accuracy findings

Both fresh-segmentation runs publish all pieces in one candidate component. That is not equivalent to a correct assembly. The real six-piece preview has a fragment in the wrong position/orientation; the rest forms much of a coherent page. The four-piece generator-image/predicted-mask preview is also incorrectly arranged. It has no claim of a real-camera test; its source images are generator crops, and predicted-mask IoU against the exact masks ranges approximately 0.67–0.87.

The real route uses `joint_continuation_development.json`, the earlier general continuation configuration. It does not run the V21 exact-mask complete-page profile. Consequently the 14/14 exact-mask construction baseline neither supplies a guarantee for these masks nor rejects every non-page-looking candidate on this route. Do not change the real capture profile to `generator_exact_bypass` to circumvent that distinction.

## Integration findings and fixes

Stage 1's eight-connected acceptance can include pixels attached only diagonally. Stage 2 requires four-connectivity; two synthetic predicted masks were rejected for tiny groups of 5+4 and 7 pixels. The bounded adapter retains the main component only within strict absolute/fractional limits and records the changes. It removed no foreground pixels from the six real masks.

Stage 1 exports tight crops. Stage 2's half-pixel boundary coordinates could then lie outside Stage 3's accepted native image frame. The adapter adds a zero-mask border inside the original capture and updates bounding boxes/alpha masks with explicit native offsets. Raw Stage 1 files remain unchanged. Meaningful fragment removal or capture-edge truncation requires review.

## Next work

1. Qualify a truthful complete-page profile for predicted masks, with measured uncertainty and native-frame support, so real captures can use the mature Stage 4 geometry/search checks. Preserve the exact-mask baseline and its guards.
2. Diagnose the real result's incorrect attachment using observed contacts, candidate alternatives and a page-geometry assessment. A single connected component does not certify the chosen pose or neighbor identity. No reference-based bank-constructibility proof was performed for this real page.
3. Check acquisition scale/perspective on the actual images. Fixed distance and flatness help, but PDF/scanner-app processing can change scale; the declared rigid/common-scale model remains an assumption until verified.
4. Reduce Stage 3 CPU work after profiling. Stage 2 produces 37 candidates in the prepared four-piece control, 65 with its predicted masks and 71 for the six real pieces. Stage 3's ranking-score timer rises from 89.72 s to 276.78 s and 327.02 s respectively; those inclusive timers contain feature/evidence preparation as well as neural scoring. Investigate redundancy and coverage before merely increasing search budgets.

The release is CPU-executable. It is not yet a validated reliable real-photo reconstruction baseline, and this six-piece runtime does not establish thirty-piece CPU throughput.
