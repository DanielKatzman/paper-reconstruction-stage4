# Paper Reconstruction v2.0.3 - Continuous-rotation capture recovery

Native graph growth, bridge replay and cut repair now honor the authenticated rotation family consistently. Continuous rotations keep scale fixed at 1.0; quarter-turn captures retain their declared restriction. Support cycles preserve measured rigid angles instead of rounding matrix entries.

The capture path gives ordinary search routes an initial turn, independently verifies partial output, reopens growth after repairing earlier partial attachments, and fits targeted angles from the existing authenticated archive. Finite angle preparation precedes persistent search so a marginal incumbent cannot starve archive recovery. Serving contact, collision and page acceptance thresholds are unchanged. CPU remains the default; Stage 3 source and all four LFS model weights are unchanged.

Validated: 352 Stage 4 tests passed, 16 documented skips, 16 portable runner tests passed, four-stage/model preflight, and fourteen cold development construction controls through 30 pieces. All benchmark primaries are sealed before reference evaluation.

**Real-photo accuracy is still experimental.** Cold Stage 4 on the six photographed pieces now publishes one supported six-piece candidate in about 10m9s, using the earlier immutable CPU-produced upstream evidence. Visual review still finds a misaligned small corner. The run fits 97 additional archive angles and retains 775 additional native proposals, but does not reach its geometry improvement target. Supported status is not a certificate of correct page content or placement. This is not a fresh end-to-end timing or a thirty-piece real-photo qualification.

See [the capture recovery review](https://github.com/DanielKatzman/paper-reconstruction-stage4/blob/v2.0.3/docs/V203_CAPTURE_RECOVERY_REVIEW.md) and the updated Stage 3 handoff. Real user photos/results remain local.

The original tags remain intact. This release uses unchanged v2.0.0 assets: [models](https://github.com/DanielKatzman/paper-reconstruction-stage4/releases/download/v2.0.0/models-v2.0.0.zip), [benchmark](https://github.com/DanielKatzman/paper-reconstruction-stage4/releases/download/v2.0.0/benchmark-v2.0.0.zip), [checksums](https://github.com/DanielKatzman/paper-reconstruction-stage4/releases/download/v2.0.0/checksums.json). Git LFS obtains the same four pinned weights with this source tag.
