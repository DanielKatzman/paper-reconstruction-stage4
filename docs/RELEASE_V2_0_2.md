# Paper Reconstruction v2.0.2 - Shared-scale capture integration

“1:1” means common relative physical/native pixel scale, with unknown rigid rotations/translations. It does not mean exact masks or pixel-perfect placement.

The capture pipeline now selects V21 with explicit predicted-mask provenance and its existing complete-page/support/collision checks. Native fitting accepts observed continuous rigid angles while refitting translation and keeping scale fixed at 1.0. Exact-mask quarter-turn controls retain their existing route. CPU remains the installation and execution default; Stage 3 source and all four LFS weights are unchanged.

Validated: 343 Stage 4 tests passed, 16 documented skips, 16 portable pipeline tests passed, four-stage/model preflight, both real rankers loaded with CPU-only Torch, and cold 4-/8-piece construction controls passed. The fourteen-case v2.0.0 baseline remains historical evidence, not a new real-photo success rate.

**Real-photo assembly is still experimental.** The six photographed pieces were rerun through cold V21 using the earlier immutable CPU-produced evidence. Continuous native generation was active, but search reached five pieces and exhausted its ten-minute deadline without a verified full page. The published result is unresolved and currently falls back to singleton pieces. It must not be described as a successful real reconstruction or a thirty-piece CPU performance qualification.

See [the shared-scale review](https://github.com/DanielKatzman/paper-reconstruction-stage4/blob/v2.0.2/docs/V202_SHARED_SCALE_PROFILE_REVIEW.md) and the updated Stage 3 handoff. Real user captures/results remain local.

This release uses the unchanged v2.0.0 assets: [models](https://github.com/DanielKatzman/paper-reconstruction-stage4/releases/download/v2.0.0/models-v2.0.0.zip), [benchmark](https://github.com/DanielKatzman/paper-reconstruction-stage4/releases/download/v2.0.0/benchmark-v2.0.0.zip), [checksums](https://github.com/DanielKatzman/paper-reconstruction-stage4/releases/download/v2.0.0/checksums.json). Git LFS obtains the same four pinned weights with the source tag.
