# Shared-scale capture integration review

Historical v2.0.2 review. Subsequent telemetry investigation found a nested quarter-turn check still blocking continuous-angle graph growth, plus starvation of ordinary search routes. The v2.0.3 corrections and validation are recorded in `V203_CAPTURE_RECOVERY_REVIEW.md`; generation being active in this old run did not establish that its angles reached graph expansion.

The user's acquisition contract is a common physical/native pixel scale: same-DPI scans, or fixed-distance flat phone captures with consistent camera geometry. Rotation and translation remain unknown. “1:1” does not require exact segmentation or pixel-perfect fragment placement. Every fragment of one complete rectangular page is present; page dimensions, format and DPI are unknown.

## Corrected routing and provenance

The v2.0.1 full CPU test did use this repository's current source, but its capture branch selected the older general continuation configuration. V21's page profile additionally coupled shared scale to exact generator masks, and native proposal generation required quarter turns. Those were integration/development restrictions, not consequences of common scale.

Version 2.0.2 separates exact-generator and predicted-mask provenance under the same proper-rigid complete-page policy. Both pipeline routes select `joint_v21_cut_repair_development.json`. Capture declarations remain bound into immutable upstream receipts. Real masks are never labeled generator ground truth.

For continuous rotations, native fitting uses distinct angles from working input evidence plus quarter-turn fallback basins. Angles within 0.25 degrees are represented by the highest-ranked observed mode. Translation is refitted from full observed contours. Scale remains 1.0; reflection, shear and perspective remain forbidden. Existing contact, support, collision, ambiguity and page gates are unchanged. This finite angular coverage is not exhaustive; a failed run cannot prove that the archive or physical pieces have no real construction.

The public example is `profiles/shared_scale_predicted_capture_v1.json`. It declares acquisition assumptions; it does not measure calibration or guarantee photo accuracy.

## Checks and outcomes

- Stage 4: 343 tests passed; 16 historical/archive-dependent tests explicitly skipped. New tests cover mask provenance, immutable binding, scale/perspective guards, non-quarter rigid fitting with deliberately wrong input translation, and native bridge replay. Sixteen portable runner tests pass.
- Four pinned stages and all four frozen model files pass preflight. Both Stage 3 rankers load with Torch 2.7.1+cpu, no CUDA build and CPU tensors.
- Cold exact-mask controls at 4 and 8 pieces pass the construction regression, with zero false selected neighbor pairs. The original fourteen-case baseline is retained; these two controls are the new patch regression sample, not a new fourteen-case qualification.
- First profile-only six-piece V21 rerun: partial components 5+1, unresolved. Telemetry showed native augmentation was still disabled by the quarter-turn restriction. This was not an active continuous-native test.
- Corrected continuous-native six-piece V21 rerun: 603.6 seconds including publication, status `budget_exhausted`, assessment `unresolved`, no verified complete candidate. The native search reached five pieces but the published primary contains six singleton components. This is worse partial-result presentation than the profile-only run and remains Stage 4 work; it is not a successful real-photo assembly.

The final rerun reused the immutable Stage 2/3 evidence produced by the earlier CPU-only six-PDF run. It was cold Stage 4 inference with no old layout/reference seed, not a second fresh end-to-end upstream run. The primary was sealed without opening evaluation truth. Real capture images, masks, poses and previews stay local.

## What the failed real run establishes

Native generation was active for all fifteen physical pairs. The expanded configured vote/fitting work completed: 245 angle modes, 2,148 fit checks and 2,035 retained proposals. Search made approximately 1.21 million proposal checks, reached depth five, and never generated a complete six-piece page for final geometry/support verification. Thus page geometry could not select a hidden complete candidate in this run. Neither completed finite fitting nor the failed bounded search proves physical or bank impossibility.

Observed replay, collision and component-area rejections explain where work went, but do not alone identify the ground-truth root cause. Next work should preserve the best verified partial result, cache equivalent native replay work, inspect the missing attachment's angle/translation modes and predicted boundaries, and independently measure scale/perspective consistency. Stage 3 should retain useful rotation/translation diversity with uncertainty. Any broader angular fitting or segmentation adjustment needs separate provenance and downstream comparison. Increasing timeouts or relaxing verification is not an established fix.

The CPU execution baseline is portable. Reliable real-photo assembly and thirty-piece full-pipeline CPU latency remain unqualified.
