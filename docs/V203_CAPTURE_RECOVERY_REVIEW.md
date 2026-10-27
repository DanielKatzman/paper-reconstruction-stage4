# Continuous-rotation capture recovery

The v2.0.2 real six-piece run exposed Stage 4 integration defects. Its generator and capture declaration accepted continuous rigid rotations, but graph expansion still rejected all non-quarter turns. Independent replay found that 1,612 of 2,095 stored native proposals passed native seam checks but were blocked by that rotation restriction. Native search also exhausted the main search window before ordinary search routes ran, and raw depth-five growth hypotheses did not become verified partial outputs.

## Changes in v2.0.3

One authenticated rotation validator now handles native graph expansion, bridge replay and cut repair. It keeps relative scale 1.0, prohibits reflection/shear/perspective, and enforces quarter turns only when declared by acquisition. Continuous support cycles retain their measured rigid rotation rather than rounding matrix entries. Native ordering retains distinct continuous-angle modes.

For continuous captures, the existing ordinary cold portfolio receives its initial turn before persistent native recovery. Finite working-mode and targeted archive-angle fitting happens before persistent graph search, so a marginal incumbent cannot starve the archive phase. Native frontiers then share time with full support verification, partial support verification and partial cut repair. The overall geometry stopping target/deadline remain the governing limits; partial repair does not claim exhaustive search or impose a new global proposal ceiling.

Partial cut repair revisits bridges inside a placed component, preserves the two sides' internal relative poses, and leaves unplaced components separate. Every replacement is independently replayed and checked for physical/component-area feasibility before the resulting partial reopens native growth. Partial publication builds real selected support records and uses the unchanged independent support/collision checks. A raw five-piece growth state is never labeled a verified page. Supported partial progress ranks ahead of empty singleton output, while complete acceptance still uses the existing full-page gates.

When working modes fail, the capture route inspects the authenticated existing archive for additional angles on unresolved physical pairs. The complete archive iterator must validate successfully before modes are used. Native fitting inherits only angles and stable source IDs, recomputes translation from contours and preserves scale 1.0. Coverage and timing remain explicit; additional modes are not proof of a compatible global construction.

## Validation

The source was frozen before the real-capture and fourteen-case cold Stage 4 runs. Each benchmark primary is sealed before a separate process opens references. The real test reuses the immutable Stage 2/3 evidence from the earlier CPU-only six-PDF pipeline; it does not rerun upstream stages or use an old assembly seed. Real images/results remain local.

- 352 Stage 4 tests pass, with 16 documented archive-dependent skips; 16 portable runner tests pass.
- New integration tests exercise a complete continuous-angle native graph with independently verified cycles, angle-policy negative controls, continuous cut repair, preservation of unplaced pieces, ordinary-route execution, verified partial output, stalled-partial reinjection, validated archive-angle recovery and phase ordering that prevents archive starvation.
- Four-stage source/model preflight and CPU-only ranker loading remain required.

All fourteen cold development controls pass: two each at 4, 8, 12, 16, 20, 24 and 30 pieces. All selected layouts are supported and connected, with zero false selected neighbor pairs. The largest saved-probe displacement remains 30 pixels, with no relative orientation or scale errors. Both 30-piece Stage 4 runs finish in about 120 and 102 seconds. Reference evaluation occurred only after all fourteen primaries were sealed and did not modify them. These remain development controls, not held-out accuracy or whole-pipeline CPU latency measurements.

## Real six-piece outcome

The final source produces a `complete_candidate`, six pieces in one component, with independently verified selected native support and observed assessment `supported`. Stage 4 takes 609.42 seconds including publication under the 600-second cooperative solve deadline. All three ordinary routes execute. Working fitting tests 245 angle modes and retains 2,035 proposals; targeted recovery validates 97 additional archive angles on nine unresolved physical pairs, performs 808 fits and retains 775 additional proposals. The upstream archive is unchanged.

The scheduler reopens 89 repaired partial roots, verifies 41 partial candidates and retains 16. This is progress over v2.0.2's six singletons, but **visual review still finds the small corner fragment misaligned**. The page geometry score is 0.9408 and the existing improvement target is not reached. A supported status is local/physical evidence, not a certificate of correct document content or orientation.

The corner's selected attachment spans approximately 595 native pixels, has shape spread 0.0619 and p95 contact error 9.02 pixels, so it passes the existing local checks. Those checks do not establish that the entire corner is placed correctly. A post-seal diagnostic using the saved incumbent and existing modes can improve geometry to about 0.9518 but still does not give a visually correct corner. That diagnostic is not a blind solve and is not the published primary.

The remaining accuracy work is global outline/orientation consistency and useful angular/translation uncertainty for this attachment. The run does not prove that the complete archive lacks the correct construction. Keep this distinction explicit in Stage 3 feedback; the demonstrated Stage 4 policy/scheduling bugs are fixed, while full real-photo accuracy remains unqualified.

Timings come from concurrent correctness tests on this workstation. They are not isolated speed comparisons, a fresh whole-pipeline timing, or evidence for thirty-piece real-photo CPU throughput. The exact-mask development benchmark and its post-seal reference measurements are recorded in `provenance/V203_CAPTURE_RECOVERY_VALIDATION_2026-10-27.json`. Raw real photos, masks, poses and previews remain local.
