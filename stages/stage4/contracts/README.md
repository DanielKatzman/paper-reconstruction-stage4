# Stage 4 result contracts — experimental producer

The original producer ID is `stage4.assembly-result.v1`; the opt-in joint solver uses `stage4.assembly-result.v2`. Both are validated by [assembly-result.schema.json](../src/paper_reconstruction_stage4/schemas/assembly-result.schema.json). This is Stage 4's output, separate from the unchanged incoming Stage 3 v2 contract.

The v2 joint output may add `active_support` to a selected join. It records a versioned consumer policy, original occurrence ID, support digest, raw and active native circular intervals/points, lengths, coverage and extraction diagnostics. Raw incoming `support` remains intact. Publication verifies derived intervals against the input occurrence/contour and enforces contact distance and interval exclusivity. Optional initial serving results also enter the run identity. See [the joint profile guide](../docs/JOINT_ASSEMBLY_V3.md). Existing v1 controls remain readable; frozen old readers are not expected to accept the new solver/schema.

```text
result_directory/
  result.json
  preview.png
  manifest.json       written last in staging; directory published atomically
```

The ready manifest has ID `stage4.result-manifest.v1`. It binds the run ID, each file's relative path, byte length and SHA-256. Its own digest excludes `manifest_sha256`. Canonical bytes use UTF-8 Python JSON, sorted keys, compact separators, ASCII escaping and no nonfinite numbers. `read_result()` verifies every receipt before schema and semantic validation.

The result records the immutable input manifest identity, exact code/SDK hashes, installed dependency versions, configuration and resource/termination diagnostics. Its run ID hashes those inputs; it is not a confidence score. The independent `layout_sha256` excludes runtime/feasibility/refinement telemetry and display packing and identifies the retained layout content.

Each layout partitions the complete physical inventory into components. Native-to-component matrices use column homogeneous coordinates and homogeneous division. The smallest physical ID in each component is the identity gauge. No measured relation exists between separate components. `display_packing` is a separate native-component-to-canvas map used only by the preview.

Each selected join names one geometry, its original B-to-A transform, its final implied transform, and its supporting occurrence. The improved solver requires a checked compatible occurrence assignment. The legacy control instead records null occurrence assignments and explicitly reports that support was not enforced. Additional selected non-tree contacts supply cycle constraints; no guarantee of complete k-best enumeration is made.

`complete_candidate` means all observed pieces are connected. It does not mean correct reconstruction. `partial_candidate`, `budget_exhausted` and `interrupted` preserve physical inventory and explicit termination. Invalid inputs fail before a ready manifest exists. The legacy control can preserve a raw winner whose stronger feasibility report is invalid; downstream consumers must inspect `feasibility.valid`.

`native_probes` are a fixed deterministic sample of observed mask pixels, not truth or recovered reference coordinates. Confidence stays uncalibrated; demonstrated reliable capacity remains null. The serving result always records `evaluation_truth_opened=false`.

## Separate evaluation input

`evaluation/evaluate.py` first verifies a committed prediction, then reads a caller-supplied `stage4.evaluation-reference.v1` JSON file. Its input fields are:

- `input_manifest_sha256`: binds the reference to the exact observable evidence bundle.
- `split`: descriptive split/exposure label; confirmation/test/holdout requires the explicit CLI flag.
- `pieces`: physical ID -> `native_to_reference` 3x3 transform.
- `contacts`: canonical physical pairs with a fixed `points_b` sample and optional saved corresponding `points_a` anchors. Without explicit A anchors, the reference relative transform maps B probes into A.

These references are never solver inputs. The evaluator examines only the primary committed layout, not the best alternative chosen using truth. It preserves the 4-native-pixel p95 contact criterion on the provided anchors and separately computes the proposed strict whole-page criterion on fixed observed whole-piece probes. Legacy Stage 3 reference maps need an explicit, verified adapter; generic probes do not establish parity with every historical saved-reference evaluator.

Unresolved contacts and inter-component relative placements remain failures in the full denominator. A spanning forest need not explicitly select every expected adjacency. Evaluation reports implied-contact performance, selected-join precision, false neighbor joins, unresolved pairs and a contact-based largest correct selected component. The maximum whole-layout error is null when inter-component poses are unresolved. These are debugging metrics, not calibrated confidence or a capacity claim.
