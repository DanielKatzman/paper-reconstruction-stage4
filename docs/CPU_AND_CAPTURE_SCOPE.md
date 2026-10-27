# CPU portability and capture qualification

The portable baseline must run without a GPU. The v2.0.1 umbrella runner and Stage 3 worker default to CPU; setup installs a CPU-only Torch wheel. Stages 1/2 use CPU ONNX inference, Stage 3's frozen rankers and geometric work run on CPU, and Stage 4 assembly uses CPU numerical libraries. CUDA remains an explicit optional acceleration choice.

The immutable v2.0.0 release already implements Stage 3's `--device cpu` path, but its setup and runner default to CUDA. Its preparation measurements used CUDA for ranker scoring. A successful Stage 4 CPU benchmark is not proof that upstream inference was CPU-only or that the entire pipeline meets a ten-minute goal.

`tools/check_cpu_runtime.py` loads the real frozen rankers, requires a CPU-only Torch build and verifies their tensors are on CPU. CI exercises that check on Windows and Linux. This is a runtime/model portability check; it is not a thirty-piece full-pipeline performance benchmark.

The CPU review also found boundary mismatches. Stage 1 diagnoses eight-connected masks, while Stage 2 requires one four-connected component. On four re-segmented generator captures, two accepted masks contained diagonal fringe islands of 5+4 and 7 pixels. Stage 1 also exports a tight crop, while Stage 3 requires pixel-cell boundary contours to fit within the native image frame.

Version 2.0.1 preserves raw `stage1_native` output and publishes a separate Stage 2 input with an explicit bounded adapter/audit. It removes at most 64 pixels and 0.01% of a mask; a meaningful or genuinely disconnected fragment requires review. It adds up to four pixels of zero-mask border within the original capture, preserving RGB content on the piece, alpha masks, full-image crop coordinates and native pixel scale. It requires review when a piece touches the original capture frame. Generator-provided Stage 1 benchmark inputs bypass the adapter. These are explicit contract normalization steps, not perfect-mask declarations or scale/perspective correction.

## Real photographed pieces

The 14/14 construction result comes from exact generator masks and declared common native pixel scale. It bypassed Stage 1 inference and does not establish assembly reliability with predicted masks from real captures. Version 2.0.1's capture route used an older configuration; v2.0.2 uses V21 with explicit predicted-mask provenance and unchanged geometry/support gates.

The user's intended input is same-DPI scans or fixed-distance photographs with flat pieces. “1:1” means common relative physical/native pixel scale, not generator masks or perfect alignment. V21 accepts both mask provenances under the same rigid/common-scale model. Keep camera orientation/settings consistent and avoid independent image resizing. Native rotations may be continuous; 1:1 scale does not imply 90-degree turns.

Version 2.0.3 fixes the remaining continuous-angle restrictions inside graph expansion, cut repair and support-cycle pooling. The capture path lets ordinary routes run before persistent native recovery, retains only independently verified partial results, and can revisit earlier attachments/use targeted archived angles. Same-scale acquisition does not itself guarantee that a complete construction will be found. The capture review records actual outcomes separately from execution and synthetic controls.

Do not mark real images as `generator_exact_bypass` to get past the development-only page guards. Use the truthful capture route, inspect masks/acceptance flags and assess the resulting construction. If perspective or scale differences are significant, add measured normalization or qualify an appropriate transform model before assuming rigid matching applies.

The next real-input acceptance check is an end-to-end CPU run on the actual photographed piece set, including segmentation review, observable seam/support checks, geometry assessment and visual construction review. A small synthetic CPU smoke test establishes execution, not that real photographs or all thirty-piece pages are solved reliably.
