# Paper Reconstruction Stage 2

This repository is the complete Stage 2 boundary-understanding component of the
torn-paper reconstruction pipeline. It does not replace Stage 1 segmentation.
In production it consumes the canonical Stage 1 `segmentation_manifest.jsonl`,
validates each accepted piece and mask, predicts a probability distribution over
every ordered contour point, and emits ranked edge-fragment hypotheses for Stage 3.

Stage 2 deliberately does **not** choose one final edge or match pieces. The full
per-point probabilities and alternative fragments remain available so the future
matcher and global solver can revise local guesses.

The data pipeline produces three linked datasets:

1. a **canonical/oracle dataset** containing exact page partitions, piece masks, seam ownership, and dense shared seam geometry;
2. a **direct observation dataset** containing independently transformed and degraded Stage 1-like piece inputs, with exact homographies back to canonical truth;
3. a **boundary-supervision dataset** containing ordered observed contours, local geometry, soft boundary roles, and canonical seam-correspondence tracks.

The separation is intentional. Ground truth remains clean and reusable while
observation noise can be regenerated or intensified. The production adapter is
separate and truth-free: it compiles the same observable representation directly
from frozen Stage 1 outputs.

## What the v1.0 release implements

- a frozen, documented Stage 1 → Stage 2 production contract;
- accepted-only Stage 1 import with provenance, artifact, topology, and path validation;
- truth-free contour compilation using the exact trained geometry/window recipe;
- a frozen ONNX model with SHA-256 enforcement and source-checkpoint lineage;
- portable CPU-only production inference with no PyTorch or CUDA dependency;
- atomic per-piece inference, safe resume, checksummed artifacts, and a process lock;
- per-point role/transition probabilities in crop and source-capture coordinates;
- ranked, non-final fragment hypotheses with stable IDs and endpoint coordinates;
- a standalone verifier for every completed Stage 2 handoff;
- model card, holdout report, schemas, CI, integration fixture, and release checklist;
- deterministic seeded generation from 2 through 30+ total pieces;
- one-cut/two-piece examples as the minimum case;
- simple, moderate, hard, and extreme difficulty tiers with exact dataset schedules;
- straight, smooth-wave, sharp-zigzag, deep-notch, torn, and mixed-scissor cuts;
- source-file-level train, validation, calibration, and holdout splits;
- all pages from one PDF kept in one split, preventing page-family leakage;
- source SHA-256 records in `source_split_manifest.json`;
- per-piece compactness, fill, perimeter, thickness, aspect-ratio, and extreme-shape metrics;
- per-seam chord, tortuosity, and turning metrics;
- independently rotated, scaled, perspective-warped, illuminated, blurred, and JPEG-degraded observations;
- controlled mask erosion, dilation, smoothing, and local boundary damage;
- connected-mask validation and exact forward/inverse homographies;
- deterministic clockwise contour extraction from every observed mask;
- uniformly resampled boundary points with tangent, inward normal, curvature, corner strength, arc length, and inward thickness;
- soft seam/page-border/uncertain supervision after projection and distance-aware snapping;
- seam tracks indexed by canonical seam position, piece owner, view, and traversal direction;
- compact per-puzzle NPZ arrays instead of millions of tiny edge files;
- boundary-role and matching-seam visual previews;
- safe atomic puzzle-level `--resume` for every generation stage;
- one public pipeline command with status, verification, stage stopping, and a
  cross-platform single-process lock;
- a bounded-memory boundary-window loader balanced across seam, page-border,
  uncertain, and explicit role-transition centers;
- aligned RGB/mask raster strips, contour geometry sequences, and soft targets;
- an optional deterministic PyTorch iterable adapter;
- a leakage-safe 7-channel observable geometry input (truth-only distances and
  confidences are never model inputs);
- a compact dual-stream geometry plus RGB/mask contour model;
- soft seam/page-border/uncertain classification plus endpoint-transition learning;
- exact batch-level training resume, safe Ctrl+C, best/last checkpoints, early
  stopping, and natural-distribution validation;
- deterministic multi-worker raster-strip prefetch whose output is unchanged by
  worker count, scheduling, or prefetched-but-unconsumed batches;
- eight controlled contour-model candidates, including an approximately
  order-of-magnitude larger hybrid and a whole-piece/global-contour model;
- whole-piece mask and 256-point full-contour context with local-to-global
  cross-attention for page-border-versus-straight-seam ambiguity;
- an OOM-safe GPU stress matrix that profiles both local and global data paths;
- resumable successive halving with fragment-aware model selection (3 epochs for
  all models, 10 for the top two, and 20 for the winner);
- full closed-contour inference with overlapping windows and probability averaging;
- calibration-split temperature scaling and fragment-threshold selection;
- circular hysteresis fragment proposals with transition-aware splitting and top-K
  ranking instead of permanent hard edge declarations;
- hierarchical fragment hypotheses that retain raw and endpoint-adjusted parents
  alongside standard and relaxed transition-split children;
- endpoint adjustment guards that prevent circular near-full seams from collapsing
  below the configured minimum fragment length;
- full-contour metrics, point and fragment difficulty breakdowns, fragment
  recall@K, and a protected holdout evaluation path;
- a complete validation audit matrix at K=1/3/5/10 and IoU=0.50/0.80,
  candidate-count percentiles, and explicit rank-6-through-10 recovery counts;
- automatic failure manifests and green-truth/magenta-prediction overlays for
  every validation seam below top-5 IoU 0.80;
- cyan overlays for failures whose better candidate is found only at ranks 6-10;
- resumable prediction exports containing probabilities, candidate fragments, and
  visual contour overlays;
- non-overwriting output directories, fingerprints, JSON Schemas, previews, and tests.

## Piece-count semantics

`max_pieces` is the maximum **total pieces produced from one page**, not the maximum number of cuts.

For every page:

```text
target_piece_count ~ UniformInteger(min_pieces, max_pieces)
successful_cut_count = target_piece_count - 1
```

The default minimum is 2 pieces, which means exactly 1 successful cut. Both values are recorded in every puzzle manifest.

## Installation

### Windows PowerShell

Run from the repository root:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-inference.txt
python -m pip install -e .
```

`PyMuPDF` supplies the `fitz` module used for PDF input.

Production uses ONNX Runtime's CPU provider. It does not install or use PyTorch,
CUDA, or an NVIDIA driver. Confirm the runtime after installation:

```powershell
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

Only maintainers training or exporting models need PyTorch. Use
`requirements-training.txt` for CUDA training and `requirements-export.txt` for
checkpoint-to-ONNX export and backend parity checks.

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-inference.txt
python -m pip install -e .
```

## Production: run Stage 1 and Stage 2 together

The frozen `models/stage2_boundary_v1.0.0.onnx` model is part of the release and
is tracked with Git LFS. `models/release.json` freezes preprocessing,
calibration, fragment settings, tensor names, source-checkpoint lineage, and the
ONNX SHA-256. The public command accepts no GPU/device option and always uses
`CPUExecutionProvider`.

```powershell
python .\scripts\run_stage2_inference.py `
  --input "C:\path\to\stage1_output" `
  --output ".\generated\stage2_edge_hypotheses" `
  --resume

python .\scripts\verify_stage2_output.py `
  --input ".\generated\stage2_edge_hypotheses" `
  --release ".\models\release.json"
```

Stage 1 remains immutable, rejected pieces are skipped by default, and the Stage
2 output is the only handoff Stage 3 will need. See
[`docs/STAGE1_INTEGRATION.md`](docs/STAGE1_INTEGRATION.md) for the exact input,
output, coordinate, integrity, and resume contracts.

## Research/training: one public resumable data command

The stage-specific modules remain separate, but normal dataset generation now uses
one public script:

```powershell
python .\scripts\run_stage2_pipeline.py `
  --sources "C:\Users\danka\Desktop\source_pages" `
  --output ".\generated\stage2_dataset" `
  --count 1000 `
  --min-pieces 2 `
  --max-pieces 30 `
  --views-per-piece 2 `
  --resume
```

Its layout is:

```text
stage2_dataset/
  pipeline_state.json
  pipeline.lock
  canonical/
  observations/
  boundaries/
```

Every completed puzzle is an atomic checkpoint in every stage. Pressing `Ctrl+C`
records an interrupted state. Re-running the identical command with `--resume`
reuses every committed puzzle and repeats only the puzzle that was being processed.
The same recovery applies after a process crash or power loss.

Useful controls:

```powershell
# Read checkpoint counts without changing the dataset.
python .\scripts\run_stage2_pipeline.py --output ".\generated\stage2_dataset" --count 1000 --status

# Verify all fingerprints and puzzle manifests after completion.
python .\scripts\run_stage2_pipeline.py --output ".\generated\stage2_dataset" --count 1000 --verify-only

# Intentionally stop after one stage; resume later with the full command.
python .\scripts\run_stage2_pipeline.py ... --stop-after observations --resume
```

The script also accepts `--canonical-dir`, `--observations-dir`, and
`--boundaries-dir`. These let it validate and adopt already completed v0.2/v0.3
datasets without copying or regenerating them, provided the supplied count and
stage settings match their manifests.

## Advanced/debug: individual stages

The three original commands remain available for targeted experiments and partial
reruns.

### Step 1: generate canonical puzzles

Windows PowerShell:

```powershell
python .\scripts\generate_stage2_dataset.py `
  --sources "C:\Users\danka\Desktop\source_pages" `
  --output ".\generated\pilot_canonical" `
  --count 1000 `
  --min-pieces 2 `
  --max-pieces 30
```

Linux or macOS:

```bash
python scripts/generate_stage2_dataset.py \
  --sources /path/to/source_pages \
  --output generated/pilot_canonical \
  --count 1000 \
  --min-pieces 2 \
  --max-pieces 30
```

Omit `--sources` to create deterministic synthetic document pages for testing. An output directory must be absent or empty; existing datasets are never silently overwritten.

The same job can use the supplied configuration:

```powershell
python .\scripts\generate_stage2_dataset.py `
  --config .\configs\pilot_2_to_30.yaml `
  --sources "C:\Users\danka\Desktop\source_pages" `
  --output ".\generated\pilot_canonical"
```

With the default four nonzero splits, a real source directory must contain at least four source files. A PDF counts as one source group regardless of its page count. This constraint prevents the same document from leaking into multiple splits.

### Step 2: compile direct observations

```powershell
python .\scripts\compile_stage2_observations.py `
  --canonical ".\generated\pilot_canonical" `
  --output ".\generated\pilot_observed" `
  --views-per-piece 2 `
  --seed 20261005
```

Each piece is transformed independently. Two views per piece are useful for contrastive or consistency training; one view is faster for initial inspection. Important controls include:

```text
--scale-min / --scale-max
--perspective-jitter
--max-mask-morphology
--local-damage-probability
--max-blur-radius
--jpeg-quality-min / --jpeg-quality-max
```

The observation compiler is a synthetic direct renderer, not a claim that its errors perfectly match Stage 1. A later frozen-Stage-1 replay adapter should be added when a stable Stage 1 checkpoint and inference contract are available.

## Canonical output layout

```text
canonical_dataset/
  dataset_manifest.json
  source_split_manifest.json
  sources/
    <source_id>.png
  puzzles/
    <puzzle_id>/
      puzzle.json
      labels.npy
      truth.npz
      canonical_pieces/
        piece_001.png
        piece_001_mask.png
      previews/
        partition.png
        seams.png
        scattered.png
```

`labels.npy` is the integer canonical label map. `truth.npz` stores compact dense seam and page-border polylines. Every true seam is stored once with two final piece owners instead of materializing all pair combinations.

## Observation output layout

```text
observation_dataset/
  observation_dataset_manifest.json
  puzzles/
    <puzzle_id>/
      observation_manifest.json
      pieces/
        piece_001__view_00.png
        piece_001__view_00_mask.png
      previews/
        observations.png
```

Each observation record stores:

- canonical crop → observed homography;
- observed → canonical crop inverse;
- source page → observed homography;
- observed → source page inverse;
- sampled geometric, photometric, and mask-corruption parameters;
- connected-component, area, and matrix round-trip validation.

The homographies describe the ideal geometric warp. The observed mask may deliberately deviate by a few pixels to simulate segmentation error; this distinction is explicit in the manifest.

### Step 3: compile ordered boundaries and seam supervision

Run this only after both the canonical and observation datasets are complete:

```powershell
python .\scripts\compile_stage2_boundaries.py `
  --canonical ".\generated\pilot_canonical" `
  --observations ".\generated\pilot_observed" `
  --output ".\generated\pilot_boundaries" `
  --resume
```

Existing v0.2 canonical and observation datasets are accepted directly. They do **not** need to be regenerated for v0.3. Future observation generation uses stricter connectivity safeguards, while the boundary compiler marks individual correspondence tracks as usable or unusable so an occasional damaged v0.2 view cannot contaminate matching training. With two views per piece, another valid track can still represent the same seam owner.

The compiler verifies that the observation dataset fingerprint belongs to the supplied canonical dataset. For each observed mask it traces one closed contour, resamples it, calculates local geometry, projects canonical seam and page-border truth through the stored homography, and softly snaps the truth to the possibly perturbed observed boundary.

`--resume` is recommended for the full dataset. Each puzzle is compiled in a temporary directory and atomically committed only after its arrays, previews, manifest, and validation are complete. A restarted run reuses completed puzzles and rebuilds only missing ones.

The most important controls are:

```text
--boundary-spacing             default 2.0 observed pixels
--truth-sigma                  default 3.0 pixels
--max-snap-distance            default 12.0 pixels
--correspondence-spacing       default 3.0 canonical pixels
--minimum-matchable-seam-length default 24.0 canonical pixels
--minimum-track-valid-fraction default 0.60
--minimum-aligned-fraction     default 0.80
```

Do not increase `--max-snap-distance` merely to hide a systematic mapping error. A low aligned fraction should first be investigated in the previews and manifests.

## Boundary output layout

```text
boundary_dataset/
  boundary_dataset_manifest.json
  puzzles/
    <puzzle_id>/
      boundary_manifest.json
      boundaries.npz
      correspondences.npz
      previews/
        boundary_roles.png
        seam_ids.png
```

`boundary_roles.png` uses red for a true torn seam, cyan for an original page border, and yellow for uncertainty or a junction. `seam_ids.png` gives both owners and all views of the same physical seam the same stable color.

`boundaries.npz` contains concatenated arrays and observation offsets. Its main arrays are:

```text
points_xy
arc_length_px
tangent_xy
inward_normal_xy
curvature_per_px
corner_strength
inward_thickness_px
seam_target
page_border_target
uncertainty_target
label_confidence
nearest_truth_distance_px
nearest_seam_index
nearest_border_index
observation_offsets
```

`correspondences.npz` stores canonical seam positions, their projected observed locations, nearest boundary indices, projection distances, validity flags, and track offsets. The accompanying JSON records identify the seam, owner, view, and whether that owner traverses the canonical seam forward or in reverse.

## Inspect the streaming training loader

The model loader does not enumerate or load all boundary points into memory.
It indexes the small per-puzzle JSON manifests, keeps only a bounded number of NPZ
bundles and images cached, and samples circular contour windows by boundary role.

```powershell
python .\scripts\inspect_stage2_training_data.py `
  --boundaries ".\generated\stage2_dataset\boundaries" `
  --observations ".\generated\stage2_dataset\observations" `
  --split train `
  --samples 64 `
  --output ".\generated\stage2_dataset\training_loader_report.json"
```

Each sampled item contains:

- a `sequence_length × 7` observable contour-geometry sequence: tangent, inward
  normal, curvature, corner strength, and log inward thickness;
- a `4 × sequence_length × strip_width` RGB-plus-mask raster strip;
- soft seam, page-border, and uncertainty targets for every sequence position;
- label confidence and puzzle/observation metadata.

`label_confidence` weights the loss but is deliberately not part of the geometry
input. `nearest_truth_distance_px` remains a dataset diagnostic only. Both depend
on synthetic truth and using either as an input would leak the answer to the model.

`TorchBoundaryWindowDataset` is an optional iterable adapter. PyTorch is not pinned
in `requirements.txt`, because Windows CUDA builds should be installed explicitly
for the target GPU and driver. The NumPy loader and its inspection command work with
the standard project requirements.

## Train the boundary-role model

The baseline public command runs four modular stages in order:

1. resumable training-statistics compilation;
2. exact-resume model training;
3. calibration on the reserved calibration split;
4. natural full-contour validation and fragment recall.

Any v0.5/v0.6 run already in progress remains a useful baseline and should be
allowed to finish with its original release. Start v0.6.2 training in a new output
directory: transition-focused sampling changes the exact data stream, so older
training checkpoints are intentionally not resumed. New runs default to two loader
workers. Pascal GPUs do not have Tensor Cores, so AMP is selected only if the
supplied profiler measures a benefit.

```powershell
python .\scripts\run_boundary_role_pipeline.py `
  --boundaries ".\generated\stage2_dataset\boundaries" `
  --observations ".\generated\stage2_dataset\observations" `
  --output ".\generated\boundary_role_v062" `
  --batch-size 16 `
  --workers 2 `
  --precision fp32 `
  --epochs 20 `
  --samples-per-epoch 50000 `
  --validation-samples 10000
```

Press `Ctrl+C` once during training to finish the current batch and atomically save
the exact model, optimizer, scheduler, scaler, sample position, and process RNG
state. Every sample is counter-seeded from epoch plus absolute sample index, so
worker prefetch cannot change or skip the resumed data stream.
Continue with the identical training settings plus `--resume`:

```powershell
python .\scripts\run_boundary_role_pipeline.py `
  --boundaries ".\generated\stage2_dataset\boundaries" `
  --observations ".\generated\stage2_dataset\observations" `
  --output ".\generated\boundary_role_v062" `
  --batch-size 16 `
  --workers 2 `
  --precision fp32 `
  --epochs 20 `
  --samples-per-epoch 50000 `
  --validation-samples 10000 `
  --resume
```

The completed output contains `training/best.pt`, `training/last.pt`,
`calibrated_best.pt`, `calibration_report.json`, and `validation_report.json`.
Calibration and routine validation default to 512 complete observed contours so an
iteration finishes in reasonable time. Pass `--calibration-observations 0` or
`--evaluation-observations 0` to use the entire respective split.

### Required before the model search: hardware and model stress test

The profiler first chooses the fastest loader-worker count using both the base
model and the global-context model. It then tests every contour-model candidate
across batch sizes and FP32/AMP. It measures data wait, GPU compute, end-to-end
windows/second, parameters, and peak VRAM. CUDA OOM is recorded and recovered from
instead of aborting the matrix. Statistics must be compiled first. Existing v0.5
or v0.6 statistics are compatible, even though their model checkpoints are not:

```powershell
python .\scripts\compute_stage2_training_statistics.py `
  --boundaries ".\generated\stage2_dataset\boundaries" `
  --observations ".\generated\stage2_dataset\observations" `
  --output ".\generated\training_statistics.json" `
  --resume

python .\scripts\profile_boundary_role_training.py `
  --boundaries ".\generated\stage2_dataset\boundaries" `
  --observations ".\generated\stage2_dataset\observations" `
  --statistics ".\generated\training_statistics.json" `
  --output ".\generated\training_profile_v062.json" `
  --batch-sizes "8,16,32,64,96,128" `
  --workers "0,2,4" `
  --precisions "fp32,amp"
```

The JSON report includes `recommended_common_configuration`, constrained to at
most 82% of VRAM for every candidate. This is the fair shared recipe used by the
experiment runner; the largest batch is not assumed to be the fastest.

### Recommended model search

After profiling, run the resumable successive-halving experiment. It uses the same
seed, sample schedule, optimizer recipe, and profiled batch configuration for every
candidate. Point metrics alone do not choose the winner. The v0.6.2 promotion score
weights point quality 30%, top-5 recall at IoU 0.50 20%, stricter top-5 recall at
IoU 0.80 35%, and transition F1 15%.

```powershell
python .\scripts\run_boundary_role_experiments.py `
  --boundaries ".\generated\stage2_dataset\boundaries" `
  --observations ".\generated\stage2_dataset\observations" `
  --statistics ".\generated\training_statistics.json" `
  --profile ".\generated\training_profile_v062.json" `
  --output ".\generated\boundary_role_search_v062" `
  --samples-per-epoch 50000 `
  --validation-samples 10000 `
  --resume
```

The eight candidates are `base_tcn`, `wide_tcn`, `large_tcn`,
`hybrid_attention`, `xlarge_hybrid_attention`, `global_context_hybrid`,
`geometry_only`, and `raster_only`. All train to epoch 3, the top two resume to
epoch 10, and the validation winner resumes to epoch 20. Only that final winner is
calibrated. The runner checkpoints every normal training batch interval, every
completed epoch, every full-contour report, and every search stage. Each evaluation
also writes a sibling `*_failures` directory with a JSON manifest and diagnostic
overlays. It never touches holdout and records that invariant in
`experiment_state.json`.

If the process is stopped, repeat the identical command with `--resume`. The
screening and promotion budgets are execution limits rather than changes to the
20-epoch optimization schedule, so promoted models continue from the exact same
optimizer and learning-rate state.

### Refine a completed v0.6.2 winner without retraining

v0.6.3 changes only the probability-to-fragment proposal layer. It reuses the
existing uncalibrated `final_best.pt` weights, recalibrates hierarchical fragment
hypotheses on the reserved calibration split, and evaluates them on validation.
The command is resumable at the completed calibration and validation stages and
never accesses holdout:

```powershell
python .\scripts\refine_boundary_fragments.py `
  --checkpoint ".\generated\boundary_role_search_v062\candidates\global_context_hybrid\final_best.pt" `
  --boundaries ".\generated\stage2_dataset\boundaries" `
  --observations ".\generated\stage2_dataset\observations" `
  --output ".\generated\boundary_role_refinement_v063" `
  --batch-size 8 `
  --device cuda `
  --resume
```

The output contains `calibrated_best_v063.pt`, `calibration_report_v063.json`,
`final_validation_v063.json`, the strict-IoU failure overlays, and a refinement
state file. Use `0` for either observation limit to process the complete split.
Do not run holdout until the v0.6.3 validation report has been reviewed and the
fragment proposer is frozen.

### Audit the frozen v0.6.3 checkpoint before holdout

v0.6.4 is measurement-only: it does not retrain the model, recalibrate the
checkpoint, or change fragment proposals. It evaluates the same frozen checkpoint
with Recall@1/@3/@5/@10 at IoU 0.50 and 0.80, reports candidate-count percentiles
overall and by difficulty, and records whether each top-5 strict-IoU failure is
recovered at ranks 6-10.

```powershell
python .\scripts\evaluate_boundary_role_model.py `
  --checkpoint ".\generated\boundary_role_refinement_v063\calibrated_best_v063.pt" `
  --boundaries ".\generated\stage2_dataset\boundaries" `
  --observations ".\generated\stage2_dataset\observations" `
  --output ".\generated\boundary_role_refinement_v063\final_validation_v064_metrics.json" `
  --split validation `
  --maximum-observations 512 `
  --batch-size 8 `
  --device cuda
```

The sibling `final_validation_v064_metrics_failures` directory contains a v2
failure manifest. Green is truth, magenta is the best top-5 proposal, and cyan is
the better rank-6-through-10 proposal when one exists. Before opening holdout,
confirm that top-10 recall is at least 99% at IoU 0.50, preferably at least 98% at
IoU 0.80, and that the p95/max candidate counts remain practical for the matcher.

### Final holdout evaluation

The normal pipeline never touches holdout. Run this once only after architecture,
losses, thresholds, and calibration are frozen:

```powershell
python .\scripts\evaluate_boundary_role_model.py `
  --checkpoint ".\generated\boundary_role_refinement_v063\calibrated_best_v063.pt" `
  --boundaries ".\generated\stage2_dataset\boundaries" `
  --observations ".\generated\stage2_dataset\observations" `
  --output ".\generated\boundary_role_refinement_v063\holdout_report.json" `
  --split holdout `
  --batch-size 8 `
  --device cuda `
  --allow-holdout
```

### Export predictions from the synthetic research dataset

This produces full per-point probabilities, transition probabilities, ranked
candidate seam fragments, and optional color overlays. Every observation JSON is
the atomic commit marker; `--resume` reuses all fully committed observations even
if a run stopped in the middle of a puzzle.

```powershell
python .\scripts\predict_boundary_roles.py `
  --checkpoint ".\generated\boundary_role_refinement_v063\calibrated_best_v063.pt" `
  --boundaries ".\generated\stage2_dataset\boundaries" `
  --observations ".\generated\stage2_dataset\observations" `
  --output ".\generated\boundary_role_predictions" `
  --split all `
  --resume
```

Use `--no-overlays` for a full production export when PNG previews are no longer
needed. The NPZ probabilities and JSON fragment candidates are unchanged.

## Coordinate convention

- Source coordinates use intact-page pixels.
- `x` increases rightward and `y` increases downward.
- Integer mask indices address pixel centers.
- Canonical seam polylines use source-page grid vertices.
- Owner A traverses a shared seam forward; owner B traverses it in reverse.
- Canonical crop coordinates map to the source page by adding the piece `bbox_xywh` origin.
- Homographies are 3×3 matrices acting on column vectors `[x, y, 1]^T`.

## Splits and difficulty

The defaults are:

| Dimension | Distribution |
|---|---|
| Split | 70% train, 10% validation, 10% calibration, 10% holdout |
| Difficulty | 15% simple, 35% moderate, 35% hard, 15% extreme |

Schedules use largest-remainder allocation, so counts are exact for the requested dataset size and deterministically shuffled. Difficulty affects both the cut-family mix and cut amplitude. The stored geometry metrics let future training code audit or rebalance examples instead of trusting tier names alone.

## Tests

After the editable installation:

```powershell
python -m unittest discover -s tests -v
```

Without installing the package, Linux/macOS can run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite covers 2-piece semantics, a 30-piece stress case, deterministic partitions,
exact split counts, source-level split exclusivity, PDF input, observation
determinism, connected masks, homography round trips, contour orientation, inward
normals, soft role targets, seam tracks, resumability, overwrite protection,
leakage-safe statistics, calibration, fragment proposals, metrics, prediction
rendering, canonical Stage 1 import, truth-free production export, corruption-safe
resume, output verification, ONNX release integrity and feed contracts, and (when
PyTorch is installed) model/loss tensor behavior.

## Next implementation milestone

Stage 2 v1.0 ends at the verified `edge_hypotheses_manifest.json` contract. Stage 3
belongs in a separate repository/component: it should build a lazy
positive/hard-negative fragment-pair dataset from the canonical seam tracks, train
the fragment embedder and compatibility reranker, and feed top-K uncertain matches
into the global solver. It should not materialize a dense all-pairs dataset or
treat Stage 2's first-ranked fragment as final truth.
