# Frozen model artifacts

Production uses `stage2_boundary_v1.0.0.onnx`, a 25.7 MB CPU model tracked with
Git LFS. `release.json` freezes its SHA-256, tensor contract, preprocessing,
calibration, and fragment-proposal settings. The public inference CLI verifies
the file before opening it and selects only ONNX Runtime's CPU provider.

Verify it on Windows PowerShell:

```powershell
(Get-FileHash .\models\stage2_boundary_v1.0.0.onnx -Algorithm SHA256).Hash.ToLower()
```

The result must match `stage2_boundary_v1.0.0.onnx.sha256` and
`production_model.sha256` in `release.json`.

`stage2_boundary_v1.0.0.pt.sha256` records the 76 MB training checkpoint from
which the ONNX graph was exported. The PyTorch checkpoint is not needed for
production and is kept outside Git history; maintainers need it only to reproduce
the export or run PyTorch/ONNX parity validation.

`onnx_end_to_end_parity.json` records the checked full-contour comparison on the
bundled Stage 1 fixture. It complements the random multi-batch graph parity
results embedded in `release.json`.

