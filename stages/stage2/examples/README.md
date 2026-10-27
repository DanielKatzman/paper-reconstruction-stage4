# Examples

`stage1_output/` is a tiny, self-contained example of the frozen Stage 1 disk
contract consumed by `run_stage2_inference.py`. It contains two accepted pieces
in deliberately non-alphabetical manifest order.

Recreate it deterministically enough for interface testing with:

```powershell
python .\scripts\create_stage1_handoff_example.py --force
```

The example images are interface fixtures, not model-quality samples and not
part of the training or evaluation datasets.

