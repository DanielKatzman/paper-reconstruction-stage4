# Validated construction baseline

The original V21r4 run passes all fourteen development pages (228 pieces). Its source fingerprint, preserved by tag v2.0.0, is `10270c9ef986d15529e05fd2bdc7b8d39675ba598d59b1876be815445084761e`. Later releases have their own source fingerprints and validation receipts; [v2.0.4](V204_CAPTURE_ALIGNMENT_REVIEW.md) passes the same fourteen controls and adds the cold six-piece real-capture result. The unchanged Stage 3 candidate is `6b0ac07074ff3ecf340aa7c30deae6760d8a8cd4807e3dd9b71348d72e86e66a`.

The original request is `5a657b03d83db9a47eddb4954b0fc972ebd1f70ab0408a776a726a7e8587632f`; the prediction barrier is `78f23b8cd2b36f6be95a6df64e8b8985eef7f2f068e70a3f36d98e6b40fca68d`. These authenticate the original run, not a newly relocated execution. Component byte inventories and optional asset hashes are supplied separately.

| Pieces | Puzzle suffix | Stage 4 worker seconds | Whole-page p95 pixels |
| --- | --- | ---: | ---: |
| 4 | 330 | 6.80 | 5.10 |
| 4 | 450 | 6.80 | 9.22 |
| 8 | 678 | 18.59 | 4.12 |
| 8 | 917 | 18.18 | 27.31 |
| 12 | 341 | 58.30 | 21.95 |
| 12 | 362 | 30.96 | 5.39 |
| 16 | 681 | 46.46 | 3.16 |
| 16 | 801 | 246.43 | 17.26 |
| 20 | 600 | 162.63 | 30.00 |
| 20 | 654 | 62.73 | 9.06 |
| 24 | 187 | 89.83 | 15.03 |
| 24 | 976 | 167.27 | 18.36 |
| 30 | 062 | 125.03 | 22.83 |
| 30 | 847 | 105.77 | 11.66 |

Independent replay checked 392 selected native joins and 2,242 physical piece pairs. All fourteen fine geometry checks passed. Every preview was manually reviewed. Small overlaps are permitted by the configured bounded interior-sampling physical verifier; this does not claim zero pixel overlap or exact scan reconstruction. Full measurements are in `baseline/v21r4/review.json` and the original review in `docs/history`.

All Stage 4 cases are below ten minutes; the slowest takes 4m06s. Total worker time is 19m06s. The previously unresolved 20-piece case 600 now succeeds. Across the thirteen previously successful controls alone, runtime increased 26.9%; 801 and 976 remain explicit Stage 4 performance targets. Aggregate speedup against a prior failed run is not a general speedup claim.

Stage 3 native inference plus pooling still takes 52.2 and 45.5 minutes on the two 30-piece pages in the reused preparation measurements. The complete pipeline has not achieved a ten-minute target. Stage 1 segmentation and arbitrary captures have not been qualified by this regression cohort. Puzzle 976 is mostly blank with a header/border; further content-rich holdouts are needed.

The source export retains the original review as evidence. New portable smoke/integration results are recorded separately in `provenance/PACKAGING_VALIDATION.json`; they do not replace the original receipts or count as a fresh fourteen-case upstream run.
