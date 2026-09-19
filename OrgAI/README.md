# OrgAI

This folder contains the organoid-specific pseudo-label unmixing heads:

- `overlap_judge.py`: `PLU_Overlap_Judge` returns one reliability logit per ROI.
- `decomposition_mask.py`: `PLU_Decomposition_Branch` returns candidate mask logits and instance-existence logits.

The original interfaces are preserved. Inputs are ROI features shaped `(N, C, H, W)`.
The decomposition branch uses `K=5` candidate instances by default. The trainer imports both classes from the root-level `OrgAI` package.
