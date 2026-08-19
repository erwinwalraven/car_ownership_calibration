# Car ownership calibration example

This folder contains a minimal reference implementation of the subgroup calibration approach used in the paper below.

The file [calibration_example.py](calibration_example.py) provides one function:
- `apply_group_calibration(...)`

The function focuses only on the calibration logic:
- Learn class-wise multiplicative subgroup calibrators from a calibration-train set.
- Keep subgroup calibrators only when they improve KL divergence on a calibration-holdout set.
- Apply accepted calibrators to target probabilities and renormalize.

This example is intentionally minimal and is meant to explain the method, not to provide a full experimental pipeline.

## Reference

Walraven, E., Rogier, J., & Snelder, M. (Data Science for Transportation).
*Calibration of car ownership models for population group analysis*.
