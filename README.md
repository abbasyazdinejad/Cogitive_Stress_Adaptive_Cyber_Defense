# Cogitive_Stress_Adaptive_Cyber_Defense

## Reproducibility & Scope (Important)

This repository provides research code and simulation scripts supporting the paper
“Towards Stress-Adaptive Cyber Defense: Cognitive–Physiological Synchronization in IoT Environments.”

### What the results represent
- Physiological stress inference is evaluated using leave-one-subject-out (LOSO) cross-validation on the WESAD dataset.
- SOC performance results are produced via controlled, programmatic SOC simulations (including CICIDS2017-driven incident timelines) to study decision-policy behavior under stress, not via deployment in a live Security Operations Center.

### Why your numbers may differ
Exact numerical replication is not guaranteed across environments because results can vary due to:
- stochastic training dynamics (random initialization, batching, GPU/CPU nondeterminism),
- preprocessing / signal-windowing differences,
- library and hardware differences,
- simulation randomness (alert/task generation, injected stress timelines),
- calibration and threshold selection on subject-disjoint splits.

We therefore emphasize *comparative trends* and *behavioral effects* of the proposed CPS/UATR/SWM components rather than identical point estimates in every run.

### Recommended practice
For best comparability, use the provided scripts, fixed seeds (when available), and the documented hyperparameter ranges and evaluation protocol.
