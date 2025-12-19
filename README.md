# Cognitive Stress–Adaptive Cyber Defense

This repository contains the research code and demonstration for:

**Towards Stress-Adaptive Cyber Defense: Cognitive–Physiological Synchronization in IoT Environments**

The project investigates how physiological stress inference and cognitive decision modeling can be integrated into cyber-physical and Security Operations Center (SOC) workflows to enable **stress-aware, adaptive cyber defense strategies**.

---

## Project Overview

Modern cyber-physical and IoT environments increasingly rely on human analysts operating under high cognitive load. This project explores a **stress-adaptive cyber defense framework** that:

- infers operator stress from multimodal physiological signals,
- models cognitive decision-making under stress,
- adapts SOC response strategies accordingly,
- evaluates system behavior under controlled simulation conditions.

The framework integrates:
- physiological signal processing,
- machine-learning–based stress classification,
- cognitive agents (e.g., SpeedyIBL-inspired reasoning),
- simulated SOC environments driven by synthetic and benchmark attack traces.

---

## Repository Structure
stress_adaptive_defense/
│
├── src/ # Core framework implementation
│ ├── data/ # Signal processing and feature extraction
│ ├── models/ # Stress classifier, CPS layer, cognitive agents
│ ├── soc/ # Simulated SOC environments
│ ├── evaluation/ # Metrics and visualization utilities
│ └── utils/ # Helper functions
│
├── experiments/ # Experiment scripts (e.g., LOSO evaluation)
├── Sample_Demo/ # Demonstration notebooks
├── requirements.txt # Core dependencies
├── requirements-optional.txt
└── README.md


---

## Reproducibility & Scope

### Nature of the Evaluation
All results produced by this codebase are obtained through **controlled experiments and simulations**.
- Physiological stress inference is evaluated using **leave-one-subject-out (LOSO) cross-validation** on the WESAD dataset.
- SOC performance is evaluated using **simulated SOC environments**, including:
  - synthetic alert streams,
  - abstractions derived from benchmark intrusion datasets (e.g., CICIDS2017).

**This repository does not represent deployment in a live SOC, production IoT system, or operational cyber-physical infrastructure.**

---

### Why Your Results May Differ
Exact numerical reproduction is **not guaranteed** due to:
- stochastic model initialization and training dynamics,
- signal preprocessing and windowing variability,
- hardware and software version differences,
- simulation randomness (alert generation, task scheduling),
- calibration and threshold selection on subject-disjoint splits.

Accordingly, this work emphasizes **comparative trends and behavioral effects** rather than identical point estimates across runs.
---

### Recommended Practice
For best comparability:
- use the provided experiment scripts,
- follow the documented evaluation protocol,
- apply fixed random seeds where available,
- interpret results at the level of **relative performance and adaptive behavior**.

---
## Demonstration Notebooks
The `Sample_Demo/` directory contains Jupyter notebooks illustrating:
- stress-aware decision pipelines,
- cognitive agent behavior under varying stress states,
- SOC simulation workflows,
- example result visualizations.
These notebooks are intended for **demonstration and conceptual understanding** and are not designed to exactly reproduce all numerical results reported in the paper.
---

## Dataset Availability

This repository does **not** redistribute large or licensed datasets.

- **WESAD**: available from the original dataset authors.
- **CICIDS2017**: available from the Canadian Institute for Cybersecurity  
  https://www.unb.ca/cic/datasets/ids-2017.html

Users should obtain datasets directly from official sources and preprocess them according to the methodology described in the paper.

---

## Ethical and Research Use Disclaimer

This code is provided **for research and educational purposes only**.

No claims are made regarding real-world SOC deployment, operational readiness, or production use. Any application beyond simulation and research contexts requires additional validation, governance review, and human-in-the-loop safeguards.

---



