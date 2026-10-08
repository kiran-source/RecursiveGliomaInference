# Bayesian Inverse Biophysical Modeling for Glioblastoma Growth

[![FEniCS](https://img.shields.io/badge/FEniCS-2019.1.0-blue.svg)](https://fenicsproject.org/)
[![hIPPYlib](https://img.shields.io/badge/hIPPYlib-3.0.0-green.svg)](https://hippylib.github.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://hub.docker.com/r/hippylib/fenics)
[![Dataset](https://img.shields.io/badge/Cohort-10%20LUMIERE%20Patients%20%2B%20Rat%20Baseline-orange.svg)](#datasets)

A PDE-constrained Bayesian inverse framework for simulating, calibrating, and forecasting patient-specific **Glioblastoma Multiforme (GBM)** progression and tissue deformation using longitudinal MRI scans.

This repository couples:
1. **Reaction-Diffusion Dynamics**: Heterogeneous tumor motility ($D$) and logistic proliferation ($G$).
2. **Nonlinear Hyperelastic Mechanics**: Brain tissue deformation governed by Neo-Hookean strain energy ($E$) under tumor mass-effect swelling.
3. **Bayesian Inversion & Uncertainty Quantification**: Adjoint-based Inexact Newton-CG optimization, Vector Bi-Laplacian spatial priors, and Laplace approximation via randomized SVD.

---

## 🔬 Core Features

* **Biomechanical First Principles**: Accounts for tumor mass effect, compressive mechanical stress, and midline shift.
* **One-Scan-Ahead Forecasting**: Calibrated on longitudinal scans ($t=2, 4, 5, 6$) to predict future, unseen scan morphology at $t=9$ (**Forecast DICE = 0.9102**).
* **Multi-Subject Cohort**:
  * **Preclinical Rat Baseline**: `W05` rat glioma dataset.
  * **Human Clinical Cohort**: 10 patients (`Patient_001` through `Patient_010`) constructed from the **LUMIERE** longitudinal MRI dataset.
* **Complete Reproducibility**: Automated Docker runners (`run.ps1` / `run_cohort.ps1`) require zero local FEniCS compilation.

---

## 📊 Summary of Results (Patient 1)

| Metric | Result | Physical / Clinical Meaning |
| :--- | :--- | :--- |
| **Forecast DICE ($t=9$)** | **0.9102** | **91% volumetric overlap** on unseen future scan. |
| **Model NTA vs. True NTA ($t=9$)** | **0.2439** vs. **0.2454** | Error $< 0.6\%$; captures accurate mass expansion. |
| **Newton-CG Convergence** | **9 iterations** | Objective misfit dropped from 2,717 to 780. |
| **Effective Parameter Dimension** | **14.253** | Dominant directions informed by patient imaging. |
| **Bayesian Log-Evidence** | **$-1.393 \times 10^7$** | Metric for statistical model comparison. |

---

## 📁 Repository Structure

```
.
├── BayesianInference/
│   ├── patient_hyper.py           # Master solver for multi-patient cohort
│   ├── Rat-3_hyper.py             # Preclinical rat baseline solver
│   ├── mech_hyper.py              # Neo-Hookean hyperelastic constitutive laws
│   ├── td_mechHyp.py              # Coupled PDE forward & adjoint engine
│   └── results/
│       └── Patient_001/           # Verified Patient 1 outputs (DICE, NTA, XDMF)
├── mri_data/                      # 10-Patient Human Cohort (Patient_001 - 010)
│   └── Patient_XXX/2D/            # Axial meshes (.xml) and observations (.h5)
├── W05_data/2D/                   # Rat baseline data & mesh
├── hippylib/                      # hIPPYlib library with upstream prior bugfix
├── build_lumiere_cohort.py        # Cohort builder from LUMIERE metadata
├── convert_state_to_paraview.py   # XDMF adapter for ParaView 6+
├── plot_metrics.py                # Publication-quality metrics plotter
├── run.ps1 / run.sh               # Baseline rat runner
└── run_cohort.ps1 / run_cohort.sh # Clinical cohort runner
```

---

## 🚀 Quickstart

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### 1. Run the Human Patient Cohort (Windows PowerShell)
Run an individual patient:
```powershell
.\run_cohort.ps1 -patient Patient_001 -max_iter 10 -nsamples 100
```
Or run all patients sequentially:
```powershell
.\run_cohort.ps1 -all -max_iter 7 -nsamples 100
```

### 2. Run the Original Rat Baseline
```powershell
.\run.ps1
```

---

## 📈 Visualizing Outputs

### Option A: Instant PNG Summary
View generated metrics without external tools:
* Open `BayesianInference/results/Patient_001/metrics_summary.png` to inspect the DICE curve, NTA comparison, and volume growth.

### Option B: 2D Interactive Spatiotemporal Fields in ParaView
1. Open **[ParaView](https://www.paraview.org/)**.
2. Open `BayesianInference/state_trajectory_paraview.xdmf` (or `results/Patient_001/state_trajectory_paraview.xdmf`).
3. Select **XDMF Reader** $\rightarrow$ click **Apply**.
4. Change coloring dropdown from `Solid Color` to **`u`** (tumor concentration).
5. Click **Play ($\blacktriangleright$)** to animate tumor invasion and brain deformation up to Day 9 forecast!
6. Open `MAP.xdmf` to view the estimated spatial maps of tissue stiffness $E$, diffusion $D$, and proliferation $G$.

---

## 📚 Citation & Acknowledgments
Based on:
* *Recursive Bayesian inference and model selection for mechanically coupled tumor growth models* (PCELab / hIPPYlib).
* Clinical metadata derived from the **LUMIERE** dataset (*Longitudinal Glioblastoma MRI Dataset with Expert Segmentations*).