# Quickstart & Project Guide: Recursive Glioma Inference & LUMIERE Cohort

This workspace contains the complete implementation of **Recursive Bayesian Inference and Model Selection for Mechanically Coupled Hyperelastic Tumor Growth** ([PCELab/RecursiveGliomaInference](https://github.com/PCELab/RecursiveGliomaInference)), extended to support patient cohorts from the **LUMIERE Glioblastoma Dataset**.

---

## 📁 Workspace Architecture

```text
PR/Paper/
├── BayesianInference/
│   ├── Rat-3_hyper.py       # Paper baseline script (Rat-3 / W05)
│   ├── patient_hyper.py     # Generalized patient runner (100% paper logic preserved)
│   ├── mech_hyper.py        # Hyperelastic strain energy & stress formulation
│   ├── td_mechHyp.py        # Time-dependent forward PDE & Adjoint solver
│   ├── utils.py             # Coordinate interpolation & helpers
│   └── test_env.py          # Verification script
├── mri_data/
│   ├── W05_data/2D/         # Original paper rat dataset (W05.h5, W05_0.xml)
│   ├── Patient_001/2D/      # LUMIERE Patient 1 (Patient_001.h5, Patient_001_0.xml)
│   ├── Patient_002/2D/      # LUMIERE Patient 2
│   ├── ...                  # ...
│   └── Patient_010/2D/      # LUMIERE Patient 10
├── hippylib/                # hIPPYlib library (PDE-constrained inverse problems)
├── preprocess_lumiere.py    # Converter for raw LUMIERE NIfTI segmentations -> FEniCS
├── build_lumiere_cohort.py  # Builder for 10-patient clinical cohort
├── LUMIERE-datacompleteness.csv # Official LUMIERE scan completeness table
├── LUMIERE-ExpertRating.csv # Official LUMIERE expert clinical RANO ratings
├── run.ps1                  # 1-click runner for paper baseline (W05)
├── run_cohort.ps1           # 1-click runner for LUMIERE patient cohort
└── run_cohort.sh            # Bash runner for Linux/WSL/Mac
```

---

## 🚀 How to Run

### 1. Run the Paper Baseline (Rat-3 / W05)
```powershell
# Fast test run (2 iterations to test forward/adjoint solvers)
.\run.ps1 -max_iter 2 -nsamples 10

# Full paper run
.\run.ps1 -sigma 0.035 -rhoGM 800 -nsamples 500
```

---

### 2. Run Any LUMIERE Patient (Patient_001 to Patient_010)
```powershell
# Run Patient 1 with 20 iterations
.\run_cohort.ps1 -patient Patient_001 -max_iter 20 -nsamples 100

# Run Patient 2
.\run_cohort.ps1 -patient Patient_002 -max_iter 20 -nsamples 100

# Fast verification test (2 iterations)
.\run_cohort.ps1 -patient Patient_001 -max_iter 2 -nsamples 10
```

---

### 3. Run the Entire 10-Patient Cohort in Batch
```powershell
.\run_cohort.ps1 -all -max_iter 20 -nsamples 100
```

---

## 📊 Outputs & Results

Results are saved to isolated folders under `BayesianInference/results/<Patient_ID>/`:
* **`DICE_scores.txt`**: Overlap scores between model prediction and ground truth scans (including the held-out forecast scan).
* **`nta.txt`**: Normalized Tumor Area comparison (`NTA_model` vs `NTA_mri`).
* **`evidence.txt`**: Laplace log-evidence $\ln p(\mathbf{y} \mid M)$ and effective dimension for Bayesian model selection.
* **`MAP.xdmf`**: Continuous spatial maps of inferred parameter fields: Elasticity ($E$), Diffusivity ($D$), and Proliferation ($G$).
* **`state_trajectory.xdmf`**: 2D movie of predicted tumor progression (open in **ParaView**).
* **`TumorData.xdmf`**: Ground truth MRI observations mapped on the mesh.
