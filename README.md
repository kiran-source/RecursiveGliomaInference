# Bayesian Inference for Mechanically-Coupled Hyper-elastic Tumor Growth
Recursive Bayesian inference and model selection for mechanically coupled tumor growth models using longitudinal MRI data.

## Project Overview

This project develops and calibrates a **mechanically coupled reaction–diffusion (Mech-RD)** tumor growth model using **Bayesian inference**. The framework integrates tumor biology with tissue mechanics to better capture realistic tumor evolution.

MRI tumor data from different rats are used to infer key parameters:

1. **E** (Elastic Modulus)
2. **D** (Diffusivity)
3. **G** (Proliferation Rate)

Model performance is evaluated using:

* **DICE coefficient** (shape overlap)
* **Normalized Tumor Area (NTA)**

---


## Code Dependencies

To execute these scripts, your environment must include the following dependencies:
* **FEniCS** A popular open-source computing platform designed for solving partial differential equations (PDEs) using the finite element method (FEM). Its standout feature is allowing researchers to write mathematical models in a high-level language that closely resembles standard mathematical notation, which the software then automatically translates into highly optimized, efficient code to simulate physical systems.
* **hIPPYlib**: An advanced mathematical framework built directly on top of FEniCS. hIPPYlib is dedicated to working backward (inverse problems). It provides the  optimization and statistical tools required to infer unknown parameters from observational data, and quantify the uncertainty of those estimates. It computes gradient and Hessian in an automatic manner, which can be utilized to solve PDE-constrained Bayesian inverse problems.

## How to Run
To run the code, execute the main script from your command line. The script accepts optional arguments to configure the noise variance, tissue density, and the number of posterior evaluation samples.
```bash
python Rat-3_hyper.py --sigma 0.035 --rhoGM 800 --nsamples 500
```

## 📁 Repository Structure

```
RecursiveGliomaInference/
│
├── BayesianInference/        → Core modeling, and Inference
├── W05_data/2D/                 → Rat data and mesh
```

---

## 🔍 Folder Descriptions

### `BayesianInference`

Contains the complete computational framework:

* Forward models (Mech-RD basically the Hyperelastic setup)
* Bayesian inference (MAP, LA)

### Data

The folder `W05_data/2D/` contains a murine glioma MRI dataset used for Bayesian inference. The data include MRI-derived tumor volume fraction fields at multiple imaging time points and the finite-element mesh used for the reference-domain computation

📌 Detailed instructions and module descriptions are provided in:

```
BayesianInference/README.md
W05_data/2D/READNE.md
```