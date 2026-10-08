# ----------------------------------------------------------------------
# Mech-RD Tumor Model___Inferring 3 Params (E, D, G)
########## Hyper-elasticity ###########
############## Rat-3 ###################
# ----------------------------------------------------------------------

import time
import math
import numpy as np
import dolfin as dl
from fenics import *
import ufl
import sys
import os
import argparse

sep = "\n"+"#"*80+"\n"
# hippylib setup
for p in ['../hippylib', '../hippylib/hippylib', '../../hippylib', 'hippylib', '/workspace/hippylib']:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)
from hippylib import BiLaplacianPrior
from hippylib import *
sys.path.append('..')
from mech_hyper import MechCoupledTumorVarf
from td_mechHyp import VectorTD, TimeDependentPDEVariationalProblem, MisfitTD
from utils import interp, vector2Function

# Start execution timer
start_time = time.time()

# Set log level
dl.set_log_level(40)
parameters["form_compiler"]["quadrature_degree"] = 4

def computeGammaDelta(corr_len, std, alpha=2, ndim=2):
    nu = alpha - 0.5 * float(ndim)
    assert alpha > 0., "Alpha must be larger than ndim/2"
    kappa = math.sqrt(8. * nu) / corr_len
    gamma = math.sqrt(math.gamma(nu) / math.gamma(alpha)) / (math.pow(4.0 * math.pi, 0.25 * float(ndim)) * math.pow(kappa, nu) * std)
    delta = kappa * kappa * gamma
    return gamma, delta

# ------------------------------
# Problem Configuration Section
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--sigma', default=0.035, type=float)
parser.add_argument('--rhoGM', default=800, type=float)
parser.add_argument('--nsamples', default=500, type=int)
parser.add_argument('--max_iter', default=200, type=int)
args = parser.parse_args()

# Load Mesh
name = "W05_0"
data_dir = None
for candidate in ["../mri_data/W05_data/2D/", "../W05_data/2D/", "W05_data/2D/", "./W05_data/2D/", "/workspace/W05_data/2D/"]:
    if os.path.exists(candidate):
        data_dir = candidate
        break
if data_dir is None:
    data_dir = "../mri_data/W05_data/2D/"
mesh = dl.Mesh(data_dir + name + ".xml")
dx = dl.Measure("dx", domain=mesh)

# Function Space
Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vh_param = dl.FunctionSpace(mesh, dl.MixedElement([Vh1.ufl_element(), Vh1.ufl_element(), Vh1.ufl_element()]))
Vh = [Vh2, Vh_param, Vh2]

ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
print(f"STATE={ndofs[0]}, PARAMETER={ndofs[1]}, ADJOINT={ndofs[2]}")

# Load Data
h5 = dl.HDF5File(mesh.mpi_comm(), data_dir + "W05.h5", "r")

time_to_group = {
    2: "/W05/W05_0",
    4: "/W05/W05_1",
    5: "/W05/W05_2",
    6: "/W05/W05_3",
    9: "/W05/W05_4",
}

data = {}
tlist = [2,4,5,6,9]
for t in tlist:
    u = dl.Function(Vh[STATE])
    h5.read(u, time_to_group[t])
    data[str(t)] = u

h5.close()

obs_out = dl.XDMFFile("TumorData.xdmf")
obs_out.parameters["flush_output"] = True
obs_out.parameters["functions_share_mesh"] = True

for t in tlist:
    f = dl.Function(Vh[STATE])
    f.assign(data[str(t)])
    f.rename("tumor", "tumor")   # IMPORTANT for consistent naming
    obs_out.write(f, float(t))

obs_out.close()

#correlation length and sigmas [E, D, G]
lc1, sigma1 = 5.5, 0.1
lc2, sigma2 = 6.5, 0.3
lc3, sigma3 = 3.5, 0.08
gamma1, delta1 = computeGammaDelta(lc1, sigma1)
gamma2, delta2 = computeGammaDelta(lc2, sigma2)
gamma3, delta3 = computeGammaDelta(lc3, sigma3)


gamma = [gamma1, gamma2, gamma3]
delta = [delta1, delta2, delta3]
theta_x = 0.55
theta_y = 0.55

anis_diff = dl.Constant([[theta_x, 0.], [0., theta_y]])
prior = VectorBiLaplacianPrior(Vh[PARAMETER], gamma, delta,Theta=anis_diff,robin_bc=True)

mean = dl.Function(Vh[PARAMETER])

# Create constant expressions for each field
mean_val1 = dl.Constant(-0.1562)
mean_val2 = dl.Constant(-0.3492)
mean_val3 = dl.Constant(-1.8965)

# Extract subspaces from the mixed space
V0 = Vh[PARAMETER].sub(0).collapse()
V1 = Vh[PARAMETER].sub(1).collapse()
V2 = Vh[PARAMETER].sub(2).collapse()

# Interpolate constants into the subspaces
mean_0 = dl.interpolate(mean_val1, V0)
mean_1 = dl.interpolate(mean_val2, V1)
mean_2 = dl.interpolate(mean_val3, V2)

# Assign the subfield values into the mixed function
assigner = dl.FunctionAssigner(Vh[PARAMETER], [V0, V1,V2])
assigner.assign(mean, [mean_0, mean_1, mean_2])

# Set it as the prior mean
prior.mean.axpy(1.0, mean.vector())

#------------------------ Initial Condition -------------------------
t0_mat = os.path.join(data_dir, "tumor_t0.mat")
if os.path.exists(t0_mat):
    u0 = interp(t0_mat, "cells_t", Vh[STATE])
else:
    u0 = dl.Function(Vh[STATE])
    u0.assign(data[str(tlist[0])])

#----------------------- PDE Model Setup ------------------------
dt = 1.0
obs_times = [2, 4, 5, 6]
sim_times = [2, 4, 5, 6, 9]
H_D_val = 1.5
log_H_val = math.log(H_D_val)
pde_varf = MechCoupledTumorVarf(Vh, dt, dx=dl.dx, log_H_fixed=dl.Constant(log_H_val))
pde = TimeDependentPDEVariationalProblem(
    Vh,
    pde_varf, 
    [],
    [],
    u0,
    t_init=sim_times[0],
    t_final=sim_times[-1], 
    is_fwd_linear=False,
    times=sim_times
)
   
#--------------------- Misfit setup ------------------------

misfits = []
for t in obs_times:
    #t_val = float(t)
    misfit_t = ContinuousStateObservation(Vh[STATE], dl.dx, [])
    obs_vec = data.get(str(int(t)))
    if obs_vec is not None:
        misfit_t.d.axpy(1.0, obs_vec.vector())
        
    else:
        raise RuntimeError(f"Missing observation at time {t}")
    misfit_t.noise_variance = args.sigma**2
    misfits.append(misfit_t)
    
misfit = MisfitTD(misfits, obs_times)

#-------------------- Solve for the MAP estimate -------------------------
print( sep, "Find the MAP point", sep)
model = Model(pde, prior, misfit)
m0 = prior.mean.copy()

params = ReducedSpaceNewtonCG_ParameterList()

params["rel_tolerance"] = 1e-6
params["abs_tolerance"] = 1e-10
params["max_iter"] = args.max_iter
params["globalization"] = "LS"
params["print_level"] = 0
params["GN_iter"] = 10
params["cg_coarse_tolerance"] = 1e-6
params["cg_max_iter"] = 200
params["gdm_tolerance"] = 1e-12

solver = ReducedSpaceNewtonCG(model, params)
x = solver.solve([None, m0, None])
t_map = time.time() - start_time
print("\n##################################################################")
print ("Termination reason: ", solver.termination_reasons[solver.reason])
print ("Final gradient norm: ", solver.final_grad_norm)
print ("Final cost: ", solver.final_cost)
print ("Final simulation time (min): ", t_map/60)
convergence = solver.converged
print( "\nConverged in ", solver.it, " iterations.")
print("##################################################################")

# Save MAP
map_func = vector2Function(x[PARAMETER], Vh[PARAMETER], name="MAP")
dlx = dl.XDMFFile("MAP.xdmf")
dlx.parameters["flush_output"] = True
dlx.parameters["functions_share_mesh"] = True
dlx.write(map_func, 0.0)
dlx.close()

# Save Prediction
pde.solveFwd(x[STATE], x)
pde.exportState(x[STATE], "state_trajectory.xdmf")

print(sep, "Compute low-rank Gaussian Approximation of the posterior", sep)

model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
Hmisfit = ReducedHessian(model, misfit_only=True)

k = 50      # number of dominant modes
p = 20      # oversampling
print(f"Double Pass Algorithm: computing {k} modes with oversampling {p}")

Omega = MultiVector(x[PARAMETER], k + p)
parRandom.normal(1.0, Omega)

d, eU = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
posterior = GaussianLRPosterior(prior, d, eU)
posterior.mean = x[PARAMETER]

# ----------------------------------------------------------------------
# Compute Model Evidence (Laplace Approximation)
# ----------------------------------------------------------------------
print(sep, "Computing model evidence (Laplace Approximation)", sep)
print("\nRunning detailed evidence diagnostics...")

Nd_check = 0
for t in obs_times:
    v = data[str(t)].vector().get_local()
    Nd_check += v.size
print("Total Nd from data =", Nd_check)

residuals = []
for t in obs_times:
    xi_t = pde.generate_static_state()
    x[STATE].retrieve(xi_t, float(t))
    pred = vector2Function(xi_t, Vh[STATE])
    obs_vec = data[str(t)]
    r = pred.vector().get_local() - obs_vec.vector().get_local()
    residuals.append(r)
res_all = np.hstack(residuals)
Phi_direct = 0.5 * np.dot(res_all, res_all) / (args.sigma**2)
print(f"Phi_direct (manual residual) = {Phi_direct:.4e}")

# --- Misfit term Φ(m_MAP) ---
Phi_MAP = model.misfit.cost(x)
print("Φ(m_MAP) =", Phi_MAP)

# --- Prior norm term: (m_MAP - m_pr)^T Γ_pr^{-1} (m_MAP - m_pr) ---
m_diff = x[PARAMETER].copy()
m_diff.axpy(-1.0, prior.mean)  # m_MAP - m_pr

# Solve R * y = m_diff for y, then prior_norm_sq = m_diff^T y
y_temp = dl.Vector(m_diff)              # allocate solution vector
prior.Rsolver.solve(y_temp, m_diff)     # solves R y_temp = m_diff
prior_norm_sq = m_diff.inner(y_temp)
print("Prior norm squared =", prior_norm_sq)

# --- Noise variance / normalization term ---
sigma_noise = args.sigma
try:
    Nd = sum([m.d.size() for m in misfits])  # if misfit stores .d vector with .size()
except Exception:
    Nd = 0
    for m in misfits:
        try:
            Nd += m.d.size()
        except Exception:
            try:
                Nd += len(m.d)
            except Exception:
                pass

term_noise = -0.5 * Nd * math.log(2.0 * math.pi * sigma_noise**2)
print("Noise term =", term_noise, "  (Nd =", Nd, ", sigma_noise =", sigma_noise, ")")

lambda_vals = np.asarray(d).ravel()   # shape (k,)
r = lambda_vals.size
if r == 0:
    term_eig = 0.0
else:
    # numerically stable log(1+lambda) via log1p
    term_eig = -0.5 * np.sum(np.log1p(lambda_vals))
print(f"Using r = {r} eigenvalues; Eigenvalue correction = {term_eig:.6f}")
# --- Effective dimension (diagnostic) ---
eff_dim = np.sum(lambda_vals / (1.0 + lambda_vals))
print(f"Effective dimension = {eff_dim:.3f}")

# Detailed evidence breakdown
log_evidence = -Phi_MAP - 0.5 * prior_norm_sq + term_noise + term_eig
print("\nEVIDENCE COMPONENTS:")
print(f"   -Phi_MAP                = {-Phi_MAP:.4e}")
print(f"   -0.5*prior_norm_sq      = {-0.5*prior_norm_sq:.4e}")
print(f"   term_noise              = {term_noise:.4e}")
print(f"   term_eig (correction)   = {term_eig:.4e}")

print("\n######################## Model Evidence ########################")
print(f"log_evidence = {log_evidence:.12e}")
try:
    evidence_val = math.exp(log_evidence)
    print(f"Evidence = {evidence_val:.6e}")
except OverflowError:
    print("Evidence too small to exponentiate safely (underflow). Keep log_evidence instead.")
print("################################################################")

# ------------------------
# ------------------------
def compute_dice_score(f1, f2, threshold=0.25):
    """
    Computes DICE between two FEniCS Functions by DOF thresholding.
    """
    f1_vals = f1.vector().get_local()
    f2_vals = f2.vector().get_local()

    f1_mask = (f1_vals > threshold).astype(np.int8)
    f2_mask = (f2_vals > threshold).astype(np.int8)

    intersection = np.sum(f1_mask * f2_mask)
    total = np.sum(f1_mask) + np.sum(f2_mask)

    return (2.0 * intersection) / (total + 1e-12)

# ----------------------------------------------------------------------
# Posterior NTA Evaluation: 50 samples per time point
# ----------------------------------------------------------------------
print(sep, "Evaluating NTA across posterior samples", sep)

num_nta_samples = 50

# Evaluate NTA at ALL internal solver times
eval_times = list(pde.times)

brain_area = dl.assemble(dl.Constant(1.0) * dx)

# Storage
nta_results = {t: [] for t in eval_times}
nta_mri_dict = {}

# Compute MRI NTA only where MRI exists
for t in eval_times:
    key = str(int(t))
    if key in data:
        true = data[key]
        true_indicator = dl.Function(Vh[STATE])
        true_indicator.vector().set_local(
            np.where(true.vector().get_local() > 0.25, 1.0, 0.0)
        )
        nta_mri_dict[t] = dl.assemble(true_indicator * dx) / brain_area
    else:
        nta_mri_dict[t] = np.nan

# Initialize noise vector
noise = dl.Vector()
posterior.init_vector(noise, "noise")

# File for all NTA samples
with open("NTA_posterior_samples.txt", "w") as f:
    f.write("Time,SampleID,NTA_model,NTA_mri\n")

    for i in range(num_nta_samples):
        print(f"NTA posterior sample {i+1}/{num_nta_samples}")
        parRandom.normal(1.0, noise)

        # Draw posterior sample
        pr_s = model.generate_vector(PARAMETER)
        post_s = model.generate_vector(PARAMETER)
        posterior.sample(noise, pr_s, post_s, add_mean=True)

        # Solve PDE forward for this sample
        u = pde.generate_state()
        x_sample = [u, post_s, None]
        pde.solveFwd(x_sample[STATE], x_sample)

        for t in eval_times:
            xi_t = pde.generate_static_state()
            x_sample[STATE].retrieve(xi_t, float(t))
            pred = vector2Function(xi_t, Vh[STATE])

            # Compute NTA_model
            nta_indicator = dl.Function(Vh[STATE])
            nta_indicator.vector().set_local(np.where(pred.vector().get_local() > 0.25, 1.0, 0.0))
            nta_model = dl.assemble(nta_indicator * dx) / brain_area

            nta_results[t].append(nta_model)
            f.write(f"{t},{i},{nta_model:.6f},{nta_mri_dict[t]:.6f}\n")

# --- Compute summary statistics ---
mean_nta = [np.mean(nta_results[t]) for t in eval_times]
std_nta = [np.std(nta_results[t]) for t in eval_times]
nta_mri_values = [nta_mri_dict[t] for t in eval_times]

# Save summary results
with open("NTA_summary.txt", "w") as f:
    f.write("Time,NTA_model,NTA_mri\n")
    for t, mean_val, nta_mri_val in zip(eval_times, mean_nta, nta_mri_values):
        f.write(f"{t},{mean_val:.6f},{nta_mri_val:.6f}\n")

# ----------------------------------------------------------------------
# Posterior DICE Evaluation: Mean ± Std across posterior samples
# ----------------------------------------------------------------------
print(sep, "Evaluating posterior DICE", sep)

num_dice_samples = 50
eval_times = [t for t in sim_times if str(t) in data]

dice_results = {t: [] for t in eval_times}

noise = dl.Vector()
posterior.init_vector(noise, "noise")

for i in range(num_dice_samples):
    print(f"DICE posterior sample {i+1}/{num_dice_samples}")
    parRandom.normal(1.0, noise)

    pr_s = model.generate_vector(PARAMETER)
    post_s = model.generate_vector(PARAMETER)
    posterior.sample(noise, pr_s, post_s, add_mean=True)

    u = pde.generate_state()
    x_sample = [u, post_s, None]
    pde.solveFwd(x_sample[STATE], x_sample)

    for t in eval_times:
        xi_t = pde.generate_static_state()
        x_sample[STATE].retrieve(xi_t, float(t))
        pred = vector2Function(xi_t, Vh[STATE])
        true = data[str(t)]

        score = compute_dice_score(pred, true, threshold=0.25)
        dice_results[t].append(score)

# Save raw samples
with open("DICE_posterior_samples.txt","w") as f:
    f.write("Time,SampleID,DICE\n")
    for t in eval_times:
        for i,val in enumerate(dice_results[t]):
            f.write(f"{t},{i},{val:.6f}\n")

mean_dice = [np.mean(dice_results[t]) for t in eval_times]
std_dice  = [np.std(dice_results[t])  for t in eval_times]

# ----------------------------------------------------------------------
# FINAL MAP MECHANICS EXPORT: Displacement and Hyper-elastic Stress
# ----------------------------------------------------------------------
print(sep, "Computing MAP Mechanics (Stress/Displacement) for Trajectory", sep)

# File Setup
mech_xdmf = dl.XDMFFile("MAP_mechanics_results.xdmf")
mech_xdmf.parameters["flush_output"] = True
mech_xdmf.parameters["functions_share_mesh"] = True

# Function Space for Magnitudes and visualization
V_mag = Vh1 

# Extract the MAP parameter for Stiffness (log_E)
# map_func is already created from x[PARAMETER] earlier in the script
log_E_map, _, _ = map_func.split(deepcopy=True)

# Iterate through all simulation times (start from t=2)
for t in sim_times:
    xi_t = pde.generate_static_state()
    x[STATE].retrieve(xi_t, float(t))
    pred_tumor = vector2Function(xi_t, Vh[STATE])
    
    if float(t) == float(sim_times[0]):
        # Force EVERYTHING to be absolute zero at the start
        disp_mag = dl.interpolate(dl.Constant(0.0), V_mag)
        piola_mag = dl.interpolate(dl.Constant(0.0), V_mag)

        print(f"Time t = {t:.2f}: Using forced absolute zeros.")
    else:
        # solve mechanics for all time after the initial MRI time point
        u_mech = pde_varf.solve_mechanics(pred_tumor, log_E_map, pde_varf.log_H_fixed)
        
        # Calculate magnitudes
        disp_mag = project(sqrt(dot(u_mech, u_mech)), V_mag)
        
        P2_tensor = pde_varf.first_piola_from_energy(u_mech, log_E_map)
        piola_mag = project(sqrt(inner(P2_tensor, P2_tensor)), V_mag)

    # Rename for Paraview consistency (Important!)
    disp_mag.rename("Displacement_Magnitude", "disp_mag")
    piola_mag.rename("First_Piola_Kirchhoff_Magnitude", "piola_mag")

    # Export
    mech_xdmf.write(disp_mag, float(t))
    mech_xdmf.write(piola_mag, float(t))

mech_xdmf.close()
print("MAP Mechanics saved to 'MAP_mechanics_results.xdmf'")


# --------------------------------------------------------
# MAP DICE + NTA + Volume block
# --------------------------------------------------------
print(sep, "Computing MAP DICE / NTA / Volume", sep)

eval_times = [t for t in sim_times if str(t) in data]

dice_file = open("DICE_scores.txt", "w")
dice_file.write("Time,DICE\n")

nta_file = open("nta.txt", "w")
nta_file.write("Time,NTA_model,NTA_mri\n")

vol_file = open("TumorVolumes.txt", "w")
vol_file.write("Time,Volume\n")

brain_area = dl.assemble(dl.Constant(1.0) * dx)

# --- Paraview writers ---
map_pred_xdmf = dl.XDMFFile("MAP_predictions.xdmf")
map_pred_xdmf.parameters["flush_output"] = True
map_pred_xdmf.parameters["functions_share_mesh"] = True

map_mask_xdmf = dl.XDMFFile("MAP_indicator_masks.xdmf")
map_mask_xdmf.parameters["flush_output"] = True
map_mask_xdmf.parameters["functions_share_mesh"] = True

for t in eval_times:
    # retrieve state at time t
    xi_t = pde.generate_static_state()
    x[STATE].retrieve(xi_t, float(t))
    pred = vector2Function(xi_t, Vh[STATE], name=f"MAP_pred_t{t}")
    true = data[str(t)]

    # --- Save tumor field for paraview ---
    map_pred_xdmf.write(pred, float(t))

    # --- New DICE ---
    dice_score = compute_dice_score(pred, true, threshold=0.25)

    # --- New NTA / Volume ---
    pred_indicator = dl.Function(Vh[STATE], name=f"MAP_indicator_t{t}")
    pred_indicator.vector().set_local(
        np.where(pred.vector().get_local() > 0.25, 1.0, 0.0)
    )

    # save mask for paraview
    map_mask_xdmf.write(pred_indicator, float(t))

    nta_model = dl.assemble(pred_indicator * dx) / brain_area

    true_indicator = dl.Function(Vh[STATE])
    true_indicator.vector().set_local(
        np.where(true.vector().get_local() > 0.25, 1.0, 0.0)
    )
    nta_mri = dl.assemble(true_indicator * dx) / brain_area

    # volume (non-normalized)
    vol = dl.assemble(pred_indicator * dx)

    # --- Save text values ---
    dice_file.write(f"{t},{dice_score:.6f}\n")
    nta_file.write(f"{t},{nta_model:.6f},{nta_mri:.6f}\n")
    vol_file.write(f"{t},{vol:.6f}\n")

dice_file.close()
nta_file.close()
vol_file.close()
map_pred_xdmf.close()
map_mask_xdmf.close()

print("MAP DICE / NTA / Volume stored in TXT + XDMF files.")
print("Inference complete. Results stored as XDMF and PVD files.")
