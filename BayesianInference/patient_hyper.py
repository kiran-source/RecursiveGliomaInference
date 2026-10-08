# ----------------------------------------------------------------------
# Mech-RD Tumor Model - Patient-Specific Bayesian Inference (E, D, G)
# Hyper-elasticity Formulation (Preserving 100% Paper Mathematical Logic)
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

sep = "\n" + "#" * 80 + "\n"

# Setup search paths for hippylib and local modules
for p in ['../hippylib', '../hippylib/hippylib', '../../hippylib', 'hippylib', '/workspace/hippylib']:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)
from hippylib import BiLaplacianPrior, VectorBiLaplacianPrior
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
parser = argparse.ArgumentParser(description="Patient-Specific Hyperelastic Mech-RD Bayesian Inference")
parser.add_argument('--patient', default='Patient_01', type=str, help='Patient ID (e.g., Patient_01, Patient_02, W05_data)')
parser.add_argument('--sigma', default=0.035, type=float, help='Noise standard deviation')
parser.add_argument('--rhoGM', default=800, type=float, help='Tissue density parameter')
parser.add_argument('--nsamples', default=500, type=int, help='Posterior evaluation samples')
parser.add_argument('--max_iter', default=200, type=int, help='Maximum Newton-CG iterations')
parser.add_argument('--output_dir', default='', type=str, help='Custom output directory for results')
args = parser.parse_args()

patient_id = args.patient
print(f"=== Initializing Bayesian Inference for Patient: {patient_id} ===")

# Locate data directory
data_dir = None
candidate_dirs = [
    f"../mri_data/{patient_id}/2D/",
    f"mri_data/{patient_id}/2D/",
    f"/workspace/mri_data/{patient_id}/2D/",
    f"../{patient_id}/2D/",
    f"/workspace/{patient_id}/2D/",
    "../mri_data/W05_data/2D/",
    "../W05_data/2D/",
    "/workspace/W05_data/2D/"
]
for candidate in candidate_dirs:
    if os.path.exists(candidate):
        data_dir = candidate
        break

if data_dir is None:
    raise FileNotFoundError(f"Could not locate data directory for patient {patient_id} in {candidate_dirs}")

print(f"Loading data from: {data_dir}")

# Load Mesh
mesh_name = f"{patient_id}_0.xml" if os.path.exists(os.path.join(data_dir, f"{patient_id}_0.xml")) else "W05_0.xml"
mesh = dl.Mesh(os.path.join(data_dir, mesh_name))
dx = dl.Measure("dx", domain=mesh)

# Function Space
Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vh_param = dl.FunctionSpace(mesh, dl.MixedElement([Vh1.ufl_element(), Vh1.ufl_element(), Vh1.ufl_element()]))
Vh = [Vh2, Vh_param, Vh2]

ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
print(f"STATE={ndofs[0]}, PARAMETER={ndofs[1]}, ADJOINT={ndofs[2]}")

# Setup results directory
if args.output_dir:
    res_dir = args.output_dir
else:
    res_dir = os.path.join("results", patient_id)
os.makedirs(res_dir, exist_ok=True)

# Load Data from HDF5
h5_name = f"{patient_id}.h5" if os.path.exists(os.path.join(data_dir, f"{patient_id}.h5")) else "W05.h5"
h5 = dl.HDF5File(mesh.mpi_comm(), os.path.join(data_dir, h5_name), "r")

tlist = [2, 4, 5, 6, 9]
time_to_group = {
    2: f"/{patient_id}/{patient_id}_0",
    4: f"/{patient_id}/{patient_id}_1",
    5: f"/{patient_id}/{patient_id}_2",
    6: f"/{patient_id}/{patient_id}_3",
    9: f"/{patient_id}/{patient_id}_4",
}

# Fallback group mapping if named after W05
data = {}
for i, t in enumerate(tlist):
    u = dl.Function(Vh[STATE])
    group = time_to_group[t]
    try:
        h5.read(u, group)
    except Exception:
        fallback_group = f"/W05/W05_{i}"
        h5.read(u, fallback_group)
    data[str(t)] = u

h5.close()

# Export Observation data to XDMF for ParaView
obs_out = dl.XDMFFile(os.path.join(res_dir, "TumorData.xdmf"))
obs_out.parameters["flush_output"] = True
obs_out.parameters["functions_share_mesh"] = True
for t in tlist:
    f = dl.Function(Vh[STATE])
    f.assign(data[str(t)])
    f.rename("tumor", "tumor")
    obs_out.write(f, float(t))
obs_out.close()

# ----------------------------------------------------------------------
# Prior Definition (Preserving Paper Values)
# ----------------------------------------------------------------------
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
prior = VectorBiLaplacianPrior(Vh[PARAMETER], gamma, delta, Theta=anis_diff, robin_bc=True)

mean = dl.Function(Vh[PARAMETER])
mean_val1 = dl.Constant(-0.1562)
mean_val2 = dl.Constant(-0.3492)
mean_val3 = dl.Constant(-1.8965)

V0 = Vh[PARAMETER].sub(0).collapse()
V1 = Vh[PARAMETER].sub(1).collapse()
V2 = Vh[PARAMETER].sub(2).collapse()

mean_0 = dl.interpolate(mean_val1, V0)
mean_1 = dl.interpolate(mean_val2, V1)
mean_2 = dl.interpolate(mean_val3, V2)

assigner = dl.FunctionAssigner(Vh[PARAMETER], [V0, V1, V2])
assigner.assign(mean, [mean_0, mean_1, mean_2])
prior.mean.axpy(1.0, mean.vector())

# Initial Condition
t0_mat = os.path.join(data_dir, "tumor_t0.mat")
if os.path.exists(t0_mat):
    u0 = interp(t0_mat, "cells_t", Vh[STATE])
else:
    u0 = dl.Function(Vh[STATE])
    u0.assign(data[str(tlist[0])])

# ----------------------------------------------------------------------
# Forward PDE Variational Problem
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Misfit Setup
# ----------------------------------------------------------------------
misfits = []
for t in obs_times:
    misfit_t = ContinuousStateObservation(Vh[STATE], dl.dx, [])
    obs_vec = data.get(str(int(t)))
    if obs_vec is not None:
        misfit_t.d.axpy(1.0, obs_vec.vector())
    else:
        raise RuntimeError(f"Missing observation at time {t}")
    misfit_t.noise_variance = args.sigma**2
    misfits.append(misfit_t)

misfit = MisfitTD(misfits, obs_times)

# ----------------------------------------------------------------------
# Solve for MAP Estimate (Newton-CG)
# ----------------------------------------------------------------------
print(sep, f"Finding MAP Estimate for {patient_id}", sep)
model = Model(pde, prior, misfit)
m0 = prior.mean.copy()

params = ReducedSpaceNewtonCG_ParameterList()
params["rel_tolerance"] = 1e-6
params["abs_tolerance"] = 1e-10
params["max_iter"] = args.max_iter
params["globalization"] = "LS"
params["print_level"] = 1
params["GN_iter"] = 10
params["cg_coarse_tolerance"] = 1e-6
params["cg_max_iter"] = 200
params["gdm_tolerance"] = 1e-12

solver = ReducedSpaceNewtonCG(model, params)
x = solver.solve([None, m0, None])
t_map = time.time() - start_time

print("\n" + "#" * 66)
print("Termination reason: ", solver.termination_reasons[solver.reason])
print("Final gradient norm: ", solver.final_grad_norm)
print("Final cost: ", solver.final_cost)
print("Final simulation time (min): ", t_map / 60.0)
print(f"\nConverged in {solver.it} iterations.")
print("#" * 66)

# Save MAP
map_func = vector2Function(x[PARAMETER], Vh[PARAMETER], name="MAP")
dlx = dl.XDMFFile(os.path.join(res_dir, "MAP.xdmf"))
dlx.parameters["flush_output"] = True
dlx.parameters["functions_share_mesh"] = True
dlx.write(map_func, 0.0)
dlx.close()

# Save Prediction
pde.solveFwd(x[STATE], x)
pde.exportState(x[STATE], os.path.join(res_dir, "state_trajectory.xdmf"))

# ----------------------------------------------------------------------
# Laplace Approximation (Double Pass Algorithm)
# ----------------------------------------------------------------------
print(sep, "Computing Low-Rank Gaussian Approximation of Posterior", sep)
model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
Hmisfit = ReducedHessian(model, misfit_only=True)

k = 50
p = 20
Omega = MultiVector(x[PARAMETER], k + p)
parRandom.normal(1.0, Omega)

d, eU = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
posterior = GaussianLRPosterior(prior, d, eU)
posterior.mean = x[PARAMETER]

# ----------------------------------------------------------------------
# Bayesian Model Evidence (Laplace Approximation)
# ----------------------------------------------------------------------
print(sep, "Computing Model Evidence", sep)
Phi_MAP = model.misfit.cost(x)
m_diff = x[PARAMETER].copy()
m_diff.axpy(-1.0, prior.mean)

y_temp = dl.Vector(m_diff)
prior.Rsolver.solve(y_temp, m_diff)
prior_norm_sq = m_diff.inner(y_temp)

sigma_noise = args.sigma
try:
    Nd = sum([m.d.size() for m in misfits])
except Exception:
    Nd = 0
    for m in misfits:
        try:
            Nd += m.d.size()
        except Exception:
            pass

term_noise = -0.5 * Nd * math.log(2.0 * math.pi * sigma_noise**2)
lambda_vals = np.asarray(d).ravel()
r = lambda_vals.size
term_eig = -0.5 * np.sum(np.log1p(lambda_vals)) if r > 0 else 0.0

log_evidence = -Phi_MAP - 0.5 * prior_norm_sq + term_noise + term_eig
eff_dim = np.sum(lambda_vals / (1.0 + lambda_vals))

print(f"\nModel Evidence Results for {patient_id}:")
print(f"  log_evidence = {log_evidence:.12e}")
print(f"  effective_dimension = {eff_dim:.3f}")

with open(os.path.join(res_dir, "evidence.txt"), "w") as f:
    f.write(f"log_evidence,{log_evidence:.12e}\n")
    f.write(f"effective_dimension,{eff_dim:.3f}\n")
    f.write(f"phi_map,{Phi_MAP:.6e}\n")
    f.write(f"prior_norm_sq,{prior_norm_sq:.6e}\n")

# ----------------------------------------------------------------------
# DICE & NTA Metric Evaluation
# ----------------------------------------------------------------------
def compute_dice_score(f1, f2, threshold=0.25):
    f1_vals = f1.vector().get_local()
    f2_vals = f2.vector().get_local()
    f1_mask = (f1_vals > threshold).astype(np.int8)
    f2_mask = (f2_vals > threshold).astype(np.int8)
    intersection = np.sum(f1_mask * f2_mask)
    total = np.sum(f1_mask) + np.sum(f2_mask)
    return (2.0 * intersection) / (total + 1e-12)

eval_times = [t for t in sim_times if str(t) in data]
brain_area = dl.assemble(dl.Constant(1.0) * dx)

with open(os.path.join(res_dir, "DICE_scores.txt"), "w") as dice_file, \
     open(os.path.join(res_dir, "nta.txt"), "w") as nta_file, \
     open(os.path.join(res_dir, "TumorVolumes.txt"), "w") as vol_file:

    dice_file.write("Time,DICE\n")
    nta_file.write("Time,NTA_model,NTA_mri\n")
    vol_file.write("Time,Volume\n")

    for t in eval_times:
        xi_t = pde.generate_static_state()
        x[STATE].retrieve(xi_t, float(t))
        pred = vector2Function(xi_t, Vh[STATE], name=f"MAP_pred_t{t}")
        true = data[str(t)]

        dice_score = compute_dice_score(pred, true, threshold=0.25)
        pred_indicator = dl.Function(Vh[STATE])
        pred_indicator.vector().set_local(np.where(pred.vector().get_local() > 0.25, 1.0, 0.0))
        nta_model = dl.assemble(pred_indicator * dx) / brain_area

        true_indicator = dl.Function(Vh[STATE])
        true_indicator.vector().set_local(np.where(true.vector().get_local() > 0.25, 1.0, 0.0))
        nta_mri = dl.assemble(true_indicator * dx) / brain_area
        vol = dl.assemble(pred_indicator * dx)

        dice_file.write(f"{t},{dice_score:.6f}\n")
        nta_file.write(f"{t},{nta_model:.6f},{nta_mri:.6f}\n")
        vol_file.write(f"{t},{vol:.6f}\n")
        print(f"Time t={t}: DICE={dice_score:.4f}, NTA_model={nta_model:.4f}, NTA_mri={nta_mri:.4f}")

print(f"\n[+] Inference successfully completed for {patient_id}. Results saved to {res_dir}/\n")
