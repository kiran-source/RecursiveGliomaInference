#!/usr/bin/env bash
set -e

PATIENT=${1:-"Patient_001"}
SIGMA=${2:-0.035}
RHOGM=${3:-800}
NSAMPLES=${4:-100}
MAX_ITER=${5:-20}

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "Recursive Glioma Inference - Cohort Runner"
echo "Patient: $PATIENT, max_iter: $MAX_ITER, nsamples: $NSAMPLES"
echo "=========================================================="

docker run --rm -v "${DIR}:/workspace" -w /workspace/BayesianInference hippylib/fenics \
  "python3 patient_hyper.py --patient $PATIENT --sigma $SIGMA --rhoGM $RHOGM --nsamples $NSAMPLES --max_iter $MAX_ITER"
