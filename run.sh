#!/usr/bin/env bash
set -e

SIGMA=${1:-0.035}
RHOGM=${2:-800}
NSAMPLES=${3:-500}
MAX_ITER=${4:-200}

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "Recursive Glioma Inference (FEniCS + hIPPYlib)"
echo "=========================================================="
echo "Parameters: sigma=$SIGMA, rhoGM=$RHOGM, nsamples=$NSAMPLES, max_iter=$MAX_ITER"
echo "=========================================================="

docker run --rm -v "${DIR}:/workspace" -w /workspace/BayesianInference hippylib/fenics \
  "python3 Rat-3_hyper.py --sigma $SIGMA --rhoGM $RHOGM --nsamples $NSAMPLES --max_iter $MAX_ITER"
