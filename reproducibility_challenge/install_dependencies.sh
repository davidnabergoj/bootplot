#!/bin/bash
# Installs everything needed to reproduce Figure 6 into THIS directory.
# Vanilla macOS 11+, no admin password, no prompts, no system changes.
# Run from the replicability directory:  bash install_dependencies.sh
set -e
cd "$(dirname "$0")"

# Miniconda (self-contained Python; picks Apple Silicon or Intel build)
curl -L -o miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-$(uname -m).sh"
bash miniconda.sh -b -u -p ./miniconda3 && rm miniconda.sh

# Do not register this local environment in the user's ~/.conda/environments.txt
# This avoids intentionally modifying conda bookkeeping files outside this directory.
export CONDA_REGISTER_ENVS=false

# Python 3.12 + pip + pycairo (prebuilt, bundles the cairo library), then bootplot
rm -rf ./env
./miniconda3/bin/conda create -y -p ./env --override-channels -c conda-forge python=3.12 pip pycairo
./env/bin/pip install --no-cache-dir "bootplot==0.0.18"

# Reproduce the figure and show it
./env/bin/python reproduce_figure6.py
open mean_estimate_95.png
echo "Done. The figure was saved as mean_estimate_95.png in this directory."
