#!/bin/bash
module load anaconda3/2023.09
source $(conda info --base)/etc/profile.d/conda.sh
conda activate liif_env
conda install -c conda-forge pyevtk -y
