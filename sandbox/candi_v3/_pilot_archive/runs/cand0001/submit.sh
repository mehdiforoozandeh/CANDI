#!/bin/bash
#SBATCH --account=def-maxwl_gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=12G
#SBATCH --time=0:10:00
#SBATCH --output=/project/6014832/mforooz/EpiDenoise/sandbox/candi_v3/runs/cand0001/slurm-%j.out

source /project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate 2>/dev/null
module load samtools 2>/dev/null
cd /project/6014832/mforooz/EpiDenoise
export PYTHONPATH=/project/6014832/mforooz/EpiDenoise

cd /project/6014832/mforooz/EpiDenoise/sandbox/candi_v3/runs/cand0001
/project/6014832/mforooz/EpiDenoise/candi_venv/bin/python -u program.py
