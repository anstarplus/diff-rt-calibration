#!/bin/bash
#SBATCH --job-name=test-training-cross-band      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=a100        # choose a100 or v100
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=za1320@princeton.edu
module purge
module load anaconda3/2023.9
conda activate tfanplus

## python -u ../dataset/gen_dataset.py
cd /home/za1320/Github/diff-rt-calibration/code/
python -u train_crossband.py
# python -u gen_dataset.py