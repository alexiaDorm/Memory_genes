#!/bin/bash -l

#SBATCH --job-name="test"
#SBATCH --error="test%a.err"
#SBATCH --output="test%a.out"
#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --chdir /home/dormann/Memory_genes/src
#SBATCH --mem 64G
#SBATCH --mail-type=END
#SBATCH --mail-user=alexia.dormann@epfl.ch

norm_path='../data/processed_data/AE3.csv'
 
echo STARTING AT `date`

module load gcc/9.3.0
module load python/3.7.7

srun python mixdata.py $norm_path
 
echo FINISHED at `date`

