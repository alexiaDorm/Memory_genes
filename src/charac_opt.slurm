#!/bin/bash -l

#SBATCH --job-name="test"
#SBATCH --error="test%a.err"
#SBATCH --output="test%a.out"
#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --chdir /scratch/dormann/Memory_genes/src
#SBATCH --mem 64G
#SBATCH --mail-type=END
#SBATCH --mail-user=alexia.dormann@epfl.ch
 
echo STARTING AT `date`

      
module load gcc/8.4.0
module load intel/19.0.5
module load python/3.7.7

srun python LogRegopt.py 
 
echo FINISHED at `date`

