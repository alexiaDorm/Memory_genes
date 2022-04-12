#!/bin/bash -l

#SBATCH --nodes 1-1
#SBATCH --exclusive
#SBATCH --mem 60000
#SBATCH --time 00:05:00
 
echo STARTING AT `date`

module load gcc 
module load phython

srun python mixdata.py
 
echo FINISHED at `date`
