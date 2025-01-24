#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1 
#SBATCH --mem=100GB # total memory
#SBATCH --time=24:00:00 	
#SBATCH --account=<accountname>
#SBATCH --partition=<partitionname>
#SBATCH --output=output/%j.out
#SBATCH --array=1-3

# Get started
echo " "
echo "Job started on `hostname` at `date`"
echo " "

# Environment setup
module purge
module load r/4.3.2
export R_LIBS="/path/to/rlibs"

option=$SLURM_ARRAY_TASK_ID
if [ $option -eq "1" ]; then
    Rscript fit_species.R "species = 'ecoli'"
elif [ $option -eq "2" ]; then
    Rscript fit_species.R "species = 'yeast'"
elif [ $option -eq "3" ]; then
    Rscript fit_species.R "species = 'human'"
else
    echo "unknown option"
fi

# Finish up
echo " "
echo "Job Ended at `date`"
echo " "


