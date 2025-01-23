#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=<NUMBER_OF_TASKS>
#SBATCH --cpus-per-task=<CPUS_PER_TASK>
#SBATCH --mem=<MEMORY>
#SBATCH --time=<TIME_LIMIT>
#SBATCH -A <ACCOUNT_NAME>
#SBATCH -p <PARTITION>
#SBATCH -q <QOS>

# Go to submission directory
cd $SLURM_SUBMIT_DIR

# Create a directory for the run based on SPLITID and navigate into it
mkdir run_${SPLITID}
cd run_${SPLITID}
cp ../<INPUT_FILE_PREFIX>_split${SPLITID} .

# Environment setup
module use <MODULE_PATH>
module load <MODULE_NAME>
conda activate <CONDA_ENV_PATH>

# Use fewer threads for the first iteration for performance reasons
numthreads=<INITIAL_NUM_THREADS>
while IFS=',' read -ra array; do
  echo ${array[2]} > rawseq.txt
  echo "Starting ${array[1]}" >> log.log
  SECONDS=0
  psiblast -query rawseq.txt \
           -num_threads ${numthreads} \
           -evalue <EVALUE> \
           -save_pssm_after_last_round \
           -db <BLAST_DB_PATH> \
           -num_iterations <NUM_ITERATIONS> \
           -out_ascii_pssm protein.pssm.${array[1]}
  echo "Finished: ${SECONDS}" >> log.log
  numthreads=<SUBSEQUENT_NUM_THREADS>
done < <INPUT_FILE_PREFIX>_split${SPLITID}

