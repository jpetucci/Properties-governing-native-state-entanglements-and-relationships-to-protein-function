# Software Environment

This directory contains files and configurations to help set up a consistent environment for running all scripts in the repository.

- **environment.yml**  
  - Conda environment file listing the required Python packages and their versions.

- **singularity.def**  
  - A Singularity definition file for building a container with all required dependencies.

## Usage

### Using Conda (environment.yml)

```bash
conda env create -f environment.yml
conda activate <env_name>
```

### Using Singularity

```bash
sudo singularity build myenv.sif singularity.def
singularity exec myenv.sif python myscript.py
```
