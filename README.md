# Non-Covalent Lasso Entanglements ML

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

This repository contains code and data (externally linked) to reproduce the analysis in "Properties governing native state entanglements and relationships to protein function"
 
Please refer to the paper for the exact details of the Methods and Supplementary Information for the data. 

Note that the code was run and optimized on the Linux based Penn State Roar Collab Research Computing System.

---

## Installation

Clone the project

```bash
git clone <insert repo link>
```

### Option 1: Apptainer/Singularity
A singularity/apptainer container and definition file is available that contains the full software environment to run the analysis code available in this repo.
* Download pre-built container <insert link> or build the container using the provided definition file
#### 
```bash 
sudo singularity build centos_r_container.sif singularity.def 
```

### Option 1: conda/mamba
* Create a conda environment using the provided environment file
```bash
conda env create --prefix <path to install location> --file <path to environment.yml>
```

---

## Usage

Each section below is independent and corresponds to the main results subsections from the accompanying manuscript. To run a given analysis section, shell into the provided/created 
container or activate the created conda environment and follow the steps below.

### Section 1 - Amino Acid, Polar/Hydrophobic Residues, Secondary Structure, and 3-letter PH Motif Enrichment Study

---

### Section 2 - Functional Region Enrichment Study

---

### Section 3 - Machine Learning Robust and Predictive Feature identification

---

### Section 4 - Controlling for Residue Burial with Propensity Score Matching

---

## Citation

If you found the code or data useful, please consider citing the paper: 

```bibtex 
@article {Non-covalent_lasso_entanglementsi_ML,
  author       = {Justin Petucci, Ian Sitarik, Yang Jiang, Viraj Rana, Hyebin Song, and Edward P. O'Brien},
  journal      = {},
  title        = {Properties governing native state entanglements and relationships to protein function},
  year         = {2025},
  doi          = {},
  URL          = {},
}
```

