# Section 2

This section focuses on functional data generation and analysis. There are two steps:

- **step1**
  - **Code**: `generate_functional_data.py`
  - **Input**:  
    - `functional_data/` : Pre-collected functional annotations or data.  
    - `pdbs/` : PDB files for structures.  
    - `seq_ent_files/` : Sequence entanglement files from Section 1.  
  - **Output**: Processed functional data in `./output/`.

- **step2**
  - **Code**: `functional_sites.R`
  - **Input**: Results from step1 or any additional annotation files.
  - **Output**: Summaries, plots, or site-based data.

## Dependencies

- Python for step1 (`generate_functional_data.py`).
- R environment (including packages like `tidyverse`, `dplyr`, `ggplot2`, etc.) for step2.

## Usage

1. Ensure you have installed the required dependencies from `software_environment/`.
2. Run step1 to generate or merge functional data with your entanglement data.
3. Switch to step2 and run `functional_sites.R` for advanced analyses or visualizations.

---
