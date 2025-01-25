# Section 4

This section involves calculating solvent-accessible surface area (SASA) values, aligning them with previously generated features, and matching them to additional data (like PSM or other structural metrics).

### Steps Overview

- **step1**
  - **Code**: `generate_sasa_values.py`
  - **Input**: 
    - `entanglement_lists/`
    - `pdbseqdir/`
    - `pdbs/`
    - `sasa_data/`
    - `ML_features/`
  - **Output**: SASA calculation results in `./output/`

- **step2**
  - **Code**: `generate_aligned_sasa_df.py`
  - **Input**: SASA outputs from step1, plus any relevant alignment data
  - **Output**: Merged or aligned dataframe with SASA info

- **step3**
  - **Code**: `perform_sasa_psm_matching.py`
  - **Input**: Aligned SASA data, psm data
  - **Output**: `psm_matching_results.txt` or other final combined dataset

## Dependencies

- Python libraries that compute SASA (e.g., `biopython`, `mdtraj`, or a custom algorithm).
- Possibly additional command-line tools for structural analysis.

## Usage

1. Run `generate_sasa_values.py` in `step1/code/` to obtain SASA data.
2. Use `generate_aligned_sasa_df.py` in `step2/code/` to merge or align these results with existing features.
3. Perform PSM matching or other final processing in step3 (`perform_sasa_psm_matching.py`).

---

