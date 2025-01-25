# Section 1

This section contains steps (1 through 7) that cover the initial processing and generation of core data. Below is a brief overview of what each step does and how they connect:

- **step1**  
  - _Sub-steps (`stepa`, `stepb`) each have scripts and notebooks for generating experimental entanglement lists, data exploration, etc._
  - **Input**: Provide raw data in `./input/`
  - **Code**: Scripts in `./code/`
  - **Output**: Processed results in `./output/`

- **step2**  
  - Contains `rep_seq_final.py` for representative sequence analysis.
  - **Input**: Directory for entanglement files, PDB data in `./input/ENTANGLEFILES` and `./input/PDBDIRECTORY`.
  - **Output**: Various processed files in `./output/`.

- **step3**  
  - Generates entanglement vectors (`generate_entanglement_vectors.py`).
  - **Input**: Clustered GE, experimental entanglement lists, PDB structures.
  - **Output**: Saved vector data in `./output/`.

- **step4**  
  - Uses `stride_pipeline.py` to extract secondary structure info.
  - **Input**: Entanglement lists, PDBs, etc.
  - **Output**: Stride-based analysis and intermediate files.

- **step5**  
  - Script `sequence_entanglement_data_final.py` to combine sequence-based and entanglement data.
  - **Input**: Clustered GE, entanglement lists, stride files, etc.
  - **Output**: Results at `./output/`.

- **step6**  
  - Enrichment study (`Enrichment_study.py`) to analyze entanglement features.
  - **Input**: Various data from previous steps.
  - **Output**: Summaries/plots in `./output/`.

- **step7**  
  - Motif enrichment analysis (`motif_enrichment.py`).
  - **Input**: Data from previous steps.
  - **Output**: Final results in `./output/`.

## Dependencies

Most scripts are written in Python, with additional libraries as needed (e.g., `pandas`, `numpy`, `biopython`, etc.). Check the `software_environment/` directory for environment files or containers.

## Usage

1. Navigate into each step's `code` directory.
2. Review or edit any configuration, paths, or parameters as needed.
3. Run the Python scripts. Example:

   ```bash
   cd step1/stepa/code
   python generate_exp_ent_list_final.py
