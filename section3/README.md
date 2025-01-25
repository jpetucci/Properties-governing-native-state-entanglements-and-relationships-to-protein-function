# Section 3

In Section 3, you generate advanced features (PSSM, B-factors, Coordination Number, and more) and then apply various machine learning or statistical models to your data.

### Overview of Steps

- **step1**  
  - **Code**: `generate_pssm_list.py`, `helper_pssm_tasks.py`
  - **Input**: `entanglement_lists/` and `pdb_seqs/`
  - **Output**: PSSM data in `./output/`

- **step2**  
  - **Code**: `generate_bfactors.py`
  - **Input**: Entanglement list, PDBs
  - **Output**: B-factor results in `./output/`

- **step3**  
  - **Code**: `generate_CN.py`
  - **Input**: Entanglement lists, PDB structures, stride data
  - **Output**: Coordination Number (CN) data in `./output/`

- **step4**  
  - **Code**: `generate_full_features.py`
  - **Input**: Consolidates bfactor_data, CN_data, pssms_data, stride_data, etc.
  - **Output**: A combined or merged feature dataset in `./output/`

- **step5**  
  - **Code**: R scripts (`L1logistic_ss.R`, `plot_ss.R`, etc.)
  - **Input**: The merged features from step4 plus any species-specific data
  - **Output**: Model results and figures for logistic regression or statistical modeling

- **step6**  
  - **Code**: `ML_models.py`
  - **Input**: The final features or subsets from step5
  - **Output**: Machine learning predictions, metrics, or classification results

## Dependencies

- Python 3 with libraries like `biopython`, `numpy`, `pandas`, `scipy`, `sklearn`, etc.
- R (for steps 5) with libraries for statistical modeling and data visualization.

## Usage

1. Generate feature data sequentially (steps1 → steps2 → steps3 → steps4).
2. Perform statistical modeling or machine learning in steps5 and steps6.
3. Refer to each step's `code` directory for specific instructions or parameter settings.

---

