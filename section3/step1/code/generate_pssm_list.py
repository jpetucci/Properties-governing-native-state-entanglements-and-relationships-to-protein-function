#!/usr/bin/env python3
"""
A refactored script that:
1) Reads an input CSV (columns: geneid, pdbid, chainid).
2) Loads PDB sequences from .npz files (pdbseqdir).
3) Converts 3-letter codes to 1-letter codes, applying modification lookups.
4) Filters out invalid or unknown codes (UNK, TYX).
5) Creates a final `rawdata_df` with flattened sequences in two styles:
    - pdb_seq_converted_flat: using convertAA/convertAAmod logic
    - pdb_seq_flat: direct use of `three_to_one_dict`

Optionally:
    - Exports (geneid, pdb_seq_flat) for downstream tools (e.g., PSSM).

Usage Example:
    python script_name.py \
      --input-file /path/to/entanglement_list.csv \
      --pdb-seq-dir /path/to/npz_dir \
      --pssm-output my_output.csv
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

# --------------------------------------------------------------------
#                         Global Dictionaries
# --------------------------------------------------------------------

# Maps canonical 3-letter codes to single-letter
aalocal_dict = {
    'ALA': 'A','ARG': 'R','ASN': 'N','ASP': 'D','CYS': 'C','GLU': 'E','GLN': 'Q','GLY': 'G',
    'HIS': 'H','ILE': 'I','LEU': 'L','LYS': 'K','MET': 'M','PHE': 'F','PRO': 'P','SER': 'S',
    'THR': 'T','TRP': 'W','TYR': 'Y','VAL': 'V'
}

# Maps modified codes to a standard parent code
modification_dict = {
    'MSE': 'MET','LLP': 'LYS','MLY': 'LYS','CSO': 'CYS','KCX': 'LYS','CSS': 'CYS','OCS': 'CYS',
    'NEP': 'HIS','CME': 'CYS','SEC': 'CYS','CSX': 'CYS','CSD': 'CYS','SEB': 'SER','SEP': 'SER',
    'SMC': 'CYS','SNC': 'CYS','CAS': 'CYS','CAF': 'CYS','FME': 'MET','143': 'CYS','PTR': 'TYR',
    'MHO': 'MET','ALY': 'LYS','BFD': 'ASP','TPO': 'THR','DHA': 'SER','CSP': 'CYS','AME': 'MET',
    'YCM': 'CYS','T8L': 'THR','TPQ': 'TYR','SCY': 'CYS','MLZ': 'LYS','TYS': 'TYR','SCS': 'CYS',
    'LED': 'LEU','KPI': 'LYS','PCA': 'GLN','DSN': 'SER'
}

# Direct 3-letter → 1-letter mapping (including some modified residues)
three_to_one_dict = {
    'ALA': 'A','ARG': 'R','ASN': 'N','ASP': 'D','CYS': 'C','GLU': 'E','GLN': 'Q','GLY': 'G',
    'HIS': 'H','ILE': 'I','LEU': 'L','LYS': 'K','MET': 'M','PHE': 'F','PRO': 'P','SER': 'S',
    'THR': 'T','TRP': 'W','TYR': 'Y','VAL': 'V','MSE': 'M','LLP': 'K','MLY': 'K','CSO': 'C',
    'KCX': 'K','CSS': 'C','OCS': 'C','NEP': 'H','CME': 'C','SEC': 'U','CSX': 'C','CSD': 'C',
    'SEB': 'S','SEP': 'S','SMC': 'C','SNC': 'C','CAS': 'C','CAF': 'C','FME': 'M','143': 'C',
    'PTR': 'Y','MHO': 'M','ALY': 'K','BFD': 'D','TPO': 'T','DHA': 'S','CSP': 'C','AME': 'M',
    'YCM': 'C','T8L': 'T','TPQ': 'Y','SCY': 'C','MLZ': 'K','TYS': 'Y','SCS': 'C','LED': 'L',
    'KPI': 'K','PCA': 'Q','DSN': 'S'
}


# --------------------------------------------------------------------
#                         Helper Functions
# --------------------------------------------------------------------
def convertAA(AA3):
    """
    Map a canonical 3-letter code to single-letter code via `aalocal_dict`.
    If not found, return the original code.
    """
    return aalocal_dict.get(AA3, AA3)

def convertAAmod(AA3):
    """
    Replace a modified residue with its standard parent code (if found),
    else return the original code.
    """
    return modification_dict.get(AA3, AA3)

def singlefeat(item_list):
    """
    Join a list of single characters into a single string.
    """
    return ''.join(item_list)


# --------------------------------------------------------------------
#                         Main Pipeline
# --------------------------------------------------------------------
def main():
    parser = ArgumentParser(
        description=(
            "Reads gene/pdb/chain data, loads each .npz file of 3-letter PDB sequences, "
            "converts them to 1-letter codes (handling modifications), filters out unknowns, "
            "and prepares a final DataFrame. Optionally writes out a CSV for PSSM usage."
        )
    )
    parser.add_argument("--input-file", required=True,
                        help="Path to space-delimited file with columns: geneid, pdbid, repchain.")
    parser.add_argument("--pdb-seq-dir", required=True,
                        help="Directory containing .npz files named geneid_pdbid_chain.npz.")
    parser.add_argument("--pssm-output", default=None,
                        help="Optional CSV to which we export (geneid, pdb_seq_flat).")
    args = parser.parse_args()

    inputfile = args.input_file
    pdbseqdir = args.pdb_seq_dir

    # 1) Read the input CSV (space-delimited).
    rawdata_df = pd.read_csv(inputfile, sep=r'\s+', names=['geneid', 'pdbid', 'repchain'])
    print(f"Loaded {len(rawdata_df)} rows from {inputfile}.")

    # 2) Load the PDB sequences from .npz files.
    pdb_seq_list = []
    missing_count = 0
    for idx in tqdm(range(len(rawdata_df)), desc="Loading PDB Sequences"):
        gene   = rawdata_df.iloc[idx].geneid
        pdb_id = rawdata_df.iloc[idx].pdbid
        chain  = rawdata_df.iloc[idx].repchain
        npz_path = os.path.join(pdbseqdir, f"{gene}_{pdb_id}_{chain}.npz")
        try:
            arr = np.load(npz_path, allow_pickle=True)["arr_0"]
            pdb_seq_list.append(arr)
        except FileNotFoundError:
            print(f"Warning: Missing data for row={idx} => {npz_path}")
            pdb_seq_list.append([])
            missing_count += 1

    rawdata_df['pdb_seq'] = pdb_seq_list
    if missing_count > 0:
        print(f"Warning: {missing_count} missing .npz files. Those rows have empty sequences.")

    # 3) Convert from 3-letter to 1-letter (handle modifications).
    def convert_sequence(seq_3letter):
        return [convertAA(convertAAmod(x)) for x in seq_3letter]

    rawdata_df['pdb_seq_converted'] = rawdata_df['pdb_seq'].apply(convert_sequence)

    # 4) Drop sequences containing 'UNK' or 'TYX'.
    mask_unk = rawdata_df['pdb_seq'].apply(lambda x: 'UNK' in x)
    mask_tyx = rawdata_df['pdb_seq'].apply(lambda x: 'TYX' in x)
    n_unk = mask_unk.sum()
    n_tyx = mask_tyx.sum()
    if n_unk or n_tyx:
        print(f"Dropping {n_unk} rows with 'UNK' and {n_tyx} rows with 'TYX'.")
    rawdata_df = rawdata_df[~mask_unk & ~mask_tyx].reset_index(drop=True)

    # 5) Flatten sequences two ways:
    #    (A) using the newly converted list
    #    (B) direct dictionary from original 3-letter entries
    def flatten_to_1letter(seq_list):
        return singlefeat([three_to_one_dict.get(x, x) for x in seq_list])

    rawdata_df['pdb_seq_converted_flat'] = rawdata_df['pdb_seq_converted'].apply(singlefeat)
    rawdata_df['pdb_seq_flat']           = rawdata_df['pdb_seq'].apply(flatten_to_1letter)

    print(f"Final shape after filtering: {rawdata_df.shape}")

    # 6) Optionally export (geneid, pdb_seq_flat) to a CSV
    if args.pssm_output:
        print(f"Exporting (geneid, pdb_seq_flat) to {args.pssm_output} ...")
        rawdata_df[['geneid', 'pdb_seq_flat']].to_csv(args.pssm_output, index=False)

    print("Done! Use `rawdata_df` in memory or the optional CSV output for downstream steps.")


# --------------------------------------------------------------------
#                           Entry Point
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()

"""
---------------------------------------------------------------------
Summary:
---------------------------------------------------------------------
1. Provide --input-file to specify the CSV with (geneid, pdbid, repchain).
2. Provide --pdb-seq-dir to specify the folder containing .npz sequences.
3. (Optional) --pssm-output to store a two-column CSV (geneid, pdb_seq_flat).

Example Command:
  python script_name.py \
      --input-file /path/to/entanglement_list.csv \
      --pdb-seq-dir /path/to/npz_dir \
      --pssm-output my_output.csv

Result:
  A DataFrame `rawdata_df` with columns:
   [geneid, pdbid, repchain, pdb_seq, pdb_seq_converted, pdb_seq_converted_flat, pdb_seq_flat]
  Rows with "UNK" or "TYX" are dropped. If --pssm-output is provided, a small CSV with:
   [geneid, pdb_seq_flat]
  is saved for further analysis.
---------------------------------------------------------------------
"""

