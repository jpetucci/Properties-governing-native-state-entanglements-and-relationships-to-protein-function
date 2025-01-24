#!/usr/bin/env python3
"""
Generates a per-residue SASA feature DataFrame with a sliding window for any species.

Workflow:
1) Load an existing 'rawdata_df' pickle that contains at least:
   - geneid, pdbid, repchain
   - pdb_seq_converted (list of residues)
   - ent_seq (entanglement labels, if needed)
2) For each row, load a dictionary of SASA values from <geneid>_<pdbid>_<repchain>_sasa_dict.pkl
   in the specified directory (each dict has a key 'folded_sasa' → array or list).
3) Build a sliding window of size 'window_size' around each residue index (1..N).
4) For each window, gather SASA values from the local dictionary (one row per window).
5) Concatenate all rows from all sequences into a final feature DataFrame.
6) Save the resulting DataFrame as a pickle.

You can run this for any species by specifying:
    --input-rawdata  [pickle with rawdata_df]
    --sasa-dir       [directory containing SASA dicts]
    --output-file    [output pickle name]
    --window-size    [odd integer, default=9]
    --n-jobs         [parallel workers, default=4]

Ensure that:
- The 'rawdata_df' has columns [geneid, pdbid, repchain, pdb_seq_converted, ent_seq].
- The SASA dictionaries are named <geneid>_<pdbid>_<repchain>_sasa_dict.pkl,
  each containing {'folded_sasa': [...] } that maps 1-based residue indices to SASA values.
"""

import os
import sys
import math
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

# Basic dictionaries for residue codes (as in the older approach)
aalocal_dict = {
    'ALA': 'A','ARG': 'R','ASN': 'N','ASP': 'D','CYS': 'C','GLU': 'E','GLN': 'Q','GLY': 'G',
    'HIS': 'H','ILE': 'I','LEU': 'L','LYS': 'K','MET': 'M','PHE': 'F','PRO': 'P','SER': 'S',
    'THR': 'T','TRP': 'W','TYR': 'Y','VAL': 'V'
}

modification_dict = {
    'MSE': 'MET','LLP': 'LYS','MLY': 'LYS','CSO': 'CYS','KCX': 'LYS','CSS': 'CYS','OCS': 'CYS',
    'NEP': 'HIS','CME': 'CYS','SEC': 'CYS','CSX': 'CYS','CSD': 'CYS','SEB': 'SER','SEP': 'SER',
    'SMC': 'CYS','SNC': 'CYS','CAS': 'CYS','CAF': 'CYS','FME': 'MET','143': 'CYS','PTR': 'TYR',
    'MHO': 'MET','ALY': 'LYS','BFD': 'ASP','TPO': 'THR','DHA': 'SER','ALS': 'ALA','TPQ': 'TYR',
    'MLZ': 'LYS','TYS': 'TYR','FGP': 'SER','DDZ': 'ALA','T8L': 'THR','CSP': 'CYS','AME': 'MET',
    'PCA': 'GLN','SCY': 'CYS','YCM': 'CYS','LED': 'LEU','KPI': 'LYS','TYX': 'CYS','DSN': 'SER'
}

def convertAA(AA3):
    return aalocal_dict.get(AA3, AA3)

def convertAAmod(AA3):
    return modification_dict.get(AA3, AA3)

def slidingwindow(seq, window_size):
    """
    Return all contiguous sublists of length 'window_size' from 'seq'.
    """
    return [seq[i : i + window_size] for i in range(len(seq) - window_size + 1)]

def middle_element(input_list):
    """
    Return the middle element from a list with an odd length.
    """
    length = len(input_list)
    return input_list[(length - 1) // 2]

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(
        description="Build SASA features from per-residue dictionaries using a sliding window."
    )
    parser.add_argument("--input-rawdata", required=True,
                        help="Path to a pickle with 'rawdata_df' (columns: geneid, pdbid, repchain, pdb_seq_converted, ent_seq).")
    parser.add_argument("--sasa-dir", required=True,
                        help="Directory containing <geneid>_<pdbid>_<repchain>_sasa_dict.pkl files.")
    parser.add_argument("--output-file", default="sasafeat_df.pkl",
                        help="Output pickle file for the final SASA features DataFrame.")
    parser.add_argument("--window-size", type=int, default=9,
                        help="Sliding window size (must be an odd number). Default=9.")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Number of parallel workers for building features. Default=4.")

    args = parser.parse_args()

    if args.window_size % 2 == 0:
        sys.exit("Error: window-size must be an odd number.")

    # Load rawdata_df
    rawdata_df = pd.read_pickle(args.input_rawdata)
    print(f"Loaded rawdata_df with shape {rawdata_df.shape} from {args.input_rawdata}")

    # For each row, load the per-residue SASA from the specified directory
    sasa_df_list = []
    for idx, row in rawdata_df.iterrows():
        geneid = row["geneid"]
        pdbid  = row["pdbid"]
        chain  = row["repchain"]
        sasa_path = os.path.join(args.sasa_dir, f"{geneid}_{pdbid}_{chain}_sasa_dict.pkl")

        try:
            sasa_data = load_pickle(sasa_path)
        except FileNotFoundError:
            print(f"Warning: Missing SASA file for row {idx}: {sasa_path}")
            sasa_data = {"folded_sasa": []}

        # 'folded_sasa' should map an index -> SASA value
        folded_sasa = sasa_data.get("folded_sasa", [])
        # Convert to DataFrame
        df_sasa = pd.DataFrame(folded_sasa).T
        df_sasa.columns = ["sasa_val"]
        df_sasa.reset_index(drop=True, inplace=True)
        # Make the index from 1..n
        df_sasa.index = df_sasa.index + 1

        sasa_df_list.append(df_sasa)

    # Prepare sliding windows
    def build_indices(row):
        seq_len = len(row["pdb_seq_converted"])
        idx_list = list(range(1, seq_len + 1))
        return slidingwindow(idx_list, args.window_size)

    rawdata_df["window"] = rawdata_df.apply(build_indices, axis=1)
    # Also build 'window_seq' for reference if needed
    def build_window_seq(row):
        return [ row["pdb_seq_converted"][pos-1] for pos in [middle_element(w) for w in row["window"]] ]

    rawdata_df["window_seq"] = rawdata_df.apply(build_window_seq, axis=1)

    # Build the final SASA feature DataFrame
    def process_sasa_features(row_id):
        ent_seq = rawdata_df.iloc[row_id]["ent_seq"]
        seq_len = len(ent_seq)
        local_windows = rawdata_df.iloc[row_id]["window"]
        local_sasa    = sasa_df_list[row_id]
        features_list = []
        for w in local_windows:
            # w is a list of indices; get 'sasa_val' for each
            vals = local_sasa.loc[w, "sasa_val"].tolist()
            features_list.append(vals)
        return pd.DataFrame(features_list)

    print("Generating SASA feature rows in parallel...")
    results = Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(process_sasa_features)(i) for i in range(len(rawdata_df))
    )

    sasafeat_df = pd.concat(results, ignore_index=True).fillna(0)

    # Name columns (for window_size=9, we get: [sasa_m4, sasa_m3, sasa_m2, sasa_m1, sasa_0, sasa_p1, sasa_p2, sasa_p3, sasa_p4])
    half_w = args.window_size // 2
    col_names = []
    for i in range(-half_w, half_w+1):
        if i < 0:
            col_names.append(f"sasa_m{-i}")
        elif i == 0:
            col_names.append("sasa_0")
        else:
            col_names.append(f"sasa_p{i}")

    sasafeat_df.columns = col_names
    print(f"Final SASA feature DataFrame shape: {sasafeat_df.shape}")

    # Save to pickle
    sasafeat_df.to_pickle(args.output_file)
    print(f"Saved SASA features to: {args.output_file}")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())

