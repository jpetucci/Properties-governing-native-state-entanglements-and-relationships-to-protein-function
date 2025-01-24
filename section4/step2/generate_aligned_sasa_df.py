#!/usr/bin/env python3
"""
Build a final DataFrame combining:
1) A list of valid genes (from a DataFrame or pickle) used to filter the raw entanglement data.
2) Raw entanglement data (with 'ent_seq'), applying modify_array_class4.
3) A features DataFrame (window-based numeric features).
4) A SASA features DataFrame.
5) A final target column mapped from <2 => 0, else => 1.

Steps:
-----
1) Load a 'valid-genes' pickle or CSV that provides a list of allowed geneid values.
2) Load a 'raw-ent-file' which contains 'ent_seq' and possibly columns geneid, pdbid, repchain.
   Apply modify_array_class4 to ent_seq. Filter to genes in valid-genes.
3) Build a target array (ent_seq) that aligns with the 9-residue window (or specified window).
   For each row, skip the first half-window + last half-window from the raw ent_seq.
   Flatten these partial sequences into one Series (datatargets_df).
4) Load the main feature DataFrame (features_file).
5) Load the SASA feature DataFrame (sasa_file).
6) Concatenate [features_df, sasa_df, datatargets_df], then add 'mapped_target' => 0/1 from the original <2 =>0 else =>1.
7) Output final combined DataFrame as a pickle.

Usage:
------
python combine_ent_sasa_feats.py \
  --valid-genes-file good_genes.pkl \
  --raw-ent-file raw_ent_data.pkl \
  --features-file main_features.pkl \
  --sasa-file sasa_feats.pkl \
  --output-file df_for_psm.pkl \
  --window-size 9

All paths can be adapted to any species or dataset. The code includes no hard-coded references.
"""

import os
import sys
import math
import pickle
import argparse
import numpy as np
import pandas as pd


def modify_array_class4(arr):
    """
    Replace all 4's with 0's. Then for each 3, label the 3 preceding and following
    elements as 4 (unless they're also 3). This matches the original logic for class 4.
    Finally the array is used to build a 2-class system: <2 => 0, else => 1
    (handled outside this function).
    """
    arr[arr == 4] = 0
    temp_arr = arr.copy()
    indices = np.where(temp_arr == 3)[0]
    for index in indices:
        start_index = max(0, index - 2)
        for i in range(start_index, index):
            if arr[i] != 3:
                arr[i] = 4
        end_index = min(len(arr), index + 3)
        for i in range(index + 1, end_index):
            if arr[i] != 3:
                arr[i] = 4
    return arr

def main():
    parser = argparse.ArgumentParser(
        description="Combine raw entanglement data, features, and SASA features into one DataFrame."
    )
    parser.add_argument("--valid-genes-file", required=True,
                        help="Path to a file containing a DataFrame with a 'geneid' column to keep.")
    parser.add_argument("--raw-ent-file", required=True,
                        help="Pickle file containing the raw ent data with columns [geneid, ent_seq, etc.].")
    parser.add_argument("--features-file", required=True,
                        help="Pickle file containing the main feature DataFrame.")
    parser.add_argument("--sasa-file", required=True,
                        help="Pickle file containing the SASA feature DataFrame.")
    parser.add_argument("--output-file", default="df_for_psm.pkl",
                        help="Output pickle file name (default=df_for_psm.pkl).")
    parser.add_argument("--window-size", type=int, default=9,
                        help="Window size used in the feature DF (default=9).")
    args = parser.parse_args()

    # 1) Load 'valid-genes-file'
    #    We expect a pickle or CSV with a DataFrame containing a 'geneid' column
    #    If it's a pickle, load it directly; if CSV, read_csv. We'll attempt pickle first, else CSV.
    try:
        df_valid = pd.read_pickle(args.valid_genes_file)
    except:
        df_valid = pd.read_csv(args.valid_genes_file)

    if "geneid" not in df_valid.columns:
        sys.exit("Error: valid-genes-file does not contain 'geneid' column.")
    goodgene_list = df_valid["geneid"].tolist()
    print(f"Loaded {len(goodgene_list)} valid genes from: {args.valid_genes_file}")

    # 2) Load 'raw-ent-file', apply modify_array_class4, filter to those genes
    raw_ent_df = pd.read_pickle(args.raw_ent_file)
    if "geneid" not in raw_ent_df.columns or "ent_seq" not in raw_ent_df.columns:
        sys.exit("Error: raw-ent-file must have columns 'geneid' and 'ent_seq' at least.")
    print(f"Loaded raw ent data with shape {raw_ent_df.shape} from {args.raw_ent_file}")

    # Modify ent_seq
    raw_ent_df["ent_seq"] = raw_ent_df["ent_seq"].apply(lambda x: modify_array_class4(x))
    # Filter
    raw_ent_df = raw_ent_df[raw_ent_df["geneid"].isin(goodgene_list)]
    raw_ent_df.reset_index(drop=True, inplace=True)
    print(f"After filtering valid genes, shape={raw_ent_df.shape}")

    # 3) Build target array for the 9-residue (or specified) window
    #    For each row's ent_seq, we skip the first half-window and the last half-window
    #    to align with the features and SASA data
    seq_start = args.window_size // 2  # e.g. 4 if window=9
    adjustedseq_list = []
    for idx, row in raw_ent_df.iterrows():
        full_arr = row["ent_seq"]
        seq_end = len(full_arr) - seq_start - 1
        if seq_end >= seq_start:
            # Build partial array
            partial_arr = [full_arr[x] for x in range(seq_start, seq_end + 1)]
        else:
            partial_arr = []
        adjustedseq_list.append(partial_arr)

    # Flatten
    datatargets = pd.DataFrame(pd.core.common.flatten(adjustedseq_list), columns=["target"])
    print(f"Created target vector of length {len(datatargets)}")

    # 4) Load main features and SASA features
    feats_df = pd.read_pickle(args.features_file)
    sasa_df  = pd.read_pickle(args.sasa_file)
    print(f"Main features shape: {feats_df.shape}")
    print(f"SASA features shape: {sasa_df.shape}")

    # 5) Concatenate [features_df, sasa_df, datatargets], row-wise
    combined_df = pd.concat([feats_df, sasa_df, datatargets], axis=1)
    # optionally sum all columns containing 'sasa'
    sasa_cols = [c for c in combined_df.columns if "sasa" in c]
    combined_df["sasa_windowsum"] = combined_df[sasa_cols].sum(axis=1)
    # mapped_target => 0 if <2, else 1
    combined_df["mapped_target"] = combined_df["target"].apply(lambda x: 0 if x < 2 else 1)

    print(f"Final combined shape: {combined_df.shape}")

    # 6) Save
    combined_df.to_pickle(args.output_file)
    print(f"Saved final DataFrame to: {args.output_file}")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())

