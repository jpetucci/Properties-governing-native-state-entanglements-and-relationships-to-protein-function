#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
import statistics
import random
from collections import defaultdict
from functools import reduce
import scipy
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

#############################
# Global Dictionaries       #
#############################

protein_seq_alpha = [
    'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'
]

reduced_aa_dict = {
    'A':'H','C':'P','D':'C','E':'C','F':'H','G':'H','H':'P','I':'H',
    'K':'C','L':'H','M':'A','N':'P','P':'H','Q':'P','R':'C','S':'P',
    'T':'P','V':'H','W':'A','Y':'A'
}

twoletter_dict = {
    'A':'H',
    'H':'H',
    'P':'P',
    'C':'P'
}

#############################
# Helper Functions          #
#############################

def mapto_4letter_alphabet(inputsequence_list):
    """
    Maps an input sequence (single-letter AAs) to a 4-letter alphabet via reduced_aa_dict.
    """
    return "".join(reduced_aa_dict[x] for x in inputsequence_list)

def mapto_2letter_alphabet(inputsequence_list):
    """
    Maps an input sequence (single-letter AAs) to a 2-letter alphabet.
    Uses reduced_aa_dict first, then twoletter_dict.
    """
    return "".join(twoletter_dict[reduced_aa_dict[x]] for x in inputsequence_list)

def slidingwindow(inputstr, window_size):
    """
    Returns a list of consecutive substrings of length 'window_size'.
    Example: slidingwindow('ABCDE', 3) -> ['ABC','BCD','CDE']
    """
    return [inputstr[i:i+window_size] for i in range(len(inputstr) - (window_size - 1))]

def sum_all_but_key(dictin, notkey):
    """
    Sums the dictionary values, excluding 'notkey' (or list of keys).
    """
    if not isinstance(notkey, list):
        notkey = [notkey]
    return sum([value for key, value in dictin.items() if key not in notkey])

def insidebuffer(buffersize, inputlist, criticalvalue):
    """
    Checks if 'criticalvalue' appears in 'inputlist' within 'buffersize' positions of the middle element.
    The list length must be odd. 'middleelement' = (len-1)//2.
    """
    listlength = len(inputlist)
    middleelementid = (listlength - 1) // 2
    start = max(0, middleelementid - buffersize)
    end = min(listlength, middleelementid + buffersize + 1)
    return criticalvalue in inputlist[start:end]

def find_misaligned2(inputseq, bufferregion, criticalvalue):
    """
    For each sublist in 'inputseq' (which is a list-of-lists or list-of-arrays),
    checks if 'criticalvalue' is in the middle region (± bufferregion).
    If not, filters out that sublist index.
    Returns the indices that PASS the filter.
    """
    return_list = []
    for sublist_id, sublist in enumerate(inputseq):
        if criticalvalue in sublist:
            if insidebuffer(bufferregion, sublist, criticalvalue):
                return_list.append(sublist_id)
            # else sublist is dropped
        else:
            # If sublist doesn't have criticalvalue at all, we keep it
            # (matching your logic from the notebook)
            return_list.append(sublist_id)
    return return_list

#############################
# Main Analysis Logic       #
#############################

def run_motif_analysis(
    rawdata_df,
    alphabet_size_col="pdb_seq_converted_2letter_flat",
    windowcolumnname="2letter_windows",
    window_size=3,
    buffersize=0,
    criticalval=3,
    output_csv="motif_results.csv"
):
    """
    Performs the motif analysis as shown in the original notebook steps:
      1) Create window columns for the chosen alphabet and the ent_seq
      2) Filter windows to keep only those that pass 'find_misaligned2'
      3) Convert ent_window to middle-element-based 0 or 3
      4) Build count_df, do chi2/fisher, apply FDR
      5) Save final DataFrame to output_csv
    """

    # 1) Add window columns to rawdata_df
    rawdata_df[windowcolumnname] = rawdata_df[alphabet_size_col].apply(
        lambda x: slidingwindow(x, window_size)
    )
    rawdata_df["ent_window"] = rawdata_df["ent_seq"].apply(
        lambda x: slidingwindow(x, window_size)
    )

    # 2) Filter out windows that do NOT have CR in buffer region (or do not pass logic)
    rawdata_df["select_elements"] = rawdata_df["ent_window"].apply(
        lambda x: find_misaligned2(x, buffersize, criticalval)
    )

    def _filter_windows(row):
        indices = row["select_elements"]
        new_ent_window = [row["ent_window"][i] for i in indices]
        new_alphabet_windows = [row[windowcolumnname][i] for i in indices]
        return new_ent_window, new_alphabet_windows

    # apply row-wise
    rawdata_df["temp_tuple"] = rawdata_df.apply(_filter_windows, axis=1)
    rawdata_df["ent_window"] = rawdata_df["temp_tuple"].apply(lambda x: x[0])
    rawdata_df[windowcolumnname] = rawdata_df["temp_tuple"].apply(lambda x: x[1])
    rawdata_df.drop(columns=["temp_tuple"], inplace=True)

    # 3) Convert each (window of ent_seq) to a single integer (0 or 3)
    #    if 'criticalval' is in the buffer region => 3, else 0
    def _convert_ent_window(ent_window_list):
        # ent_window_list is a list of sublists
        # each sublist is length = window_size
        # insidebuffer(...) => True => 3, else 0
        return [
            3 if insidebuffer(buffersize, sublist, criticalval) else 0
            for sublist in ent_window_list
        ]

    rawdata_df["ent_window"] = rawdata_df["ent_window"].apply(_convert_ent_window)

    # 4) Build count_df
    ent_values = list(pd.core.common.flatten(rawdata_df["ent_window"].values))
    motif_values = list(pd.core.common.flatten(rawdata_df[windowcolumnname].values))

    count_df = pd.DataFrame({
        "ent_val": ent_values,
        "Motif":   motif_values
    })

    dict_c0 = count_df[count_df.ent_val == 0]["Motif"].value_counts().to_dict()
    dict_c3 = count_df[count_df.ent_val == 3]["Motif"].value_counts().to_dict()

    # Convert to defaultdict so missing keys => 0
    dict_c0 = defaultdict(lambda: 0, dict_c0)
    dict_c3 = defaultdict(lambda: 0, dict_c3)

    # Perform chi2 & fisher tests
    c1_list = []
    unique_motifs = count_df[count_df.ent_val == 3]["Motif"].unique().tolist()

    for motif in unique_motifs:
        tab = np.array([
            [dict_c3[motif], sum_all_but_key(dict_c3, motif)],
            [dict_c0[motif], sum_all_but_key(dict_c0, motif)]
        ])

        # G-test (aka log-likelihood ratio) via chi2_contingency
        chi_val, p_val, dof, exp_val = chi2_contingency(
            tab, correction=False, lambda_="log-likelihood"
        )
        # fisher
        odds_ratio, fisher_p = fisher_exact(tab)

        # ratio of proportions
        # classp = # in c3 / total c3; nonclassp = # in c3-other / total c3-other
        classp = round(tab[0][0] / tab[1][0], 6) if tab[1][0] != 0 else 0
        nonclassp = round(tab[0][1] / tab[1][1], 6) if tab[1][1] != 0 else 0
        ratio = np.inf if nonclassp == 0 else (classp / nonclassp)

        c1_list.append([motif, p_val, fisher_p, ratio,
                        tab[0][0], tab[0][1], tab[1][0], tab[1][1]])

    class1_results_df = pd.DataFrame(
        c1_list,
        columns=[
            "AAtype","gtest_pval","fisher_pval","oddsratio",
            "CR Motif","CR !Motif","!CR Motif","!CR !Motif"
        ]
    )

    # 5) FDR correction
    if not class1_results_df.empty:
        class1_results_df["bh_gtest_pval"] = multipletests(
            class1_results_df["gtest_pval"].values, method="fdr_bh"
        )[1]
        class1_results_df["bh_fisher_pval"] = multipletests(
            class1_results_df["fisher_pval"].values, method="fdr_bh"
        )[1]

    # Filter final results
    class1_results_df = class1_results_df[class1_results_df["bh_fisher_pval"] < 0.05]
    class1_results_df = class1_results_df.sort_values(by=["oddsratio"], ascending=False)

    # Optionally filter out motifs with < 200 total occurrences
    # (matching your notebook)
    class1_results_df["MotifCount"] = class1_results_df["CR Motif"] + class1_results_df["!CR Motif"]
    class1_results_df = class1_results_df[class1_results_df["MotifCount"] > 200].reset_index(drop=True)

    # Save
    class1_results_df.to_csv(output_csv, index=False)
    return class1_results_df

#############################
# CLI Main                  #
#############################

def main():
    parser = argparse.ArgumentParser(description="Motif Analysis Script")
    parser.add_argument("--pkl-file", required=True, help="Path to the rawdata_df pickle.")
    parser.add_argument("--alphabet", default="2letter", choices=["2letter","4letter"], 
                        help="Whether to use the 2-letter or 4-letter mapping.")
    parser.add_argument("--window-size", type=int, default=3, help="Size of the sliding window.")
    parser.add_argument("--buffer-size", type=int, default=0, help="Buffer region around the middle.")
    parser.add_argument("--critical-val", type=int, default=3, help="Value designating 'entangled' in ent_seq.")
    parser.add_argument("--output-csv", default="motif_results.csv", help="Where to save the final DataFrame.")
    args = parser.parse_args()

    # Load the DataFrame
    if not os.path.exists(args.pkl_file):
        print(f"Error: {args.pkl_file} not found.")
        sys.exit(1)
    rawdata_df = pd.read_pickle(args.pkl_file)

    # Ensure we have the needed columns
    # 'pdb_seq_converted' must exist
    if "pdb_seq_converted" not in rawdata_df.columns:
        print("Error: rawdata_df is missing 'pdb_seq_converted' column.")
        sys.exit(1)
    if "ent_seq" not in rawdata_df.columns:
        print("Error: rawdata_df is missing 'ent_seq' column.")
        sys.exit(1)

    # If user selected 4letter, create 'pdb_seq_converted_4letter_flat' if not present
    if args.alphabet == "4letter":
        if "pdb_seq_converted_4letter_flat" not in rawdata_df.columns:
            rawdata_df["pdb_seq_converted_4letter_flat"] = rawdata_df["pdb_seq_converted"].apply(mapto_4letter_alphabet)
        alphabet_col = "pdb_seq_converted_4letter_flat"
        window_col   = "4letter_windows"
    else:
        # 2letter
        if "pdb_seq_converted_2letter_flat" not in rawdata_df.columns:
            rawdata_df["pdb_seq_converted_2letter_flat"] = rawdata_df["pdb_seq_converted"].apply(mapto_2letter_alphabet)
        alphabet_col = "pdb_seq_converted_2letter_flat"
        window_col   = "2letter_windows"

    # Perform the motif analysis
    final_df = run_motif_analysis(
        rawdata_df,
        alphabet_size_col=alphabet_col,
        windowcolumnname=window_col,
        window_size=args.window_size,
        buffersize=args.buffer_size,
        criticalval=args.critical_val,
        output_csv=args.output_csv
    )

    print(f"Motif analysis complete. Results saved to {args.output_csv}")
    print(final_df.head(10))

if __name__ == "__main__":
    main()

