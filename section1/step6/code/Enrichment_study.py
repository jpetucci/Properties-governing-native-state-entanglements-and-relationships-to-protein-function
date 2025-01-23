#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import reduce
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency, fisher_exact
import tqdm

#############################
#  Global Dictionaries etc. #
#############################

SS_reduction_dict = {
    'AlphaHelix': 'Helix',
    'Strand': 'Strand',
    'Turn': 'Coil',
    'Coil': 'Coil',
    '310Helix': 'Helix',
    'Bridge': 'Strand',
    'PiHelix': 'Helix',
}

#############################
#  Helper Functions         #
#############################

def sum_all_but_key(dictin, notkey):
    """
    Sum the values of a dictionary excluding specified keys.
    """
    if not isinstance(notkey, list):
        notkey = [notkey]
    return sum(value for key, value in dictin.items() if key not in notkey)

def modify_array_class4(arr):
    """
    - Replace all 4's with 0's.
    - For each element == 3, set the previous two and next two elements to 4 (if not already 3).
    """
    arr = arr.copy()
    arr[arr == 4] = 0
    indices = np.where(arr == 3)[0]
    for index in indices:
        # previous two
        start_index = max(0, index - 2)
        for i in range(start_index, index):
            if arr[i] != 3:
                arr[i] = 4
        # next two
        end_index = min(len(arr), index + 3)
        for i in range(index + 1, end_index):
            if arr[i] != 3:
                arr[i] = 4
    return arr

def generate_count_dict(count_df):
    """
    Returns a nested dict: [ent_class][geneid][AA_type] = count
    """
    from collections import defaultdict
    count_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    grouped = count_df.groupby(['ent_class', 'geneid'])['AA_type'].value_counts()
    for (ent_class, geneid, AA_type), count in grouped.items():
        count_dict[ent_class][geneid][AA_type] = count

    # Force all combinations to exist, so we avoid key errors later
    for ent_class in count_df['ent_class'].unique():
        for geneid in count_df['geneid'].unique():
            _ = count_dict[ent_class][geneid]
    return count_dict

def get_class_counts(count_df, ent_class):
    """
    Get total counts of 'AA_type' for a specified ent_class in the entire dataset.
    Returns a defaultdict(int).
    """
    counts = count_df[count_df.ent_class == ent_class]['AA_type'].value_counts().to_dict()
    return defaultdict(lambda: 0, counts)

def compute_class_results(count_df, count_dict, dict_c0, compareclass, dict_selection=None):
    """
    Compare the specified ent_class vs class 0 using:
      - Fisher’s exact test
      - Chi2 contigency
      - CMH test (via StratifiedTable from statsmodels)
      - FDR correction (Benjamini-Hochberg) on Fisher p-values
    """
    if dict_selection is None:
        dict_selection = get_class_counts(count_df, compareclass)

    result_list = []
    AA_types = count_df['AA_type'].unique().tolist()

    for AAname in AA_types:
        # contingency table
        tab = np.array([
            [dict_selection[AAname], sum_all_but_key(dict_selection, AAname)],
            [dict_c0[AAname],       sum_all_but_key(dict_c0, AAname)]
        ])

        if tab[1][0] == 0:
            continue

        # Chi2 test
        try:
            chi_val, p_val, dof, exp_val = chi2_contingency(tab, correction=False, lambda_="log-likelihood")
        except Exception:
            # if some error occurs (like zero counts), skip
            continue

        # Fisher test
        odds_ratio, fisher_p = fisher_exact(tab)

        # ratio of proportions
        class_p = round(tab[0][0] / tab[1][0], 6)   # proportion for compareclass
        nonclass_p = round(tab[0][1] / tab[1][1], 6) if tab[1][1] != 0 else 0

        # Build per-gene contingency tables
        tab_list = []
        for geneid in count_df['geneid'].unique():
            arr_00 = count_dict[compareclass][geneid][AAname]
            arr_01 = sum_all_but_key(count_dict[compareclass][geneid], AAname)
            arr_10 = count_dict[0][geneid][AAname]
            arr_11 = sum_all_but_key(count_dict[0][geneid], AAname)
            tab_list.append([[arr_00, arr_01],[arr_10, arr_11]])

        # CMH test via statsmodels
        st = sm.stats.StratifiedTable(tab_list)
        st_summary_data = st.summary().data  # summary in tabular form

        # By observation in the printed summary table:
        # st_summary_data[6][2] => CMH p-value
        # st_summary_data[1][1] => pooled odds ratio
        st_pval = st_summary_data[6][2]
        st_odds = st_summary_data[1][1]

        or_marg = class_p / nonclass_p if nonclass_p != 0 else np.inf

        result_list.append([
            AAname,
            fisher_p,
            or_marg,
            st_pval,
            st_odds,
            tab[0][0],
            tab[1][0],
            tab[0][1],
            tab[1][1]
        ])

    results_df = pd.DataFrame(result_list,
        columns=[
            'AA_type',
            f'fe_pval_{compareclass}',
            f'OR_marg_{compareclass}',
            f'CMH_pval_{compareclass}',
            f'OR_pool_{compareclass}',
            f'AA_c{compareclass}',
            'AA_c0',
            f'!AA_c{compareclass}',
            '!AA_c0'
        ]
    )

    # FDR (BH) on Fisher p-values
    if not results_df.empty:
        bh_pvals = multipletests(results_df[f'fe_pval_{compareclass}'].values, method='fdr_bh')[1]
        results_df[f'bh_fe_pval_{compareclass}'] = bh_pvals
        results_df = results_df[results_df[f'bh_fe_pval_{compareclass}'] < 1.05]
        results_df = results_df.sort_values(by=[f'OR_marg_{compareclass}'])
        results_df.reset_index(drop=True, inplace=True)

    # Final selected columns
    keep_cols = [
        'AA_type',
        f'OR_marg_{compareclass}',
        f'OR_pool_{compareclass}',
        f'CMH_pval_{compareclass}',
        f'bh_fe_pval_{compareclass}',
        f'AA_c{compareclass}',
        'AA_c0',
        f'!AA_c{compareclass}',
        '!AA_c0'
    ]
    final_results = results_df[keep_cols] if not results_df.empty else pd.DataFrame(columns=keep_cols)
    return final_results

def compute_stats_for_class(
    rawdata_df,
    compareclass,
    species_name,
    analysis_type,
    legacy_data=True,
    dict_selection=None
):
    """
    - Optionally apply class4 modification to ent_seq if legacy_data=True.
    - Flatten ent_seq, gene IDs, and (depending on analysis_type) use different 'AA_type' data.
    - Build count_df, run statistical comparisons against class 0.
    """
    # Possibly modify ent_seq
    if legacy_data:
        rawdata_df['ent_seq'] = rawdata_df['ent_seq'].apply(modify_array_class4)

    # Flatten out ent_seq (the "ent_class" array) and gene IDs
    rawdata_df['geneid_seq'] = rawdata_df.apply(lambda x: [x.geneid for _ in x.ent_seq], axis=1)

    ent_class_flat = [item for sublist in rawdata_df['ent_seq'].values for item in sublist]
    geneid_flat = [item for sublist in rawdata_df['geneid_seq'].values for item in sublist]

    # Decide how to flatten the corresponding "AA_type" or "PH" or "SS" sequence
    if analysis_type == 'AA':
        # rawdata_df['pdb_seq'] is a list of single-letter AA codes
        AA_type_flat = [item for sublist in rawdata_df['pdb_seq'].values for item in sublist]
    elif analysis_type == 'PH':
        # rawdata_df['pdb_seq_converted_2letter_flat'] is a single string per row => list(list_of_chars)
        # We'll expand that back out
        # e.g. "HPHHH" => ['H','P','H','H','H']
        expanded_ph = rawdata_df['pdb_seq_converted_2letter_flat'].apply(list).values
        AA_type_flat = [aa for sublist in expanded_ph for aa in sublist]
    elif analysis_type == 'SS':
        # rawdata_df['secondary_structure'] is presumably a list of SS codes (like 'AlphaHelix')
        # We'll reduce them, e.g. 'AlphaHelix' -> 'Helix'
        # Then flatten
        # But let's do the reduction after the flatten for clarity
        all_ss = [item for sublist in rawdata_df['secondary_structure'].values for item in sublist]
        # apply SS_reduction_dict
        AA_type_flat = [SS_reduction_dict[ss] for ss in all_ss]
    else:
        raise ValueError(f"Unsupported analysis type {analysis_type}.")

    count_df = pd.DataFrame({
        'ent_class': ent_class_flat,
        'geneid':    geneid_flat,
        'AA_type':   AA_type_flat
    })

    # Save an intermediate CSV if you like
    count_df.to_csv(f"countdf_{species_name}_{analysis_type}.csv", index=False)

    # Build the nested count dictionary
    count_dict = generate_count_dict(count_df)
    dict_c0 = get_class_counts(count_df, 0)

    # If we didn't specify a selection dictionary, compute it
    if dict_selection is None:
        dict_selection = get_class_counts(count_df, compareclass)

    # Now compute stats
    results_df = compute_class_results(count_df, count_dict, dict_c0, compareclass, dict_selection)
    return results_df

def run_full_analysis(rawdata_df, species_name, legacy_data=True):
    """
    Runs the same code you had for each species, across analysis types ['AA','PH','SS'] and compare classes [1,2,3,4].
    Merges each set of 4 results. Saves final DataFrame to CSV/PKL.
    """
    analysis_types = ['AA', 'PH', 'SS']
    for analysis_type in analysis_types:
        result_df_list = []
        for compareclass in tqdm.tqdm([1,2,3,4], desc=f"Analysis {analysis_type} - {species_name}"):
            # Compute stats for this compareclass
            results_df = compute_stats_for_class(
                rawdata_df,
                compareclass,
                species_name,
                analysis_type,
                legacy_data=legacy_data
            )
            result_df_list.append(results_df)

        if len(result_df_list) == 0:
            print(f"No results for species={species_name} analysis={analysis_type}")
            continue

        # Merge the 4 DataFrames on 'AA_type'
        final_df = reduce(lambda x,y: pd.merge(x,y, on='AA_type', how='outer'), result_df_list).round(3)

        # Save to CSV / PKL
        csv_name = f"enrichment_{species_name}_{analysis_type}.csv"
        pkl_name = f"enrichment_{species_name}_{analysis_type}.pkl"
        final_df.to_csv(csv_name, index=False)
        final_df.to_pickle(pkl_name)

        # Optionally rename columns
        # e.g. .columns = [col.replace('AA', analysis_type) for col in final_df.columns]
        # But if you actually do that rename *before* saving, it changes the column IDs in your final output
        # If you want to replicate exactly the notebook output, do it after saving:
        final_df.columns = [col.replace('AA', analysis_type) for col in final_df.columns]

        print(f"[{analysis_type}] => saved {csv_name} and {pkl_name}")

#########################
#   CLI Main            #
#########################

def main():
    parser = argparse.ArgumentParser(description="Run entanglement enrichment analysis for species data.")
    parser.add_argument("--pkl-file", required=True, help="Pickle file with rawdata_df (e.g., *seq_ent_v1.pkl).")
    parser.add_argument("--species-name", required=True, help="Species name (e.g., 'yeast','ecoli','human').")
    parser.add_argument("--legacy-data", action="store_true", help="Whether to apply class4 modifications to ent_seq.")
    args = parser.parse_args()

    # Load the rawdata_df
    if not os.path.exists(args.pkl_file):
        print(f"Error: {args.pkl_file} not found.")
        sys.exit(1)

    rawdata_df = pd.read_pickle(args.pkl_file)

    # Run the analysis
    run_full_analysis(rawdata_df, args.species_name, legacy_data=args.legacy_data)

    # Optionally, you could add a run_info.txt here
    # ...
    print("All analyses completed successfully.")

if __name__ == "__main__":
    main()

