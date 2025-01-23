#!/usr/bin/env python3

"""
Refactored pipeline that can handle ecoli, yeast, or human,
with an optional --legacy-data flag to apply 'class4' modifications
to ent_seq if desired.

Usage Example:
  python pipeline_species.py \
    --functional-data-path /path/to/yeast_functional_data \
    --rawdata-df-path /path/to/yeast_seq_ent_v1.pkl \
    --pdb-directory /path/to/yeast_exp_pdbs \
    --output-csv yeast_functionaldata_09192024.csv \
    --species yeast \
    --legacy-data
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import tqdm
from collections import defaultdict
from sklearn import preprocessing as pp
from MDAnalysis import Universe

#########################
# -- Global Dictionaries
#########################

# Maps the standard 20 AA to a 4-letter alphabet
reduced_aa_dict = {
    'A': 'H', 'C': 'P', 'D': 'C', 'E': 'C', 'F': 'H', 'G': 'H', 'H': 'P',
    'I': 'H', 'K': 'C', 'L': 'H', 'M': 'A', 'N': 'P', 'P': 'H', 'Q': 'P',
    'R': 'C', 'S': 'P', 'T': 'P', 'V': 'H', 'W': 'A', 'Y': 'A'
}

# Maps the 4-letter alphabet down to 2-letter alphabet
twoletter_dict = {
    'A': 'H',
    'H': 'H',
    'P': 'P',
    'C': 'P'
}

# Define separate dictionaries for each species
functional_label_dict_ecoli = {
    'ecoli_V5_functional_PDB_free-ligand.npz': 1,
    'ecoli_V5_functional_protein.npz': 2,
    'ecoli_V5_functional_active site.npz': 3,
    'PDB_metal_interface_mc.npz': 4,
    'ecoli_V5_functional_PDB_RNA-binding.npz': 5,
    'ecoli_V5_functional_PDB_DNA-binding.npz': 6,
    'ecoli_V5_functional_zinc finger region.npz': 7
}

functional_label_dict_yeast = {
    'yeast_V2_functional_PDB_free-ligand.npz': 1,
    'yeast_V2_functional_protein.npz': 2,
    'yeast_V2_functional_active site.npz': 3,
    'PDB_metal_interface_mc.npz': 4,
    'yeast_V2_functional_PDB_RNA-binding.npz': 5,
    'yeast_V2_functional_PDB_DNA-binding.npz': 6,
    'yeast_V2_functional_zinc finger region.npz': 7
}

functional_label_dict_human = {
    'human_V2_functional_PDB_free-ligand.npz': 1,
    'human_V2_functional_protein.npz': 2,
    'human_V2_functional_active site.npz': 3,
    'PDB_metal_interface_mc.npz': 4,
    'human_V2_functional_PDB_DNA-binding.npz': 5,
    'human_V2_functional_PDB_RNA-binding.npz': 6,
    'human_V2_functional_zinc finger region.npz': 7
}

############################
# --- Helper Functions ---
############################

def mapto_4letter_alphabet(inputsequence_list):
    return "".join(reduced_aa_dict[x] for x in inputsequence_list)

def mapto_2letter_alphabet(inputsequence_list):
    return "".join(twoletter_dict[reduced_aa_dict[x]] for x in inputsequence_list)

def slidingwindow(inputstr, windowsize):
    return [inputstr[i:i+windowsize] for i in range(len(inputstr) - (windowsize - 1))]

def middle_element(inputlist):
    listlength = len(inputlist)
    elementid = (listlength - 1)//2
    return inputlist[elementid]

def insidebuffer(buffersize, inputlist, criticalvalue):
    listlength = len(inputlist)
    middleelementid = (listlength - 1)//2
    region = inputlist[middleelementid - buffersize : middleelementid + buffersize + 1]
    return (criticalvalue in region)

def find_misaligned2(inputseq, bufferregion, criticalvalue):
    return_list = []
    for sublist_id, sublist in enumerate(inputseq):
        if criticalvalue in sublist:
            if insidebuffer(bufferregion, sublist, criticalvalue):
                return_list.append(sublist_id)
        else:
            return_list.append(sublist_id)
    return return_list

def sum_all_but_key(dictin, notkey):
    return sum(value for key, value in dictin.items() if key not in notkey)

def modify_array_class4(arr):
    """
    1) Replace all 4's with 0.
    2) For each index where arr == 3, set the previous two and next two to 4 if they're not already 3.
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
        for i in range(index+1, end_index):
            if arr[i] != 3:
                arr[i] = 4

    return arr

def return_flat_list(inputcol, rawdata_df):
    return list(pd.core.common.flatten(rawdata_df[inputcol].values))

def get_pdb_functional_sites2(pdbidin, geneidin, chainin, function_class_in, df, rawdata_df, function_label_dict):
    subdf = df[
        (df.pdbid == pdbidin) &
        (df.geneid == geneidin) &
        (df.chain == chainin) &
        (df.function_class == function_class_in)
    ]
    pdblen = rawdata_df[
        (rawdata_df.pdbid == pdbidin) &
        (rawdata_df.geneid == geneidin) &
        (rawdata_df.repchain == chainin)
    ].ent_seq_len.values[0]
    newseq = [0]*pdblen
    pdbres_map_dict = rawdata_df[
        (rawdata_df.pdbid == pdbidin) &
        (rawdata_df.geneid == geneidin) &
        (rawdata_df.repchain == chainin)
    ].pdbres_map.values[0]

    if subdf.shape[0] == 1:
        for _, row in subdf.iterrows():
            for element in row.seq:
                idx = pdbres_map_dict[element]
                if newseq[idx] == 0:
                    newseq[idx] = function_label_dict[row.function_class]
                elif isinstance(newseq[idx], list):
                    newseq[idx].append(function_label_dict[row.function_class])
                else:
                    current_val = newseq[idx]
                    newseq[idx] = [current_val, function_label_dict[row.function_class]]
        return newseq
    else:
        return newseq

#############################
# --- Main Pipeline Code ---
#############################

def run_pipeline(
    functional_data_path,
    rawdata_df_path,
    pdb_directory,
    output_csv,
    species,
    legacy_data
):
    """
    species can be 'ecoli', 'yeast', or 'human'.
    If legacy_data is True, applies modify_array_class4 to ent_seq.
    """
    # 1) Load the correct dictionary
    if species.lower() == 'ecoli':
        function_label_dict = functional_label_dict_ecoli
    elif species.lower() == 'yeast':
        function_label_dict = functional_label_dict_yeast
    elif species.lower() == 'human':
        function_label_dict = functional_label_dict_human
    else:
        print(f"Unsupported species: {species}. Must be 'ecoli', 'yeast', or 'human'.")
        sys.exit(1)

    print(f"Using functional dictionary for {species}:\n{function_label_dict}")

    print("1) Loading raw data DataFrame...")
    rawdata_df = pd.read_pickle(rawdata_df_path)

    # Apply legacy data transformation if requested
    if legacy_data:
        print("Applying 'class4' modifications to ent_seq (legacy data mode).")
        rawdata_df['ent_seq'] = rawdata_df['ent_seq'].apply(modify_array_class4)
    else:
        print("Skipping 'class4' modifications (not legacy mode).")

    print("2) Building PDB residue mapping (MDAnalysis) ...")
    map_dict_list = []
    for _, row in tqdm.tqdm(rawdata_df.iterrows(), total=rawdata_df.shape[0]):
        gene = row.geneid
        pdb = row.pdbid
        chain = row.repchain
        pdb_path = os.path.join(pdb_directory, f"{gene}_{pdb}.pdb1")

        structure = Universe(pdb_path, format='PDB').select_atoms(f"name CA and segid {chain}")
        PDB_resids, i = np.unique(structure.resids, return_index=True)
        structure2 = structure[i]
        resids = structure2.resids

        local_map = {}
        for j, res_id in enumerate(resids):
            local_map[res_id] = j
        map_dict_list.append(local_map)

    rawdata_df['pdbres_map'] = map_dict_list

    print("3) Adding reduced-alphabet columns (2letter & 4letter) ...")
    rawdata_df['pdb_seq_converted_4letter_flat'] = rawdata_df['pdb_seq_converted'].apply(mapto_4letter_alphabet)
    rawdata_df['pdb_seq_converted_2letter_flat'] = rawdata_df['pdb_seq_converted'].apply(mapto_2letter_alphabet)

    print("Value counts for flattened 'ent_seq':")
    print(pd.Series(pd.core.common.flatten(rawdata_df.ent_seq)).value_counts())

    print("4) Loading functional data from NPZ files ...")
    functional_dict = {}
    pattern = os.path.join(functional_data_path, '*.npz')
    for file in glob.glob(pattern):
        keyname = os.path.basename(file)
        if '_all' in keyname:
            continue
        data_obj = np.load(file, allow_pickle=True)['arr_0'].item()
        functional_dict[keyname] = data_obj

    print("5) Building functional DataFrame...")
    type_gene_pdb_chain_list = []
    for key, top_dict in functional_dict.items():
        for subkey, subval in top_dict.items():
            for subsubkey, subsubval in subval.items():
                for subsubsubkey, rowval in subsubval.items():
                    type_gene_pdb_chain_list.append([
                        key, subkey, subsubkey, subsubsubkey, rowval
                    ])
    df_func = pd.DataFrame(
        type_gene_pdb_chain_list,
        columns=['function_class', 'geneid', 'pdbid', 'chain', 'seq']
    )

    print("6) Storing 'ent_seq_len' in the raw DataFrame...")
    rawdata_df['ent_seq_len'] = rawdata_df['ent_seq'].apply(len)

    print("7) Mapping functional sites onto raw data (adding new columns for each function label)...")
    for fclass in tqdm.tqdm(function_label_dict.keys(), total=len(function_label_dict)):
        rawdata_df[fclass] = rawdata_df.apply(
            lambda x: get_pdb_functional_sites2(
                x.pdbid, x.geneid, x.repchain, fclass, df_func, rawdata_df, function_label_dict
            ), axis=1
        )

    # Window-based analysis
    window_size = 3
    buffersize = 0
    criticalval = 3
    alphabetsizeselection = 'pdb_seq_converted_2letter_flat'
    windowcolumnname = '2letter_windows'

    print("8) Creating sliding windows and filtering ...")
    for functional_class in tqdm.tqdm(function_label_dict.keys(), total=len(function_label_dict)):
        rawdata_df[windowcolumnname] = rawdata_df[alphabetsizeselection].apply(
            lambda x: slidingwindow(x, window_size)
        )
        rawdata_df['ent_window'] = rawdata_df['ent_seq'].apply(
            lambda x: slidingwindow(x, window_size)
        )
        rawdata_df[f"{functional_class}_window"] = rawdata_df[fclass].apply(
            lambda x: slidingwindow(x, window_size)
        )

        rawdata_df['select_elements'] = rawdata_df['ent_window'].apply(
            lambda x: find_misaligned2(x, buffersize, criticalval)
        )

        rawdata_df['ent_window'] = rawdata_df.apply(
            lambda x: [x.ent_window[i] for i in x.select_elements], axis=1
        )
        rawdata_df[windowcolumnname] = rawdata_df.apply(
            lambda x: [x[windowcolumnname][i] for i in x.select_elements], axis=1
        )
        rawdata_df[f"{functional_class}_window"] = rawdata_df.apply(
            lambda x: [x[f"{functional_class}_window"][i] for i in x.select_elements],
            axis=1
        )

        # Convert ent_window to 1/0 if inside buffer
        rawdata_df['ent_window'] = rawdata_df['ent_window'].apply(
            lambda winlist: [1 if insidebuffer(buffersize, w, criticalval) else 0 for w in winlist]
        )

        # Convert functional class window to 1/0
        rawdata_df[f"{functional_class}_window"] = rawdata_df[f"{functional_class}_window"].apply(
            lambda winlist: [
                1 if insidebuffer(buffersize, w, function_label_dict[functional_class]) else 0
                for w in winlist
            ]
        )

    print("9) Flattening data for final analysis ...")
    rawdata_df['pdbid'] = rawdata_df[['pdbid','ent_window']].apply(
        lambda x: [x.pdbid]*len(x.ent_window), axis=1
    )
    colselect = ['ent_window','pdbid', windowcolumnname] + [f"{k}_window" for k in function_label_dict.keys()]

    from collections import defaultdict
    count_df = pd.DataFrame(
        map(lambda c: return_flat_list(c, rawdata_df), colselect)
    ).T
    count_df.columns = colselect

    count_df['center_site'] = count_df[windowcolumnname].apply(middle_element)
    count_df['CR_type'] = np.where(count_df['center_site'] == 'P', 1, 0)
    count_df.drop(columns=['center_site'], inplace=True)

    # One-hot encode 2letter_windows
    if windowcolumnname in count_df.columns:
        count_df = pd.concat([count_df, pd.get_dummies(count_df[windowcolumnname])], axis=1)
        count_df.drop(columns=[windowcolumnname], inplace=True)

    for col in count_df.columns:
        if col != 'pdbid':
            count_df[col] = count_df[col].astype(int)

    print("Polynomial expansion ...")
    from sklearn.preprocessing import PolynomialFeatures
    df_for_poly = count_df.drop(columns=['ent_window','pdbid'], errors='ignore')
    poly = PolynomialFeatures(2)
    poly_matrix = poly.fit_transform(df_for_poly)
    colnames = poly.get_feature_names(df_for_poly.columns)
    df_final = pd.DataFrame(poly_matrix, columns=colnames)

    df_final = df_final[df_final.columns[~df_final.columns.str.contains('2')]]
    if '1' in df_final.columns:
        df_final.drop(columns=['1'], inplace=True)

    df_final['ent_window'] = count_df['ent_window']
    df_final['pdbid'] = count_df['pdbid']

    summary_count_list = []
    for colname in df_final.columns[:-2]:
        local_counts = df_final[['ent_window', colname]].astype(int).value_counts().to_dict()
        local_counts = defaultdict(lambda: 0, local_counts)
        summary_count_list.append([
            colname,
            local_counts[(0,0)],
            local_counts[(1,0)],
            local_counts[(0,1)],
            local_counts[(1,1)]
        ])
    interaction_df = pd.DataFrame(summary_count_list, columns=[
        'name','ent=0,func=0','ent=1,func=0','ent=0,func=1','ent=1,func=1'
    ])
    keep_names = interaction_df[interaction_df['ent=1,func=1'] >= 0]['name'].tolist()
    df_final = df_final[keep_names + ['ent_window','pdbid']]

    # Choose columns to save (edit as needed)
    final_columns = ['CR_type','ent_window','pdbid']
    print(f"Saving final CSV to {output_csv} ...")
    out_cols = [col for col in final_columns if col in df_final.columns]
    df_final[out_cols].to_csv(output_csv, index=False)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Refactored pipeline for ecoli/yeast/human, 1-to-1 output, with legacy data option.")
    parser.add_argument("--functional-data-path", required=True,
                        help="Directory containing NPZ functional data.")
    parser.add_argument("--rawdata-df-path", required=True,
                        help="Pickle file with raw data DataFrame.")
    parser.add_argument("--pdb-directory", required=True,
                        help="Directory containing <gene>_<pdb>.pdb1 files.")
    parser.add_argument("--output-csv", required=True,
                        help="Path to the final CSV output.")
    parser.add_argument("--species", required=True,
                        choices=["ecoli","yeast","human"],
                        help="Which species to process.")
    parser.add_argument("--legacy-data", action="store_true",
                        help="Whether to apply the 'class4' modifications to ent_seq.")
    args = parser.parse_args()

    run_pipeline(
        functional_data_path=args.functional_data_path,
        rawdata_df_path=args.rawdata_df_path,
        pdb_directory=args.pdb_directory,
        output_csv=args.output_csv,
        species=args.species,
        legacy_data=args.legacy_data
    )

if __name__ == "__main__":
    main()

