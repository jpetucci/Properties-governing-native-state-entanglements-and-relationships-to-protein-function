#!/usr/bin/env python3

import os
import sys
import glob
import random
import statistics
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

###############################
# Global Dictionaries & Lists #
###############################

aalocal_dict = {
    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
    'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I',
    'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
    'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'
}

modification_dict = {
    'MSE':'MET', 'LLP':'LYS', 'MLY':'LYS', 'CSO':'CYS', 'KCX':'LYS', 'CSS':'CYS', 'OCS':'CYS',
    'NEP':'HIS', 'CME':'CYS', 'SEC':'CYS', 'CSX':'CYS', 'CSD':'CYS', 'SEB':'SER', 'SEP':'SER',
    'SMC':'CYS', 'SNC':'CYS', 'CAS':'CYS', 'CAF':'CYS', 'FME':'MET', '143':'CYS', 'PTR':'TYR',
    'MHO':'MET', 'ALY':'LYS', 'BFD':'ASP', 'TPO':'THR', 'DHA':'SER', 'CSP':'CYS', 'AME':'MET',
    'YCM':'CYS', 'T8L':'THR', 'TPQ':'TYR', 'SCY':'CYS', 'MLZ':'LYS', 'TYS':'TYR', 'SCS':'CYS',
    'LED':'LEU', 'KPI':'LYS', 'PCA':'GLN', 'DSN':'SER'
}

three_to_one_dict = {
    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLU':'E', 'GLN':'Q', 'GLY':'G',
    'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S',
    'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V', 'MSE':'M', 'LLP':'K', 'MLY':'K', 'CSO':'C',
    'KCX':'K', 'CSS':'C', 'OCS':'C', 'NEP':'H', 'CME':'C', 'SEC':'U', 'CSX':'C', 'CSD':'C',
    'SEB':'S', 'SEP':'S', 'SMC':'C', 'SNC':'C', 'CAS':'C', 'CAF':'C', 'FME':'M', '143':'C',
    'PTR':'Y', 'MHO':'M', 'ALY':'K', 'BFD':'D', 'TPO':'T', 'DHA':'S', 'CSP':'C', 'AME':'M',
    'YCM':'C', 'T8L':'T', 'TPQ':'Y', 'SCY':'C', 'MLZ':'K', 'TYS':'Y', 'SCS':'C', 'LED':'L',
    'KPI':'K', 'PCA':'Q', 'DSN':'S'
}

reduced_aa_dict = {
    'A':'H', 'C':'P', 'D':'C', 'E':'C', 'F':'H', 'G':'H',
    'H':'P', 'I':'H', 'K':'C', 'L':'H', 'M':'A', 'N':'P',
    'P':'H', 'Q':'P', 'R':'C', 'S':'P', 'T':'P', 'V':'H',
    'W':'A', 'Y':'A'
}

twoletter_dict = {'A':'H', 'H':'H', 'P':'P', 'C':'P'}

aalist = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

###############################
# Helper Functions            #
###############################

def convertAA(AA3):
    """
    Converts a three-letter amino acid code to a one-letter code using 'aalocal_dict'.
    If the code is not found, returns the original code.
    """
    return aalocal_dict.get(AA3, AA3)

def convertAAmod(AA3):
    """
    Converts a modified amino acid code to its parent amino acid code using 'modification_dict'.
    If the code is not found, returns the original code.
    """
    return modification_dict.get(AA3, AA3)

def mapto_2letter_alphabet(inputsequence_list):
    """
    Maps an input sequence of amino acids to a reduced 2-letter alphabet.
    """
    return "".join(
        twoletter_dict.get(reduced_aa_dict.get(x, x), x) 
        for x in inputsequence_list
    )

def modify_array_class4(arr):
    """
    Modifies an array of entanglement labels.
    Replaces all 4's with 0's.
    For each index where the value is 3, labels the neighboring positions within two residues as 4,
    unless they are already labeled 3.
    """
    arr = np.array(arr)
    arr[arr == 4] = 0
    indices = np.where(arr == 3)[0]
    for index in indices:
        start_index = max(0, index - 2)
        end_index = min(len(arr), index + 3)
        for i in range(start_index, index):
            if arr[i] != 3:
                arr[i] = 4
        for i in range(index + 1, end_index):
            if arr[i] != 3:
                arr[i] = 4
    return arr

def mode_rand(column_elements):
    """
    Returns the mode of the non-zero elements in 'column_elements'.
    If multiple modes, selects one at random.
    If all elements are zero, returns 0.
    """
    unique_elements = set(column_elements)
    if unique_elements == {0}:
        return 0
    else:
        nonzero = column_elements[column_elements != 0]
        return random.choice(statistics.multimode(nonzero.tolist()))

def singlefeat(item):
    """
    Concatenates a list or tuple of strings into a single string.
    """
    return ''.join(item)

###############################
# Processing Functions        #
###############################

def read_pdb_sequences(rawdata_df, pdbseqdir):
    """
    Reads PDB sequences (.npz) for each row in rawdata_df.
    Expects columns: geneid, pdbid, repchain
    """
    pdb_seq_list = []
    valid_indices = []

    for rowid in tqdm(range(len(rawdata_df)), desc="Reading PDB sequences"):
        geneid = rawdata_df.iloc[rowid].geneid
        pdbid = rawdata_df.iloc[rowid].pdbid
        repchain = rawdata_df.iloc[rowid].repchain
        filename = os.path.join(pdbseqdir, f"{geneid}_{pdbid}_{repchain}.npz")
        try:
            pdb_seq = np.load(filename, allow_pickle=True)["arr_0"]
            pdb_seq_list.append(pdb_seq)
            valid_indices.append(rowid)
        except FileNotFoundError:
            print(f"Warning: Missing data for {filename}")
            continue

    rawdata_df = rawdata_df.iloc[valid_indices].reset_index(drop=True)
    rawdata_df['pdb_seq'] = pdb_seq_list
    return rawdata_df

def process_pdb_sequences(rawdata_df):
    """
    Converts from 3-letter code to 1-letter code and replaces modifications.
    Removes sequences with unknown or bad amino acids.
    """
    valid_indices = []
    converted_sequences = []

    for idx, seq in enumerate(rawdata_df['pdb_seq']):
        converted_seq = [convertAA(convertAAmod(aa)) for aa in seq]
        if all(aa in aalist for aa in converted_seq):
            converted_sequences.append(converted_seq)
            valid_indices.append(idx)
        else:
            print(f"Invalid amino acid codes in sequence at index {idx}. Skipping.")

    rawdata_df = rawdata_df.iloc[valid_indices].reset_index(drop=True)
    rawdata_df['pdb_seq_converted'] = converted_sequences
    return rawdata_df

def load_entanglement_sequences(rawdata_df, entdirectory, clustering_dir):
    """
    Loads entanglement sequences for each gene and processes them.
    Collapses multiple entanglement definitions into a single 'mode' array.
    """
    rep_list = []
    maxcrossings_list = []
    numcrossings_list = []
    non_collapsed_list = []

    valid_indices = []

    for idx, (geneid, pdbid, repchain) in enumerate(tqdm(rawdata_df[['geneid', 'pdbid', 'repchain']].values, 
                                                         desc="Processing entanglements")):
        filelist = glob.glob(os.path.join(entdirectory, f"{geneid}*"))
        rep_seq_list = []

        # Attempt to read clustered entanglements
        clusterfile = os.path.join(clustering_dir, f"{geneid}_clustered_GE.txt")
        try:
            clusterfile_df = pd.read_csv(clusterfile, sep='|', header=None)
            clusterfile_df = clusterfile_df[clusterfile_df[0] == f'Chain {repchain} ']
            numcrossings = len(clusterfile_df)
            numcrossings_list.append(numcrossings)
            # parse crossing columns to see how many crossing residues there are
            # (the code in original snippet is approximate, so we'll store just the max)
            maxcrossings = clusterfile_df[1].apply(lambda x: len(x.split(','))).max()
            maxcrossings_list.append(maxcrossings)
        except FileNotFoundError:
            print(f"Warning: Missing cluster file for {geneid}")
            numcrossings_list.append(0)
            maxcrossings_list.append(0)

        # Load each .npz entanglement vector
        for file in filelist:
            arr = np.load(file)["arr_0"]
            # Use only valid residue indices
            arr = arr[arr != -1]  
            arr = modify_array_class4(arr)
            rep_seq_list.append(arr)
            non_collapsed_list.append(arr)

        # Collapse gene entanglements to a single vector using mode
        if rep_seq_list:
            rep_seq_array = np.array(rep_seq_list)
            rep_seq_mode = pd.DataFrame(rep_seq_array).apply(mode_rand, axis=0).values
            rep_list.append(rep_seq_mode)
            valid_indices.append(idx)
        else:
            print(f"No entanglement data for {geneid}. Skipping.")
            continue

    rawdata_df = rawdata_df.iloc[valid_indices].reset_index(drop=True)
    rawdata_df['ent_seq'] = rep_list
    return rawdata_df

def check_sequence_lengths(rawdata_df):
    """
    Checks that the entanglement sequence and pdb protein sequence have the same length.
    """
    valid_indices = []
    for index, row in tqdm(rawdata_df.iterrows(), total=rawdata_df.shape[0], desc="Checking sequence lengths"):
        if len(row['pdb_seq']) != len(row['ent_seq']):
            print(f"Entanglement sequence length {len(row['ent_seq'])} != PDB sequence length {len(row['pdb_seq'])} at index {index}. Skipping.")
        else:
            valid_indices.append(index)
    return rawdata_df.loc[valid_indices].reset_index(drop=True)

def flatten_sequences(rawdata_df):
    """
    Flattens the pdb sequences into strings for exporting or further processing.
    """
    rawdata_df['pdb_seq_converted_flat'] = rawdata_df['pdb_seq_converted'].apply(singlefeat)
    rawdata_df['pdb_seq_flat'] = rawdata_df['pdb_seq'].apply(
        lambda seq: singlefeat([three_to_one_dict.get(aa, aa) for aa in seq])
    )
    return rawdata_df

def read_secondary_structure(rawdata_df, processed_stride):
    """
    Reads secondary structure info from 'processed_stride_*.txt'. 
    Filters out sequences with mismatched lengths.
    """
    secondary_structure_list = []
    valid_indices = []

    for index, row in tqdm(rawdata_df.iterrows(), total=rawdata_df.shape[0], desc="Reading secondary structures"):
        stride_file = os.path.join(
            processed_stride, 
            f"processed_stride_{row['geneid']}_{row['pdbid']}.txt"
        )
        try:
            df = pd.read_csv(stride_file, usecols=[0,5,6,7,8], header=None)
            df.columns = ['AA', 'SS', 'phi', 'psi', 'rsaa']
            pdb_seq = row['pdb_seq']
            ss_seq = df['SS'].values

            if len(pdb_seq) != len(ss_seq):
                print(f"Length mismatch for {row['geneid']}:{row['pdbid']}. Skipping.")
                continue
            else:
                secondary_structure_list.append(ss_seq)
                valid_indices.append(index)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"Warning: Missing or empty stride data for {row['geneid']}:{row['pdbid']}")
            continue

    rawdata_df = rawdata_df.loc[valid_indices].reset_index(drop=True)
    rawdata_df['secondary_structure'] = secondary_structure_list
    return rawdata_df


####################################
# Main driver function             #
####################################

def run_analysis(
    inputfile, 
    pdbseqdir, 
    entdirectory, 
    clustering_dir, 
    processed_stride_dir,
):
    """
    Runs the entire data processing pipeline for a single set of paths.
    Returns the final processed DataFrame.
    """

    # 1) Read geneid, pdbid, repchain
    rawdata_df = pd.read_csv(inputfile, sep=r'\s+', names=['geneid', 'pdbid', 'repchain'])

    # 2) Read in pdb sequences for each pdb with entanglement
    rawdata_df = read_pdb_sequences(rawdata_df, pdbseqdir)

    # 3) Convert from 3-letter code to 1-letter code and replace modifications
    rawdata_df = process_pdb_sequences(rawdata_df)

    # 4) Load entanglement sequences
    rawdata_df = load_entanglement_sequences(rawdata_df, entdirectory, clustering_dir)

    # 5) Check that entanglement sequence matches PDB protein sequence length
    rawdata_df = check_sequence_lengths(rawdata_df)

    # 6) Flatten sequences for further processing/export
    rawdata_df = flatten_sequences(rawdata_df)

    # 7) Read secondary structure info
    rawdata_df = read_secondary_structure(rawdata_df, processed_stride_dir)

    # 8) Map sequences to reduced 2-letter alphabet
    rawdata_df['pdb_seq_converted_2letter_flat'] = rawdata_df['pdb_seq_converted'].apply(mapto_2letter_alphabet)

    return rawdata_df

def main():
    """
    Command-line interface. Example usage:
      python script.py <INPUTFILE> <PDBSEQDIR> <ENTDIRECTORY> <CLUSTERING_DIR> <PROCESSED_STRIDE_DIR> <OUTPUT_PICKLE>
    """
    if len(sys.argv) < 6:
        script_name = os.path.basename(__file__)
        print(f"Usage: python {script_name} <INPUTFILE> <PDBSEQDIR> <ENTDIRECTORY> <CLUSTERING_DIR> <PROCESSED_STRIDE_DIR> [OUTPUT_PICKLE]")
        sys.exit(1)

    inputfile = sys.argv[1]
    pdbseqdir = sys.argv[2]
    entdirectory = sys.argv[3]
    clustering_dir = sys.argv[4]
    processed_stride_dir = sys.argv[5]

    # Optional: the user can provide an output pickle path
    output_pickle = None
    if len(sys.argv) > 6:
        output_pickle = sys.argv[6]

    # Run the pipeline
    final_df = run_analysis(
        inputfile,
        pdbseqdir,
        entdirectory,
        clustering_dir,
        processed_stride_dir
    )

    # Save results if desired
    if output_pickle:
        final_df.to_pickle(output_pickle)
        print(f"DataFrame saved to {output_pickle}")

    # Create a run-info file
    # (Named similarly to "run_info.txt" or you can incorporate species in the filename.)
    run_info_path = os.path.join(os.path.dirname(output_pickle or '.'), "run_info.txt")
    with open(run_info_path, "w") as report:
        report.write("===== RUN INFO =====\n")
        report.write(f"Input file: {inputfile}\n")
        report.write(f"PDB seq dir: {pdbseqdir}\n")
        report.write(f"Entanglement dir: {entdirectory}\n")
        report.write(f"Clustering dir: {clustering_dir}\n")
        report.write(f"Processed stride dir: {processed_stride_dir}\n")
        report.write(f"Output pickle: {output_pickle}\n\n")
        report.write("===== STATUS =====\n")
        report.write("Script completed successfully.\n")

    print(f"Run info written to: {run_info_path}")
    print("Script completed successfully.")

if __name__ == "__main__":
    main()

