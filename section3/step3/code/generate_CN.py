#!/usr/bin/env python3
"""
General-purpose script to compute CN/Theta/Tau features from PDB structures across different species.
It merges the logic previously used for E. coli, yeast, and human into one codebase.

Overview:
  1) Reads a CSV file (space-delimited) containing (geneid, pdbid, repchain).
  2) Loads corresponding .npz files (with PDB sequences) for each row.
  3) Converts 3-letter amino acids (including modifications) to standard 1-letter codes.
  4) Applies species-specific filters (if chosen: e.g. remove 'UNK' for yeast, remove 'TYX' for human).
  5) Checks for sequence length mismatch with STRIDE-processed files and drops mismatches.
  6) Uses Bio.PDB to calculate CN, Theta, Tau for each residue in each structure.
  7) Saves the list of result DataFrames to a pickle file.

Usage Example:
    python cn_theta_tau.py \
        --species yeast \
        --inputfile /path/to/yeast_entanglements.csv \
        --pdbseqdir /path/to/npz_sequences \
        --pdbdir /path/to/pdb_files \
        --stride-dir /path/to/stride_processed \
        --output-pickle my_yeast_cn.pkl \
        --n-jobs 10

Species-Specific Behavior:
  - yeast: drops rows if 'UNK' is in the sequence
  - human: drops rows if 'TYX' is in the sequence, also removes geneid == 'P01833'
  - ecoli: no special dropping of 'UNK' or 'TYX'
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import tqdm
from joblib import Parallel, delayed
from Bio.PDB import PDBParser, HSExposureCB, Selection
from Bio.PDB.vectors import calc_angle, calc_dihedral

# ------------------------------------------------------------------------------
#                          Global Dictionaries & Helpers
# ------------------------------------------------------------------------------
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
    """
    Convert canonical 3-letter code to single-letter using `aalocal_dict`.
    If not found, return the input unchanged.
    """
    return aalocal_dict.get(AA3, AA3)

def convertAAmod(AA3):
    """
    Convert a modified residue code to its canonical parent using `modification_dict`.
    If not found, return the input unchanged.
    """
    return modification_dict.get(AA3, AA3)

def searchleft(position, inputarray):
    """
    Helper function to recursively search left for a non-'HOLD' value in an array.
    """
    elementvalue = inputarray[position]
    if elementvalue == 'HOLD':
        if position >= 0:
            return searchleft(position - 1, inputarray)
        else:
            return searchright(position, inputarray)
    else:
        return elementvalue

def searchright(position, inputarray):
    """
    Helper function to recursively search right for a non-'HOLD' value in an array.
    """
    elementvalue = inputarray[position]
    if elementvalue == 'HOLD':
        try:
            return searchright(position + 1, inputarray)
        except IndexError:
            return searchleft(position, inputarray)
    else:
        return elementvalue

def imputeHOLD(inputseq):
    """
    Impute values labeled 'HOLD' by averaging the nearest valid neighbors.
    If at start/end, fallback to whichever neighbor is valid.
    """
    buildseq = []
    seqlen = len(inputseq)
    for i in range(seqlen):
        if inputseq[i] == 'HOLD':
            if i == 0:
                appendvalue = searchright(i, inputseq)
            elif i == seqlen - 1:
                appendvalue = searchleft(i, inputseq)
            else:
                left_val  = searchleft(i, inputseq)
                right_val = searchright(i, inputseq)
                appendvalue = (left_val + right_val) / 2
        else:
            appendvalue = inputseq[i]
        buildseq.append(appendvalue)
    return buildseq

def slidingwindow(seq, windowsize):
    """
    Return a list of all contiguous sublists of `seq` of length `windowsize`.
    """
    return [seq[i : i + windowsize] for i in range(len(seq) - windowsize + 1)]

def export_pickle(path, myobject):
    """Write any Python object to a pickle."""
    with open(path, 'wb') as outputfile:
        pickle.dump(myobject, outputfile)

def import_pickle(path):
    """Load a Python object from a pickle."""
    with open(path, 'rb') as inputfile:
        return pickle.load(inputfile)

# ------------------------------------------------------------------------------
#                    CN/Theta/Tau Calculation with Bio.PDB
# ------------------------------------------------------------------------------
def getCN(rowid, rawdata_df, pdbdir):
    """
    For the given rowid in `rawdata_df`, parse the corresponding PDB file,
    run HSExposureCB on the chain, compute 'CN_exp' and the angles/dihedrals (Theta & Tau).

    Returns a DataFrame with columns [CN_exp, Theta_exp, Tau_exp], indexed 1..N,
    or a ['bad pdb', rowid, geneid, pdbid] placeholder if something fails.
    """
    geneid  = rawdata_df.iloc[rowid]["geneid"]
    pdbid   = rawdata_df.iloc[rowid]["pdbid"]
    chainid = rawdata_df.iloc[rowid]["repchain"]

    pdbfilename = f"{geneid}_{pdbid}.pdb1"
    pdb_path = os.path.join(pdbdir, pdbfilename)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)
    model = structure[0]

    if chainid not in model.child_dict:
        return ["bad pdb", rowid, geneid, pdbid, f"Chain {chainid} not found"]

    chain = model[chainid]

    RADIUS = 12.0
    try:
        HSExposureCB(chain, RADIUS)
    except Exception as e:
        return ["bad pdb", rowid, geneid, pdbid, str(e)]

    residue_list = Selection.unfold_entities(chain, 'R')

    HOLDflag = False
    local_CN_list = []
    residueidlist = []
    residueobjlist = []

    # Collect HSExposureCB results and store vectors
    for r in residue_list:
        if 'CA' in r.child_dict:
            resid_number = r.get_full_id()[3][1]
            if resid_number not in residueidlist:
                residueidlist.append(resid_number)
                if not r.xtra:
                    local_CN_list.append('HOLD')
                    HOLDflag = True
                else:
                    local_CN_list.append(r.xtra['EXP_HSE_B_U'] + r.xtra['EXP_HSE_B_D'])
                residueobjlist.append(r['CA'].get_vector())

    # Theta: angle over sliding window size=3
    theta_windows = slidingwindow(residueobjlist, 3)
    theta_series = pd.Series([calc_angle(*win) for win in theta_windows])
    mean_theta = theta_series.mean()
    # Insert one mean at front, one at end => total len +2
    theta_series = pd.concat([pd.Series([mean_theta]), theta_series, pd.Series([mean_theta])])

    # Tau: dihedral over sliding window size=4
    tau_windows = slidingwindow(residueobjlist, 4)
    tau_series = pd.Series([calc_dihedral(*win) for win in tau_windows])
    mean_tau = tau_series.mean()
    # Insert one mean at front, two at the end => total len +3
    tau_series = pd.concat([pd.Series([mean_tau]), tau_series, pd.Series([mean_tau]), pd.Series([mean_tau])])

    # Impute 'HOLD' if found
    if HOLDflag:
        local_CN_list = imputeHOLD(local_CN_list)

    # Build final DataFrame
    df = pd.DataFrame({
        "CN_exp": local_CN_list,
        "Theta_exp": theta_series.tolist(),
        "Tau_exp": tau_series.tolist()
    })
    # Shift index by 1
    df.index = df.index + 1

    # Check sequence length
    seq_len = len(rawdata_df.loc[rowid, "pdb_seq_converted"])
    if len(df["CN_exp"]) != seq_len:
        print(f"Length mismatch: {geneid}, {pdbid} => {len(df['CN_exp'])} vs {seq_len}")

    return df

# ------------------------------------------------------------------------------
#                               Main Function
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute CN/Theta/Tau from PDB files, with species-specific filters and stride checks."
    )
    parser.add_argument("--species", choices=["ecoli", "yeast", "human"], required=True,
                        help="Which species to process (applies species-specific filters).")
    parser.add_argument("--inputfile", required=True,
                        help="Space-delimited file with columns: geneid, pdbid, repchain.")
    parser.add_argument("--pdbseqdir", required=True,
                        help="Directory with .npz files named <geneid>_<pdbid>_<chain>.npz.")
    parser.add_argument("--pdbdir", required=True,
                        help="Directory with .pdb1 files named <geneid>_<pdbid>.pdb1.")
    parser.add_argument("--stride-dir", required=True,
                        help="Directory containing processed STRIDE files named processed_stride_<geneid>_<pdbid>.txt.")
    parser.add_argument("--output-pickle", required=True,
                        help="Path to store the final list of DataFrames (CN_df_list) as a pickle.")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Parallel processes to use when computing CN/Theta/Tau. Default=4.")
    args = parser.parse_args()

    # Read the CSV (space-delimited), naming columns
    rawdata_df = pd.read_csv(args.inputfile, sep=r"\s+", names=["geneid", "pdbid", "repchain"])
    print(f"Loaded {len(rawdata_df)} rows from {args.inputfile}.")

    # Read .npz sequences for each row
    pdb_seq_list = []
    for idx in tqdm.tqdm(range(len(rawdata_df)), desc="Loading NPZ Sequences"):
        geneid = rawdata_df.iloc[idx]["geneid"]
        pdbid  = rawdata_df.iloc[idx]["pdbid"]
        chain  = rawdata_df.iloc[idx]["repchain"]

        npz_path = os.path.join(args.pdbseqdir, f"{geneid}_{pdbid}_{chain}.npz")
        try:
            arr = np.load(npz_path, allow_pickle=True)["arr_0"]
        except FileNotFoundError:
            print(f"Warning: Missing .npz data for row={idx}, {geneid}_{pdbid}_{chain}")
            arr = []
        pdb_seq_list.append(arr)

    rawdata_df["pdb_seq"] = pdb_seq_list
    # Convert to single-letter codes
    rawdata_df["pdb_seq_converted"] = rawdata_df["pdb_seq"].apply(
        lambda seq: [convertAA(convertAAmod(res)) for res in seq]
    )

    # Apply species-specific filters
    if args.species == "yeast":
        # Drop rows containing 'UNK'
        mask_unk = rawdata_df["pdb_seq"].apply(lambda x: 'UNK' in x)
        if mask_unk.any():
            print(f"Dropping {mask_unk.sum()} row(s) containing 'UNK' for yeast.")
            rawdata_df = rawdata_df[~mask_unk]

    elif args.species == "human":
        # Drop rows containing 'TYX'
        mask_tyx = rawdata_df["pdb_seq"].apply(lambda x: 'TYX' in x)
        if mask_tyx.any():
            print(f"Dropping {mask_tyx.sum()} row(s) containing 'TYX' for human.")
            rawdata_df = rawdata_df[~mask_tyx]
        # Also remove geneid == 'P01833'
        pre_count = len(rawdata_df)
        rawdata_df = rawdata_df[rawdata_df["geneid"] != "P01833"]
        removed = pre_count - len(rawdata_df)
        if removed > 0:
            print(f"Dropped {removed} row(s) with geneid == 'P01833' for human.")

    # E. coli case has no special dropping
    rawdata_df.reset_index(drop=True, inplace=True)
    print(f"Shape after species-specific filtering: {rawdata_df.shape}")

    # Check length mismatch with STRIDE
    bad_length_genelist = []
    for idx, row in tqdm.tqdm(rawdata_df.iterrows(), total=len(rawdata_df), desc="Checking STRIDE mismatch"):
        geneid = row["geneid"]
        pdbid  = row["pdbid"]
        stride_file = os.path.join(args.stride_dir, f"processed_stride_{geneid}_{pdbid}.txt")
        if not os.path.exists(stride_file):
            # If missing stride file, treat as mismatch
            bad_length_genelist.append([geneid, pdbid])
            continue

        try:
            df_stride = pd.read_csv(stride_file, usecols=[0,5,6,7,8], header=None)
        except Exception:
            bad_length_genelist.append([geneid, pdbid])
            continue

        df_stride.columns = ["AA", "SS", "phi", "psi", "rsaa"]
        if len(row["pdb_seq"]) != len(df_stride["AA"]):
            bad_length_genelist.append([geneid, pdbid])

    if bad_length_genelist:
        print("Dropping the following for STRIDE mismatch (geneid, pdbid):")
        for item in bad_length_genelist:
            print(item)
        bad_genes = set(x[0] for x in bad_length_genelist)
        rawdata_df = rawdata_df[~rawdata_df["geneid"].isin(bad_genes)]
        rawdata_df.reset_index(drop=True, inplace=True)

    print(f"Shape after STRIDE mismatch check: {rawdata_df.shape}")

    # Parallel CN/Theta/Tau calculation
    print("Computing CN/Theta/Tau features in parallel...")
    CN_df_list = Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(getCN)(idx, rawdata_df, args.pdbdir) for idx in range(len(rawdata_df))
    )

    # Identify any 'bad pdb' placeholders
    bad_pdbs = [x for x in CN_df_list if isinstance(x, list) and 'bad pdb' in x]
    if bad_pdbs:
        print("Encountered 'bad pdb' entries:")
        for item in bad_pdbs:
            print(item)
        # Remove them from final results, or keep them if you'd like to keep a record
        CN_df_list = [x for x in CN_df_list if not (isinstance(x, list) and 'bad pdb' in x)]

    print(f"Final CN_df_list length: {len(CN_df_list)}")

    # Write results to pickle
    export_pickle(args.output_pickle, CN_df_list)
    print(f"Saved results to {args.output_pickle}")


if __name__ == "__main__":
    main()

