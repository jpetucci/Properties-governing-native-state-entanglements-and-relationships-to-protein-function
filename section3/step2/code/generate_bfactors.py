#!/usr/bin/env python3
"""
Extracts per-residue B-factors (tempfactors) from a series of PDB files using MDAnalysis.
For each row in a specified CSV (geneid, pdbid, repchain), it:
  1) Reads the PDB file (constructed as <geneid>_<pdbid>.pdb1) from the input directory.
  2) Selects CA atoms for the specified chain.
  3) Extracts unique residue IDs, residue names, and their corresponding B-factors.
  4) Writes a CSV file named bfactor_<geneid>.csv in the output directory.

Example usage:
  python bfactor_extractor.py \
      --inputdir /path/to/pdbs \
      --outputdir /path/to/output \
      --entanglefile /path/to/entangle_list.txt
"""

import os
import argparse
import pandas as pd
import numpy as np
from MDAnalysis import Universe


def extract_bfactors(inputdir, outputdir, entanglefile):
    """
    Reads an entanglement file containing (geneid, pdbid, repchain), and for each row:
      - Loads the corresponding PDB file via MDAnalysis.
      - Selects CA atoms for the given chain.
      - Collects residue IDs, names, and B-factors.
      - Writes the resulting data to 'bfactor_<geneid>.csv' in outputdir.
    """
    # Read the text/space-delimited file: geneid pdbid repchain
    rawdata_df = pd.read_csv(entanglefile, sep=r"\s+", names=["geneid", "pdbid", "repchain"])

    # Iterate each row to process the PDB
    for rowid in range(len(rawdata_df)):
        genename = rawdata_df.iloc[rowid]["geneid"]
        chainid  = rawdata_df.iloc[rowid]["repchain"]
        pdbid    = rawdata_df.iloc[rowid]["pdbid"]

        # Construct the path to the PDB
        pdb_path = os.path.join(inputdir, f"{genename}_{pdbid}.pdb1")

        # Load Universe, selecting only CA atoms from the specified chain
        # 'segid' must match the chain ID in the PDB
        u = Universe(pdb_path, topology_format="PDB").select_atoms(f"name CA and segid {chainid}")

        # Gather unique residue IDs, residue names, and corresponding B-factors
        resids, idx = np.unique(u.resids, return_index=True)
        full_aa_seq = u.resnames[idx]
        befactors   = u.tempfactors[idx]

        # Combine into a DataFrame
        df = pd.DataFrame({"resid": resids, "resname": full_aa_seq, "bfactor": befactors})

        # Write to CSV; includes an index by default (matching the original script's behavior)
        out_csv = os.path.join(outputdir, f"bfactor_{genename}.csv")
        df.to_csv(out_csv)
        print(f"Saved: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Extract per-residue B-factors from PDB files using MDAnalysis.")
    parser.add_argument("--inputdir", required=True, help="Path to input PDB files.")
    parser.add_argument("--outputdir", required=True, help="Path to directory for output CSV files.")
    parser.add_argument("--entanglefile", required=True, help="Space-delimited file with (geneid, pdbid, repchain).")

    args = parser.parse_args()
    os.makedirs(args.outputdir, exist_ok=True)
    extract_bfactors(args.inputdir, args.outputdir, args.entanglefile)


if __name__ == "__main__":
    main()

