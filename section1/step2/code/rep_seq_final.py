#!/usr/bin/env python3

import os
import sys
import numpy as np
import warnings
from collections import defaultdict
from MDAnalysis import Universe

warnings.filterwarnings("ignore")

def map_sequences(input_file, pdb_directory, output_dir):
    """
    Reads a tab-delimited file (gene, pdb, chain) and, for each row:
      - Loads the corresponding PDB structure
      - Extracts the unique residue names (amino acid sequence) via CA atoms
      - Saves the sequence as a .npz file in output_dir

    Parameters
    ----------
    input_file : str
        The path to the file that lists gene, pdb, and chain columns.
    pdb_directory : str
        The path to the directory containing PDB files for the organism.
    output_dir : str
        The path to the output directory where .npz files will be stored.

    Returns
    -------
    list
        A list of the files created (one per line in input_file).
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the lines from the input file
    data = np.loadtxt(input_file, dtype=str)

    files_created = []
    for line in data:
        # Print each line as in the original code
        print(line)

        gene, pdb, chain = line

        # Load the universe for the gene/PDB combination
        # Original code references e.g.: gene_pdb.pdb1
        pdb_path = os.path.join(pdb_directory, f"{gene}_{pdb}.pdb1")
        structure = Universe(pdb_path, format="PDB").select_atoms(f"name CA and segid {chain}")

        # Extract unique residue IDs and names
        resids, idx = np.unique(structure.resids, return_index=True)
        full_aa_seq = structure.resnames[idx]

        # Save sequence in an npz file, preserving original naming scheme
        output_npz = os.path.join(output_dir, f"{gene}_{pdb}_{chain}.npz")
        np.savez(output_npz, full_aa_seq)
        files_created.append(output_npz)

    return files_created

def main():
    """
    Main function: parses command-line arguments, runs sequence mapping,
    and creates a small run info file summarizing inputs and outputs.
    """

    # We expect 3 arguments: input_file, pdb_directory, output_dir
    if len(sys.argv) != 4:
        script_name = os.path.basename(__file__)
        print(f"Usage: python {script_name} <INPUT_FILE> <PDB_DIRECTORY> <OUTPUT_DIR>")
        sys.exit(1)

    input_file = sys.argv[1]
    pdb_directory = sys.argv[2]
    output_dir = sys.argv[3]

    # Run the mapping process
    files_created = map_sequences(input_file, pdb_directory, output_dir)

    # Generate a run info report
    report_file = os.path.join(output_dir, "run_info.txt")
    with open(report_file, "w") as report:
        report.write("===== RUN INFO =====\n")
        report.write(f"Input file: {input_file}\n")
        report.write(f"PDB directory: {pdb_directory}\n")
        report.write(f"Output directory: {output_dir}\n\n")

        report.write("===== FILES CREATED =====\n")
        for f_created in files_created:
            report.write(f"{f_created}\n")

        report.write("\n===== STATUS =====\n")
        report.write("Script completed successfully.\n")

    print(f"Run info written to: {report_file}")

if __name__ == "__main__":
    main()

