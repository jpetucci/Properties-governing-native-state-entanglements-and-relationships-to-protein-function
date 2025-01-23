#!/usr/bin/env python3

import os
import sys
import numpy as np

def generate_clustered_entanglements(base_dir, species_name, output_dir):
    """
    Reads entanglement data from an NPZ file (named <species_name>_non_covalent_lassos_4_5_no_knots.npz),
    then creates a file for each gene ID in the data. Each file includes
    the first chain of the first subkey, preserving the same
    format as the original script.

    Parameters
    ----------
    base_dir : str
        Base directory where the NPZ file is stored.
    species_name : str
        The species name (e.g., 'human', 'ecoli', etc.).
    output_dir : str
        The directory where output files will be written.
    """

    # Build the path to the NPZ file
    npz_path = os.path.join(
        base_dir,
        f"{species_name}_non_covalent_lassos_4_5_no_knots.npz"
    )

    # Read the NPZ data (arr_0 is a dictionary-like object)
    with np.load(npz_path, allow_pickle=True) as npzfile:
        data = npzfile["arr_0"].item()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each gene ID in the data
    for geneid in data.keys():
        # Create a file named {geneid}_clustered_GE.txt
        out_filename = os.path.join(output_dir, f"{geneid}_clustered_GE.txt")
        with open(out_filename, "w") as file_out:
            # Only capture the first subkey
            for subkey in data[geneid].keys():
                # Chain ID is the first key in data[geneid][subkey]
                chainid = list(data[geneid][subkey])[0]

                # Loop over the entanglements for that chain
                for ent in data[geneid][subkey][chainid]:
                    # For each ent, prepend '+' to items after the second element
                    modified_ent_format = [
                        f"+{item}" if idx > 1 else item
                        for idx, item in enumerate(ent)
                    ]
                    # Write the line in the same format as the original code
                    file_out.write(
                        "Chain "
                        + chainid
                        + " | "
                        + str(tuple(modified_ent_format))
                        + " | \n"
                    )
                # Only process the first subkey
                break

    # "Unit test" portion: check that each gene only has a single PDB
    # Record any errors (if they exist)
    error_messages = []
    for geneid in data.keys():
        subkeys = list(data[geneid].keys())
        if len(subkeys) != 1:
            error_messages.append(f"Error: {geneid} has multiple PDBs")

        # Check if that single subkey has exactly one chain
        if len(data[geneid][subkeys[0]]) != 1:
            error_messages.append(f"Error: {geneid} has multiple chains in subkey {subkeys[0]}")

    return error_messages

def main():
    """
    Main function to parse command-line arguments, run the entanglement
    processing, and create a run info file summarizing results.
    """

    # Expect 3 command-line arguments: base_dir, species_name, output_dir
    if len(sys.argv) != 4:
        script_name = os.path.basename(__file__)
        print(f"Usage: python {script_name} <BASE_DIR> <SPECIES_NAME> <OUTPUT_DIR>")
        sys.exit(1)

    base_dir = sys.argv[1]
    species_name = sys.argv[2]
    output_dir = sys.argv[3]

    # Generate the clustered entanglements and capture any errors
    errors = generate_clustered_entanglements(base_dir, species_name, output_dir)

    # Create a run info report file
    report_file = os.path.join(output_dir, f"{species_name}_run_info.txt")
    with open(report_file, "w") as report:
        report.write("===== RUN INFO =====\n")
        report.write(f"Base directory: {base_dir}\n")
        report.write(f"Species name: {species_name}\n")
        report.write(f"Output directory: {output_dir}\n\n")

        report.write("===== STATUS =====\n")
        if errors:
            report.write("Completed with errors:\n")
            for err in errors:
                report.write(f" - {err}\n")
        else:
            report.write("Script completed successfully (no errors detected).\n")

        report.write("\n===== FILES CREATED =====\n")
        report.write("One *clustered_GE.txt file per gene ID.\n")
        report.write(f"Report file: {report_file}\n")

    print(f"Processing completed. Report: {report_file}")

if __name__ == "__main__":
    main()

