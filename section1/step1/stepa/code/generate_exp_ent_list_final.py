#!/usr/bin/env python3

import os
import sys
import numpy as np

def main():
    """
    This script reads in entanglement data from an NPZ file and writes the
    first chain of the first subkey for each key to a CSV. It also generates
    a report file summarizing the inputs and outputs.
    """

    # Check for proper usage
    if len(sys.argv) != 3:
        script_name = os.path.basename(__file__)
        print(f"Usage: python {script_name} <BASE_DIR> <SPECIES_NAME>")
        sys.exit(1)

    base_dir = sys.argv[1]
    species_name = sys.argv[2]

    # Build the NPZ path
    npz_path = os.path.join(
        base_dir,
        "New_ents",
        f"{species_name}_non_covalent_lassos_4_5_no_knots.npz"
    )

    # Load the NPZ data
    with np.load(npz_path, allow_pickle=True) as npzfile:
        data_new_ents = npzfile["arr_0"].item()

    # Generate the CSV filename
    output_csv = f"{species_name}_experimental_entanglement_list.csv"

    # Write only the first chain of the first subkey for each key
    with open(output_csv, "w") as csv_file:
        for key in data_new_ents:
            for subkey in data_new_ents[key]:
                for chain in data_new_ents[key][subkey]:
                    csv_file.write(f"{key} {subkey} {chain}\n")
                    break  # only allow one chain
                break  # only allow one subkey

    # Generate a simple report file
    report_file = f"{species_name}_run_info.txt"
    with open(report_file, "w") as report:
        report.write("===== RUN INFO =====\n")
        report.write(f"Base directory (base_dir): {base_dir}\n")
        report.write(f"Species name: {species_name}\n")
        report.write(f"NPZ data path: {npz_path}\n\n")
        report.write("===== OUTPUT =====\n")
        report.write(f"Output CSV file: {output_csv}\n")
        report.write(f"Report file: {report_file}\n\n")
        report.write("===== STATUS =====\n")
        report.write("Script completed successfully.\n")

    # Print a console message
    print(f"CSV saved to: {output_csv}")
    print(f"Report file saved to: {report_file}")

if __name__ == "__main__":
    main()

