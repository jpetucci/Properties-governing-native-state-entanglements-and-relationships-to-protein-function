#!/usr/bin/env python3
"""
A single Python script that offers four sub-commands for working with PSSM-related data:

  1) find_missing_pssms
     Checks a CSV file for gene IDs, compares them to a set of existing PSSM files,
     and prints lines from the CSV corresponding to missing PSSMs.

  2) split_file
     Splits a CSV file into a specified number of chunks (default=10).

  3) collect_pssms
     Copies all files named 'protein.pssm.*' from any "run_*" subdirectory into a folder named "pssm_data".

  4) process_pssms
     Reads each 'protein.pssm.*' file from a specified directory, strips unwanted header/footer lines,
     extracts certain columns, and writes them as CSV files in a "processed" directory.

Usage Examples:
  - Find missing PSSMs:
      python pssm_toolkit.py find_missing_pssms \
          --csv my_data.csv \
          --pssm-dir /path/to/pssm_dir

  - Split a CSV file into 10 parts:
      python pssm_toolkit.py split_file \
          --csv my_data.csv \
          --num-splits 10

  - Collect PSSMs into a "pssm_data" directory:
      python pssm_toolkit.py collect_pssms

  - Process PSSMs (skip lines, select columns) into CSVs under "processed/":
      python pssm_toolkit.py process_pssms \
          --pssm-dir pssm_data
"""

import sys
import os
import glob
import shutil
import argparse


def find_missing_pssms(csv_file: str, pssm_dir: str):
    """
    1) Builds a set of gene IDs from filenames in pssm_dir,
       splitting each filename on '_', taking the first segment as the gene ID.
    2) Reads each line of the CSV file, checks if the second column (gene ID) is missing,
       and prints the entire line if so.
    """
    existing_geneids = set()
    for fname in os.listdir(pssm_dir):
        parts = fname.split("_")
        if parts:
            existing_geneids.add(parts[0])

    with open(csv_file, "r") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            cols = line_stripped.split(",")
            if len(cols) < 2:
                continue
            geneid = cols[1]
            if geneid not in existing_geneids:
                print(line, end="")  # Keep original newline


def split_file(csv_file: str, num_splits: int = 10):
    """
    Splits a CSV file into a specified number of chunks, named with two-letter suffixes.
    Example: my_data.csv_splitaa, my_data.csv_splitab, etc.
    """
    with open(csv_file, "r") as f:
        lines = f.readlines()

    total_lines = len(lines)
    chunk_size = total_lines // num_splits

    # Generate suffixes like 'aa', 'ab', 'ac', ...
    def suffix_generator():
        import string
        letters = string.ascii_lowercase
        for c1 in letters:
            for c2 in letters:
                yield c1 + c2

    suffix_iter = suffix_generator()
    start_idx = 0
    for _ in range(num_splits - 1):
        out_lines = lines[start_idx:start_idx + chunk_size]
        start_idx += chunk_size
        suffix = next(suffix_iter)
        out_name = f"{csv_file}_split{suffix}"
        with open(out_name, "w") as out_f:
            out_f.writelines(out_lines)

    # Last chunk (includes any remainder)
    suffix = next(suffix_iter)
    out_name = f"{csv_file}_split{suffix}"
    with open(out_name, "w") as out_f:
        out_f.writelines(lines[start_idx:])


def collect_pssms():
    """
    Creates a 'pssm_data' directory (if it doesn't exist),
    and copies any file named 'protein.pssm.*' from 'run_*' subdirectories into it.
    """
    if not os.path.exists("pssm_data"):
        os.mkdir("pssm_data")

    for file_path in glob.glob(os.path.join(".", "run_*", "protein.pssm.*")):
        shutil.copy2(file_path, "pssm_data")
    print("Copied all protein.pssm.* files from run_* subdirectories into ./pssm_data.")


def process_pssms(pssm_dir: str):
    """
    Looks for files named 'protein.pssm.*' in the given directory.
    For each file:
      - Skip the first 3 lines, skip the last 6 lines.
      - Extract columns 2..22 (space-delimited).
      - Write them as CSV to './processed/<uniqid>_pssm_processed'.
    """
    if not os.path.exists("processed"):
        os.mkdir("processed")

    all_files = sorted(os.listdir(pssm_dir))
    for fname in all_files:
        if "protein.pssm." in fname:
            parts = fname.split(".")
            if len(parts) < 3:
                continue
            uniqid = parts[2]
            in_path = os.path.join(pssm_dir, fname)

            with open(in_path, "r") as fin:
                lines = fin.readlines()

            # Skip the first 3 lines, skip the last 6 lines
            lines = lines[3:-6]

            out_path = os.path.join("processed", f"{uniqid}_pssm_processed")
            with open(out_path, "w") as fout:
                for line in lines:
                    cols = line.strip().split()
                    if len(cols) < 22:
                        continue
                    subset = cols[1:22]  # columns 2..22
                    fout.write(",".join(subset) + "\n")

    print("Processing complete. Check the 'processed' directory for output files.")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-function PSSM toolkit with sub-commands."
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Sub-command to run")

    # find_missing_pssms
    parser_missing = subparsers.add_parser("find_missing_pssms",
                                          help="Identify which gene IDs from a CSV are missing PSSM files.")
    parser_missing.add_argument("--csv", required=True,
                               help="Path to CSV file (comma-delimited) with gene IDs in the second column.")
    parser_missing.add_argument("--pssm-dir", required=True,
                                help="Directory containing PSSM files.")

    # split_file
    parser_split = subparsers.add_parser("split_file",
                                         help="Split a CSV into multiple chunks.")
    parser_split.add_argument("--csv", required=True,
                              help="Path to the CSV file to split.")
    parser_split.add_argument("--num-splits", type=int, default=10,
                              help="Number of chunks (default=10).")

    # collect_pssms
    subparsers.add_parser("collect_pssms",
                          help="Gather 'protein.pssm.*' files from run_* directories into 'pssm_data'.")

    # process_pssms
    parser_process = subparsers.add_parser("process_pssms",
                                           help="Process 'protein.pssm.*' files into CSV columns 2..22.")
    parser_process.add_argument("--pssm-dir", required=True,
                                help="Directory containing 'protein.pssm.*' files.")

    args = parser.parse_args()

    if args.subcommand == "find_missing_pssms":
        find_missing_pssms(args.csv, args.pssm_dir)
    elif args.subcommand == "split_file":
        split_file(args.csv, args.num_splits)
    elif args.subcommand == "collect_pssms":
        collect_pssms()
    elif args.subcommand == "process_pssms":
        process_pssms(args.pssm_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

