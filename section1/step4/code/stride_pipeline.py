#!/usr/bin/env python3

"""
Replicate the functionality of the three Bash scripts:
1) copypdbs.sh
2) run_stride2
3) process.sh

Usage examples:

  # 1) Copy PDB files and modify 'HETATM ' -> 'ATOM   '
  python stride_pipeline.py copy \
      --input-file path/to/inputfile.txt \
      --pdb-dir path/to/original_pdbs \
      --output-dir path/to/copied_pdbs

  # 2) Run STRIDE on each line of the input file
  python stride_pipeline.py run_stride \
      --input-file path/to/inputfile.txt \
      --pdb-dir path/to/copied_pdbs \
      --output-dir path/to/stride_outputs \
      --stride-exe /absolute/path/to/stride

  # 3) Process the stride output
  python stride_pipeline.py process \
      --stride-dir path/to/stride_outputs \
      --output-dir path/to/processed
"""

import os
import sys
import argparse
import shutil
import subprocess

def copy_pdbs(input_file, pdb_dir, output_dir):
    """
    Equivalent of copypdbs.sh:
      - Reads 'geneid pdbid' from input_file
      - Copies PDB from pdb_dir to output_dir
      - Replaces 'HETATM ' with 'ATOM   ' in-place
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expecting lines like: geneid pdbid chain
            # Script #1 only uses gene + pdb -> <gene>_<pdb>.pdb1
            parts = line.split()
            if len(parts) < 2:
                continue
            geneid, pdbid = parts[0], parts[1]
            name = f"{geneid}_{pdbid}"
            source_pdb = os.path.join(pdb_dir, f"{name}.pdb1")
            dest_pdb = os.path.join(output_dir, f"{name}.pdb1")

            if not os.path.exists(source_pdb):
                print(f"Warning: {source_pdb} does not exist. Skipping.")
                continue

            # Copy file
            shutil.copy2(source_pdb, dest_pdb)

            # Replace "HETATM " with "ATOM   "
            # This preserves spacing exactly as sed 's/HETATM /ATOM   /g'
            with open(dest_pdb, "r+") as pdb_file:
                content = pdb_file.read()
                content_modified = content.replace("HETATM ", "ATOM   ")
                pdb_file.seek(0)
                pdb_file.write(content_modified)
                pdb_file.truncate()


def run_stride(input_file, pdb_dir, output_dir, stride_exe):
    """
    Equivalent of run_stride2:
      - Reads 'geneid pdbid chain' from input_file
      - Calls stride -c<chain> on each <gene>_<pdb>.pdb1
      - Saves output to stride_<gene>_<pdb>.txt in output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            geneid, pdbid, chain = parts[0], parts[1], parts[2]
            name = f"{geneid}_{pdbid}"
            input_pdb = os.path.join(pdb_dir, f"{name}.pdb1")
            out_txt = os.path.join(output_dir, f"stride_{name}.txt")

            if not os.path.exists(input_pdb):
                print(f"Warning: {input_pdb} not found. Skipping stride.")
                continue

            print(name)
            cmd = [stride_exe, f"-c{chain}", input_pdb]
            # Print the command (as in the original bash script)
            print(" ".join(cmd), ">", out_txt)

            # Run the command
            with open(out_txt, 'w') as fout:
                subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, text=True)


def process_stride(stride_dir, output_dir):
    """
    Equivalent of process.sh:
      - For each .txt in stride_dir, grep lines that start with "ASG "
      - Then print columns 2..10 (9 columns total) as CSV
      - Output goes to 'processed/processed_<filename>.txt'
    """
    os.makedirs(output_dir, exist_ok=True)

    # For file in *.txt
    txt_files = [f for f in os.listdir(stride_dir) if f.endswith(".txt")]
    for txt_file in txt_files:
        in_path = os.path.join(stride_dir, txt_file)
        out_path = os.path.join(output_dir, f"processed_{txt_file}")

        lines_out = []
        with open(in_path, 'r') as fin:
            for line in fin:
                # Only lines starting with "ASG "
                if line.startswith("ASG "):
                    # Original AWK: '{print $2 "," $3 "," ... $10}'
                    # That means we skip $1 (the "ASG"), and output columns 2..10
                    parts = line.strip().split()
                    # Make sure we have enough columns
                    # If the line is "ASG X X X X X X X X X ...", columns are:
                    #   0=ASG, 1=col2, 2=col3, ... so we want parts[1]..parts[9]
                    if len(parts) >= 10:
                        cols = parts[1:10]  # 9 columns total
                        lines_out.append(",".join(cols))

        # Write to processed_ file
        with open(out_path, 'w') as fout:
            for l in lines_out:
                fout.write(l + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replicate copypdbs.sh, run_stride2, and process.sh in Python."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # copy sub-command
    copy_parser = subparsers.add_parser("copy", help="Replicate copypdbs.sh")
    copy_parser.add_argument("--input-file", required=True, help="Path to input file (e.g. geneid pdbid chain)")
    copy_parser.add_argument("--pdb-dir", required=True, help="Path to directory containing original PDBs")
    copy_parser.add_argument("--output-dir", required=True, help="Where to copy & modify PDB files")

    # run_stride sub-command
    stride_parser = subparsers.add_parser("run_stride", help="Replicate run_stride2")
    stride_parser.add_argument("--input-file", required=True, help="Path to input file (geneid pdbid chain)")
    stride_parser.add_argument("--pdb-dir", required=True, help="Directory containing PDB files to run stride on")
    stride_parser.add_argument("--output-dir", required=True, help="Directory to store stride output files")
    stride_parser.add_argument("--stride-exe", required=True, help="Absolute path to the stride executable")

    # process sub-command
    process_parser = subparsers.add_parser("process", help="Replicate process.sh")
    process_parser.add_argument("--stride-dir", required=True, help="Directory of stride output .txt files")
    process_parser.add_argument("--output-dir", required=True, help="Directory to store processed output")

    args = parser.parse_args()

    if args.command == "copy":
        copy_pdbs(args.input_file, args.pdb_dir, args.output_dir)

    elif args.command == "run_stride":
        run_stride(args.input_file, args.pdb_dir, args.output_dir, args.stride_exe)

    elif args.command == "process":
        process_stride(args.stride_dir, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

