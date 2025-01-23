#!/usr/bin/env python3

import os
import sys
import ast
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import cdist
from MDAnalysis import Universe
warnings.filterwarnings("ignore")

########################
#  HELPER FUNCTIONS    #
########################

def converttoint(inputvect):
    """
    Converts the elements of an input vector to ints (absolute values).
    """
    return [abs(int(i)) for i in inputvect]

def cleanstring(inputstring):
    """
    Converts an input string to a comma-separated list of strings.
    Example: "['23' , '42']" -> ['23','42']
    """
    return inputstring[2:-2].replace(" ,",",").replace(", ",",").replace("'", "").split(',')

def find_res_indices(residue_array, searcharray):
    """
    Given an array of residue IDs (residue_array) and a list of target
    residue IDs (searcharray), returns their indices (if present) 
    in the residue_array.
    """
    return_list = [np.where(residue_array == i)[0] for i in searcharray]
    # Flatten the list of arrays
    return np.array([idx for arr in return_list for idx in arr])

def create_resid_selection_query(resid_list):
    """
    Create a selection query from a list of residue IDs, merging consecutive IDs into a range.
    Example: [10, 11, 12, 14] -> "resnum 10:12 or resnum 14"
    """
    if not resid_list.size:
        return ""

    resid_list = sorted(set(resid_list))
    ranges = []
    start = resid_list[0]
    end = start

    for resid in resid_list[1:]:
        if resid == end + 1:
            end = resid
        else:
            ranges.append((start, end))
            start = end = resid
    ranges.append((start, end))

    query_parts = [
        f"resnum {s}:{e}" if s != e else f"resnum {s}"
        for s, e in ranges
    ]
    return " or ".join(query_parts)

def find_nearby_residues(structure, target_resid):
    """
    Finds all residues with heavy atoms within 4.5 Å of a given residue (target_resid).
    """
    threshold_distance = 4.5
    nearby_residues = []

    # Select atoms of the target residue
    atoms_target = structure.select_atoms(f"resnum {target_resid}")

    # Compare against all other residues
    for residue in structure.residues:
        other_resid = residue.resid
        if other_resid == target_resid:
            continue
        atoms_other = structure.select_atoms(f"resnum {other_resid}")
        if len(atoms_other) == 0:
            continue  # skip if no atoms found
        distances = cdist(atoms_target.positions, atoms_other.positions)
        min_distance = np.min(distances)
        if min_distance <= threshold_distance:
            nearby_residues.append(other_resid)

    return nearby_residues

########################
#  CORE LOGIC          #
########################

def parse_clustered_files(
    input_file, 
    cge_dir, 
    alphafold_flag=False
):
    """
    Parses the 'clustered GE' files and returns a dictionary mapping:
      (gene, protein, chain, native_contact_i, native_contact_j) -> [list of crossing residues]
    If alphafold_flag is True, the tuple becomes:
      (gene, chain, nc_i, nc_j) -> ...
    """
    full_rep_ent_data = defaultdict(list)

    # Each line in input_file might be "gene pdb chain" or just "uniprot_id" if alphafold
    rep_protein_data = np.loadtxt(input_file, dtype=str, delimiter="\n")

    for each_rep_protein in rep_protein_data:
        if alphafold_flag:
            # Only uniprot_id is given
            rep_gene = each_rep_protein
            rep_chain = "A"
            rep_protein = None
        else:
            rep_gene, rep_protein, rep_chain = each_rep_protein.split()

        # The file that has "clustered GE" data
        # e.g. <gene>_clustered_GE.txt
        # This file has lines like:
        # "Chain A | (native_contact_i, native_contact_j, crossing...) | "
        cge_file = os.path.join(cge_dir, f"{rep_gene}_clustered_GE.txt")
        rep_ent_data = pd.read_csv(cge_file, delimiter="|", header=None)

        # Column 1 in rep_ent_data holds the array of int conversions
        rep_ent_data[1] = rep_ent_data[1].apply(
            lambda x: converttoint(cleanstring(x))
        )

        # Check each row in the "clustered GE" file
        for index, line in rep_ent_data.iterrows():
            # Example: line[0] = "Chain A "
            #          line[1] = [nc_i, nc_j, crossing, crossing, ...]
            chain_str = line[0].split()
            if len(chain_str) < 2:
                continue

            chain_label = chain_str[1]
            if chain_label == rep_chain:
                native_contact_i = line[1][0]
                native_contact_j = line[1][1]
                crossing_resid = list(set(line[1][2:]))

                if alphafold_flag:
                    full_rep_ent_data[
                        (rep_gene, rep_chain, native_contact_i, native_contact_j)
                    ].append(crossing_resid)
                else:
                    full_rep_ent_data[
                        (rep_gene, rep_protein, rep_chain, native_contact_i, native_contact_j)
                    ].append(crossing_resid)

    return full_rep_ent_data

def create_entangled_definatons(
    uniprotid, 
    PDB, 
    chain, 
    nc_i, 
    nc_j, 
    crossing, 
    pdb_dir, 
    alphafold_flag=False
):
    """
    Build definition sets for entanglement classes:
        0 -> Not involved
        1 -> native contact forming minimal loop
        2 -> loop residue within 4.5 Å of crossing residue
        3 -> crossing residue
        4 -> thread residue within 4.5 Å of crossing residue
    """

    # Load CA atoms for the entire chain
    if alphafold_flag:
        # e.g., "uniprotid.pdb"
        structure_ca = Universe(
            os.path.join(pdb_dir, f"{uniprotid}.pdb"), format="PDB"
        ).select_atoms(f"name CA and segid {chain}")
    else:
        # e.g., "gene_protein.pdb1"
        structure_ca = Universe(
            os.path.join(pdb_dir, f"{uniprotid}_{PDB}.pdb1"), format="PDB"
        ).select_atoms(f"name CA and segid {chain}")

    # Get a list of residue IDs for those CA atoms
    resids, i = np.unique(structure_ca.resids, return_index=True)

    # Now select heavy atoms for those same residues (for distance checks)
    residue_selection = create_resid_selection_query(resids)
    if alphafold_flag:
        # e.g., "uniprotid.pdb"
        structure = Universe(
            os.path.join(pdb_dir, f"{uniprotid}.pdb"), format="PDB"
        ).select_atoms(f"({residue_selection}) and segid {chain} and not name H*")
    else:
        # e.g., "gene_protein.pdb1"
        structure = Universe(
            os.path.join(pdb_dir, f"{uniprotid}_{PDB}.pdb1"), format="PDB"
        ).select_atoms(f"({residue_selection}) and segid {chain} and not name H*")

    # Identify loop vs thread
    loop_resids = np.arange(nc_i, nc_j + 1, dtype=int)
    if crossing > nc_j:
        # crossing is on the C terminus side
        thread_resids = np.arange(nc_j + 1, resids[-1] + 1, dtype=int)
    elif crossing < nc_i:
        # crossing is on the N terminus side
        thread_resids = np.arange(resids[0], nc_i, dtype=int)
    else:
        # If crossing is within the loop, define an empty thread
        thread_resids = np.array([], dtype=int)

    # Mark native contacts (1) 
    Ent_def_1 = [nc_i, nc_j]

    # crossing residue is always 3
    Ent_def_3 = [crossing]

    # All residues within 4.5 Å of crossing are in spherical_resids
    spherical_resids = find_nearby_residues(structure, crossing)

    # loop residues within 4.5 Å get definition 2
    Ent_def_2 = np.intersect1d(spherical_resids, loop_resids)
    Ent_def_2 = Ent_def_2[Ent_def_2 != nc_i]
    Ent_def_2 = Ent_def_2[Ent_def_2 != nc_j]

    # thread residues within 4.5 Å get definition 4
    Ent_def_4 = np.intersect1d(spherical_resids, thread_resids)
    Ent_def_4 = Ent_def_4[Ent_def_4 != nc_i]
    Ent_def_4 = Ent_def_4[Ent_def_4 != nc_j]

    # Collect all residues involved
    all_involved_resids = np.concatenate(
        (Ent_def_1, Ent_def_2, Ent_def_3, Ent_def_4),
        dtype=int
    )

    # Everything else is 0
    Ent_def_0 = np.setdiff1d(resids, all_involved_resids)

    return Ent_def_0, Ent_def_1, Ent_def_2, Ent_def_3, Ent_def_4

def assemble_entanglement_definations(
    full_rep_ent_data, 
    pdb_dir, 
    output_dir,
    alphafold_flag=False
):
    """
    Builds an integer array (entanglement definition) for each residue in the PDB,
    assigning categories 0-4 as described in the docstring of create_entangled_definatons.
    Saves arrays in 'unmapped_entanglement_definatons/<gene>_<protein>_<chain>_(nc_i,nc_j,[crossings]).npz'
    or an equivalent naming scheme for alphafold.
    """
    os.makedirs(output_dir, exist_ok=True)

    for each_key, crossings_list in full_rep_ent_data.items():
        # Example of each_key if alphafold=False:
        #   (gene, protein, chain, nc_i, nc_j)
        # If alphafold=True:
        #   (gene, chain, nc_i, nc_j)
        # crossings_list might be [[crossing1, crossing2, ...]]

        if alphafold_flag:
            # each_key = (gene, chain, nc_i, nc_j)
            gene, chain, nc_i, nc_j = each_key
            protein = None
        else:
            # each_key = (gene, protein, chain, nc_i, nc_j)
            gene, protein, chain, nc_i, nc_j = each_key

        # The original code uses only the first list of crossings
        # In many cases, .append() was used, but only the first is processed
        crossings = crossings_list[0]
        combinded_information = (nc_i, nc_j, crossings)
        print(crossings)

        # Load CA atoms for the entire chain (to build the final array)
        if alphafold_flag:
            pdb_structure = Universe(
                os.path.join(pdb_dir, f"{gene}.pdb"), format="PDB"
            ).select_atoms(f"name CA and segid {chain}")
        else:
            pdb_structure = Universe(
                os.path.join(pdb_dir, f"{gene}_{protein}.pdb1"), format="PDB"
            ).select_atoms(f"name CA and segid {chain}")

        pdb_resids, pdb_i = np.unique(pdb_structure.resids, return_index=True)
        rep_pdb_size = len(pdb_resids)
        rep_ent_arr = np.array([-1] * rep_pdb_size)

        # We need to track a combined definition for crossing residues
        # Some definitions must be intersected or unioned across all crossing residues.
        Ent_0_temp = []
        Ent_2 = set()
        Ent_4 = set()

        # For each crossing residue, compute the definitions
        for cr in crossings:
            def_0, def_1, def_2, def_3, def_4 = create_entangled_definatons(
                uniprotid=gene,
                PDB=protein,
                chain=chain,
                nc_i=nc_i,
                nc_j=nc_j,
                crossing=cr,
                pdb_dir=pdb_dir,
                alphafold_flag=alphafold_flag
            )

            # Convert actual residue IDs to indices in pdb_resids
            def_0 = find_res_indices(pdb_resids, def_0)
            def_1 = find_res_indices(pdb_resids, def_1)
            def_2 = find_res_indices(pdb_resids, def_2)
            def_3 = find_res_indices(pdb_resids, def_3)
            def_4 = find_res_indices(pdb_resids, def_4)

            Ent_0_temp.append(def_0)
            Ent_2 |= set(def_2)
            Ent_4 |= set(def_4)

        # Definition 0: intersection across all crossing residues
        Ent_0 = set.intersection(*map(set, Ent_0_temp)) if Ent_0_temp else set()

        # Update the rep_ent_arr
        for idx0 in Ent_0:
            rep_ent_arr[idx0] = 0
        for idx2 in Ent_2:
            rep_ent_arr[idx2] = 2
        for idx4 in Ent_4:
            rep_ent_arr[idx4] = 4

        # Mark the native contact indices (1)
        adjusted_nc_i = find_res_indices(pdb_resids, [nc_i])[0]
        adjusted_nc_j = find_res_indices(pdb_resids, [nc_j])[0]
        rep_ent_arr[adjusted_nc_i] = 1
        rep_ent_arr[adjusted_nc_j] = 1

        # Mark the crossing residues (3)
        for crossing in crossings:
            adjusted_crossing = find_res_indices(pdb_resids, [crossing])[0]
            rep_ent_arr[adjusted_crossing] = 3

        # Print summary to console
        if alphafold_flag:
            print(gene, chain, combinded_information)
            save_filename = f"{gene}_{chain}_{combinded_information}"
        else:
            print(gene, protein, chain, combinded_information)
            save_filename = f"{gene}_{protein}_{chain}_{combinded_information}"

        # Save the array
        np.savez(os.path.join(output_dir, save_filename), rep_ent_arr)

def qc_unmapped_entanglements(output_dir):
    """
    Performs a quick check (QC) of the entanglement .npz files in output_dir
    to ensure definitions 0,1,2,3,4 appear consistent.
    """
    for filename in os.listdir(output_dir):
        if not filename.endswith(".npz"):
            continue

        filepath = os.path.join(output_dir, filename)
        # Example filename: gene_protein_chain_(nc_i, nc_j, [crossings]).npz
        # or gene_chain_(nc_i, nc_j, [crossings]).npz (alphafold)

        # Try to parse out the crossing info (the last underscore piece)
        # e.g. "gene_protein_chain_(12,34,[45,46])"
        base_name = filename.rsplit(".", 1)[0]  # remove .npz

        # everything after the last underscore
        # might be e.g. "(nc_i,nc_j,[crossings])"
        # caution: if gene/protein contain underscores themselves
        # The original code assumes the pattern is fixed; we replicate that logic:
        parts = base_name.split("_")
        ent_part = parts[-1]  # e.g. "(12,34,[45,46])"

        # Evaluate the tuple
        try:
            # example of ent_part -> "(12, 34, [45,46])"
            ent_tuple = ast.literal_eval(ent_part)
        except Exception:
            # If parsing fails for some reason, skip or print an error
            print("QC parse error:", filename)
            continue

        # ent_tuple is (nc_i, nc_j, [crossing, crossing, ...])
        if len(ent_tuple) != 3:
            print("QC parse structure mismatch:", filename)
            continue

        # The array itself
        rep_ent_arr = np.load(filepath)["arr_0"]
        size_crossings = len(ent_tuple[2])

        # Basic checks
        if 3 not in rep_ent_arr or 0 not in rep_ent_arr or 1 not in rep_ent_arr:
            print("Missing category (0,1,3) in:", filename)

        if len(np.where(rep_ent_arr == 3)[0]) != size_crossings:
            print("Number of crossing residues mismatch in:", filename)

        if len(np.where(rep_ent_arr == 1)[0]) != 2:
            print("Expected exactly two native contact residues in:", filename)

        # Check for overlap among definitions 0-4
        def_0 = set(np.where(rep_ent_arr == 0)[0])
        def_1 = set(np.where(rep_ent_arr == 1)[0])
        def_2 = set(np.where(rep_ent_arr == 2)[0])
        def_3 = set(np.where(rep_ent_arr == 3)[0])
        def_4 = set(np.where(rep_ent_arr == 4)[0])

        overlaps = (
            def_0 & def_1 or
            def_0 & def_2 or
            def_0 & def_3 or
            def_0 & def_4 or
            def_1 & def_3 or
            def_1 & def_4 or
            def_2 & def_3 or
            def_2 & def_4 or
            def_3 & def_4
        )
        if overlaps:
            print("Overlapping definitions in:", filename)

########################
#  MAIN DRIVER         #
########################

def run_entanglement_mapping(
    input_file, 
    pdb_dir, 
    cge_dir,
    output_dir="unmapped_entanglement_definatons",
    alphafold_flag=False,
    run_qc=False
):
    """
    Main driver function:
      1) Parses the 'clustered GE' data with parse_clustered_files.
      2) Assembles entanglement definitions and saves them to output_dir.
      3) Optionally performs QC checks.
      4) Returns a list of files created, plus any QC messages.
    """

    # 1) parse data
    full_rep_ent_data = parse_clustered_files(
        input_file, cge_dir, alphafold_flag
    )

    # 2) assemble entanglement definitions
    assemble_entanglement_definations(
        full_rep_ent_data=full_rep_ent_data,
        pdb_dir=pdb_dir,
        output_dir=output_dir,
        alphafold_flag=alphafold_flag
    )

    # 3) optional QC
    if run_qc:
        qc_unmapped_entanglements(output_dir)

########################
#  CLI INTERFACE       #
########################

def main():
    """
    Command-line entry point:
      usage: python script.py <INPUT_FILE> <PDB_DIRECTORY> <CGE_DIRECTORY> [OUTPUT_DIR] [--alphafold] [--qc]
    """

    # Basic usage check
    if len(sys.argv) < 4:
        script_name = os.path.basename(__file__)
        print(f"Usage: python {script_name} <INPUT_FILE> <PDB_DIRECTORY> <CGE_DIRECTORY> [OUTPUT_DIR] [--alphafold] [--qc]")
        sys.exit(1)

    input_file = sys.argv[1]
    pdb_dir = sys.argv[2]
    cge_dir = sys.argv[3]
    output_dir = "unmapped_entanglement_definatons"
    alphafold_flag = False
    run_qc = False

    # Collect any optional arguments
    # e.g., user might pass: python script.py input.txt /pdb /cge out_dir --alphafold --qc
    optional_args = sys.argv[4:]
    # Look for flags or a possible custom output_dir
    for arg in optional_args:
        if arg == "--alphafold":
            alphafold_flag = True
        elif arg == "--qc":
            run_qc = True
        elif not arg.startswith("--"):
            # If it's not a --flag, treat it as the output_dir
            output_dir = arg

    # Ensure output_dir exists (or create it)
    os.makedirs(output_dir, exist_ok=True)

    # Run the main entanglement logic
    run_entanglement_mapping(
        input_file=input_file,
        pdb_dir=pdb_dir,
        cge_dir=cge_dir,
        output_dir=output_dir,
        alphafold_flag=alphafold_flag,
        run_qc=run_qc
    )

    # Create a run-info file summarizing
    run_info_path = os.path.join(output_dir, "run_info.txt")
    with open(run_info_path, "w") as report:
        report.write("===== RUN INFO =====\n")
        report.write(f"Input file: {input_file}\n")
        report.write(f"PDB directory: {pdb_dir}\n")
        report.write(f"Clustered GE directory: {cge_dir}\n")
        report.write(f"Output directory: {output_dir}\n")
        report.write(f"AlphaFold flag: {alphafold_flag}\n")
        report.write(f"QC run: {run_qc}\n\n")
        report.write("===== STATUS =====\n")
        report.write("Script completed successfully.\n")

    print(f"Run info written to: {run_info_path}")
    print("Script completed successfully.")

if __name__ == "__main__":
    main()

