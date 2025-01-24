#!/usr/bin/env python3
"""
Propensity Score Matching (PSM) pipeline using Python + rpy2 + R's "MatchIt":

Steps:
  1) Load three DataFrames (e.g., E. coli, human, yeast) from specified pickle files.
  2) Merge them into one DataFrame after:
     - Computing a median 'sasa_median' (if desired),
     - Selecting [sasa_median, mapped_target] and an overlap list of features.
  3) Write this "unmatched" combined DataFrame to a CSV.
  4) Call R's "MatchIt" via rpy2 to perform PSM using nearest neighbor on 'mapped_target ~ sasa_median' (configurable).
  5) Extract the matched data back to Python, write it to a second CSV.
  6) Perform permutation tests (50000 resamples) on specified features between matched_target==1 vs. matched_target==0,
     storing p-values in a final CSV.

Usage Example:
    python psm_pipeline.py \
      --ecoli-file /path/to/ecoli.pkl \
      --human-file /path/to/human.pkl \
      --yeast-file /path/to/yeast.pkl \
      --unmatched-csv unmatched_allspecies.csv \
      --matched-csv matched_allspecies.csv \
      --results-csv psm_results.csv \
      --overlap-cols D_0_pssm ACH3_pssm CN_exp Theta_exp Tau_exp psi rsaa SS7__Strand G_0_pssm \
      --sasa-prefix sasa_ \
      --psm-distance glm \
      --psm-method nearest \
      --psm-ratio 1 \
      --psm-caliper 0.2 \
      --num-resamples 50000

Adjust arguments accordingly. All file paths and feature lists are user-defined, so there's no hard-coding.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import permutation_test

# rpy2 / R bridging
import rpy2.robjects as ro
from rpy2.robjects import globalenv
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri

# Confirm required R libraries are installed (MatchIt, etc.)
# The user is responsible for ensuring these are installed in their R environment.
tableone = rpackages.importr('tableone')
matchit  = rpackages.importr('MatchIt')
lmtest   = rpackages.importr('lmtest')
sandwich = rpackages.importr('sandwich')

def main():
    parser = argparse.ArgumentParser(
        description="Propensity score matching pipeline with Python & rpy2 + R MatchIt."
    )

    # Input files
    parser.add_argument("--ecoli-file", required=True, help="Pickle file for E. coli DataFrame.")
    parser.add_argument("--human-file", required=True, help="Pickle file for human DataFrame.")
    parser.add_argument("--yeast-file", required=True, help="Pickle file for yeast DataFrame.")

    # Output files
    parser.add_argument("--unmatched-csv", default="unmatched_data.csv",
                        help="Where to write the combined unmatched DataFrame.")
    parser.add_argument("--matched-csv", default="matched_data.csv",
                        help="Where to write the matched data from R's MatchIt.")
    parser.add_argument("--results-csv", default="psm_results.csv",
                        help="Where to write the final permutation test results.")

    # Overlap features & PSM config
    parser.add_argument("--overlap-cols", nargs="+", required=True,
                        help="List of overlap feature columns to keep and test in the final step.")
    parser.add_argument("--mapped-target-col", default="mapped_target",
                        help="Name of the target column in each DataFrame. Default='mapped_target'.")
    parser.add_argument("--sasa-prefix", default="sasa_",
                        help="Prefix to identify 'sasa' columns for median calculation.")
    parser.add_argument("--distance-col", default="sasa_median",
                        help="Column on which we run PSM (e.g., 'sasa_median').")

    # MatchIt parameters
    parser.add_argument("--psm-distance", default="glm",
                        help="Distance method for MatchIt (default=glm).")
    parser.add_argument("--psm-method", default="nearest",
                        help="MatchIt method (default=nearest).")
    parser.add_argument("--psm-ratio", type=int, default=1,
                        help="Ratio for matching. Default=1.")
    parser.add_argument("--psm-replace", action="store_true",
                        help="Whether to allow matching with replacement. Default=False.")
    parser.add_argument("--psm-caliper", type=float, default=0.2,
                        help="Caliper for nearest neighbor. Default=0.2.")

    # Permutation test config
    parser.add_argument("--num-resamples", type=int, default=50000,
                        help="Number of resamples in the permutation test. Default=50000.")
    parser.add_argument("--random-seed", type=int, default=0,
                        help="Random seed for permutation test. Default=0.")

    args = parser.parse_args()

    # 1) Load pickles for E. coli, human, yeast
    df_ecoli = pd.read_pickle(args.ecoli_file)
    df_human = pd.read_pickle(args.human_file)
    df_yeast = pd.read_pickle(args.yeast_file)

    # 2) For each DataFrame, compute median of columns matching prefix 'sasa_' (excluding last col if needed).
    #    We'll do a general approach: Filter columns that start with `sasa_prefix`.
    for df in [df_ecoli, df_human, df_yeast]:
        sasa_cols = [c for c in df.columns if c.startswith(args.sasa_prefix)]
        if len(sasa_cols) > 0:
            # we just compute the median across all such columns
            df["sasa_median"] = df[sasa_cols].median(axis=1)
        else:
            df["sasa_median"] = np.nan  # no sasa columns found

    # 3) Subset each DataFrame to [distance_col, mapped_target_col] + overlap_cols
    keep_cols = [args.distance_col, args.mapped_target_col] + args.overlap_cols
    df_ecoli = df_ecoli[keep_cols].copy()
    df_human = df_human[keep_cols].copy()
    df_yeast = df_yeast[keep_cols].copy()

    # 4) Concatenate all species
    df_all = pd.concat([df_ecoli, df_human, df_yeast], ignore_index=True)
    df_all.to_csv(args.unmatched_csv, index=False)
    print(f"Unmatched data saved to: {args.unmatched_csv}")

    # 5) Convert Python DataFrame -> R DataFrame
    with ro.default_converter + pandas2ri.converter:
        dfr = ro.conversion.py2rpy(df_all)

    # Put it in R global env
    globalenv["dfr"] = dfr

    # 6) Build the R command string for matchit, e.g.:
    #    matchit(mapped_target ~ sasa_median, method='nearest', distance='glm', ratio=1,
    #            replace=F, caliper=0.2, data=dfr)
    # We'll make it dynamic
    rcmd = f"""
      library(MatchIt)
      psm_matchit <- matchit(
         {args.mapped_target_col} ~ {args.distance_col},
         method="{args.psm_method}",
         distance="{args.psm_distance}",
         ratio={args.psm_ratio},
         replace={str(args.psm_replace).upper()},
         caliper={args.psm_caliper},
         data=dfr
      )
    """
    # Execute the R code
    ro.r(rcmd)

    # Print summary
    print(ro.r('summary(psm_matchit, un=TRUE)'))

    # Extract matched data
    ro.r('psm_matchit_data <- match.data(psm_matchit)')
    psm_matchit_data = ro.r('psm_matchit_data')

    with ro.default_converter + pandas2ri.converter:
        df_matched = ro.conversion.rpy2py(psm_matchit_data)

    # Write matched data
    df_matched.to_csv(args.matched_csv, index=False)
    print(f"Matched data saved to: {args.matched_csv}")

    # 7) Permutation tests on the overlap columns (excluding distance col & target if not needed)
    results = []
    for feat in tqdm(args.overlap_cols, desc="Permutation tests"):
        group1 = df_matched.loc[df_matched[args.mapped_target_col] == 1, feat]
        group0 = df_matched.loc[df_matched[args.mapped_target_col] == 0, feat]

        # SciPy >=1.9:
        #   permutation_test((group1, group0), statistic, ..., n_resamples=?, random_state=?, etc.
        # We measure difference of means (two-sided).
        # We'll replicate absolute difference by using 'alternative="two-sided"' & statistic= difference
        # but we must interpret. If you want the absolute difference, you can do something like:
        #   statistic=lambda x, y: abs(np.mean(x) - np.mean(y))
        # For a two-sided test, we can do difference in means but SciPy handles the sign logic.
        res = permutation_test(
            data=(group1, group0),
            statistic=lambda x, y: np.mean(x) - np.mean(y),
            alternative="two-sided",
            permutation_type="independent",
            n_resamples=args.num_resamples,
            random_state=args.random_seed
        )
        pval = res.pvalue
        results.append((feat, pval))

    results_df = pd.DataFrame(results, columns=["feature", "permutation_pvalue"])
    results_df.to_csv(args.results_csv, index=False)
    print(f"Permutation test results saved to: {args.results_csv}")
    print("Done.")

if __name__ == "__main__":
    sys.exit(main())

