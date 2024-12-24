import os
import pandas as pd
import numpy as np
from IPython.display import display
import multiprocessing as mp

"""
This module contains functions to load and summarize coacquisition data, at log scale intervals.
"""

# suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def load_data(res_dir):
    """
    Load chromosome location data and coacquisition data from specified directories.

    Arguments:
    res_dir (str): The directory containing the coacquisition data files.

    Returns:
    coacquisitions_dfs (dict): A dictionary where keys are method names and values are DataFrames containing coacquisition data.

    The function performs the following steps:
    1. Reads the chromosome location data from a TSV file located in `data_dir`.
    2. Reads multiple coacquisition data files from `res_dir` into a dictionary of DataFrames.
    3. Prints the methods (keys) for which coacquisition data is available in the dictionary.

    Notes:
    - Coacquisition data filenames should be of the form "coacquisitions.method.modifier.float_threshold_value.tsv".
    - The modifier is optional and doesn't contain any periods.
    - The function handles missing values represented as 'NA' or 'nan' in the coacquisition data files.
    """
    coacquisitions_dfs = {
        # key is method name: everything after "coacquisitions." and before ".tsv"
        f.partition("coacquisitions.")[2].rpartition(".tsv")[0]: pd.read_csv(
            f"{res_dir}/{f}",
            sep="\t",
            dtype={
                "recipient_branch": str,
                "nog1": str,
                "nog2": str,
                "gene1": str,
                "gene2": str,
                "pair_distances": float,
                "pair_distances_string": str,
                "source_overlap": str,
                "cotransfers": float,
                "num_genes_between": float,
                "contig:genome_size": str,
                "nog1_transfers": float,
                "nog2_transfers": float,
                "notes": str,
            },
            na_values=["NA", "nan"],
        )
        for f in os.listdir(res_dir)
        if f.startswith("coacquisitions.") and f.endswith(".tsv")
    }
    # print all the methods (keys) for which coacquisitions data is in this coacquisitions_dfs dictionary
    print("The following methods are included in the coacquisitions data:")
    print(list(coacquisitions_dfs.keys()))

    return coacquisitions_dfs


def filter_methods(coacquisitions_dfs):
    # remove methods for which all coacquisitions have no source overlap
    return {
        method: df
        for method, df in coacquisitions_dfs.items()
        if not df["source_overlap"].isna().all()
    }


def calculate_min_transfers(coacquisitions_dfs):
    # calculate the minimum number of transfers between the two NOGs for each coacquisition pair
    for method, cq_df in coacquisitions_dfs.items():
        cq_df["min_transfers"] = cq_df[["nog1_transfers", "nog2_transfers"]].min(axis=1)
    return coacquisitions_dfs


def process_coacq_threshold_df(
    cq_df_at_threshold,
    method,
    coacq_threshold,
    neighbor_genes_between_cutoffs,
    coacq_threshold_index=1.0,
):
    # process the coacq_threshold_df to summarize the coacquisitions
    # find the subset of coacquisitions with available positions
    coaq_positions_available_df = cq_df_at_threshold[
        cq_df_at_threshold["notes"] == "gene_positions_are_available"
    ]
    # count the number of coacquisitions with available positions
    num_cq_with_available_positions = len(coaq_positions_available_df)
    # if there are no coacquisitions with available positions, return None
    if num_cq_with_available_positions == 0:
        print(
            f"No coacquisitions with available positions for {
              method} at threshold: {coacq_threshold} coacquisitions."
        )
        return None

    # initialize a dictionary to store the summary records
    summary_records_dict = {
        "method": method,
        "coacquisitions with known positions": num_cq_with_available_positions,
    }
    # find the subset of coacquisitions with cotransfers
    cotransfers_df = cq_df_at_threshold[cq_df_at_threshold["cotransfers"] > 0]

    # calculate the number of neighbors for each cutoff in neighbor_genes_between_cutoffs
    # (i.e. the num of genes between a pair of coacquired genes to be considered neighbors)
    for cutoff in neighbor_genes_between_cutoffs:
        summary_records_dict.update(
            calculate_neighbors(
                coaq_positions_available_df,
                cutoff,
                method,
                coacq_threshold,
            )
        )

    summary_records_dict.update(
        {
            "cotransfers": cotransfers_df["cotransfers"].sum(),
            "cotransfer percentage": cotransfers_df["cotransfers"].sum()
            * 100.0
            / num_cq_with_available_positions,
            "transfer threshold": coacq_threshold_index,
        }
    )
    return summary_records_dict


def calculate_neighbors(coaq_positions_available_df, cutoff, method, coacq_threshold):
    # Group coacquisitions by contig and genome size
    contigwise_coacquisitions = (
        coaq_positions_available_df.groupby("contig:genome_size").size().to_frame()
    )
    contigwise_coacquisitions = contigwise_coacquisitions.rename(
        columns={0: "coacquisitions"}
    )

    # Extract genome size from the index and filter by minimum genome size
    contigwise_coacquisitions["genome_size"] = (
        contigwise_coacquisitions.index.str.split(":").str[1].astype(int)
    )
    # contigwise_coacquisitions = contigwise_coacquisitions[
    #     contigwise_coacquisitions["genome_size"] >= min_genome_size
    # ]

    # Filter coacquisitions by the number of intervening genes (cutoff)
    neighbors_by_genes_df = coaq_positions_available_df[
        coaq_positions_available_df["num_genes_between"] <= cutoff
    ]
    neighbors_by_genes_df = neighbors_by_genes_df[
        neighbors_by_genes_df["contig:genome_size"].isin(
            contigwise_coacquisitions.index
        )
    ]

    # Filter coacquisitions that have cotransfers
    cotransfer_and_neighbor_genes_df = neighbors_by_genes_df[
        neighbors_by_genes_df["cotransfers"] > 0
    ]

    # Calculate the number of coacquisitions with available positions
    this_method_num_cq_with_available_positions = len(
        coaq_positions_available_df[
            coaq_positions_available_df["contig:genome_size"].isin(
                contigwise_coacquisitions.index
            )
        ]
    )

    # if this_method_num_cq_with_available_positions == 0:
    #     print(
    #         f"No coacquisitions with available positions after filtering for genome size >= {
    #           min_genome_size} for method {method} at threshold {coacq_threshold}"
    #     )
    #     return {}

    if len(cotransfer_and_neighbor_genes_df) == 0:
        len_cotransfer_and_neighbor_genes_df = np.nan
    else:
        len_cotransfer_and_neighbor_genes_df = len(cotransfer_and_neighbor_genes_df)

    # Calculate the expected number of neighboring coacquisitions for each contig
    contigwise_coacquisitions["neighboring_coacquisition_probability"] = (
        2 * (cutoff + 1) / contigwise_coacquisitions["genome_size"]
    )
    # this is basically a binomial distribution with n = #coacquisitions, p = 2*(cutoff+1)/genome_size
    contigwise_coacquisitions["expected_number"] = (
        2
        * (cutoff + 1)
        * contigwise_coacquisitions["coacquisitions"]
        / contigwise_coacquisitions["genome_size"]
    )
    # the expected fraction of neighbors across all contigs is just the sum of the expected number of neighbors in each, divided by the sum of coacquisitions
    expected_fraction_of_neighbors = (
        contigwise_coacquisitions["expected_number"].sum()
        / contigwise_coacquisitions["coacquisitions"].sum()
    )

    # Calculate the number of neighbors by intervening genes
    num_neighbors_by_intervening_genes = float(len(neighbors_by_genes_df))

    # Return a dictionary with the calculated metrics
    return {
        f"neighbors ({cutoff} intervening genes)": num_neighbors_by_intervening_genes,
        f"neighbor (max {cutoff} intervening genes) percentage": (
            num_neighbors_by_intervening_genes
            * 100.0
            / this_method_num_cq_with_available_positions
        ),
        f"cotransfer and neighbor (max {cutoff} intervening genes)": len_cotransfer_and_neighbor_genes_df,
        f"cotransfer and neighbor (max {cutoff} intervening genes) percentage": (
            len_cotransfer_and_neighbor_genes_df
            * 100.0
            / this_method_num_cq_with_available_positions
        ),
        f"expected percentage of neighboring coacquisitions (max {cutoff} intervening genes)": expected_fraction_of_neighbors
        * 100.0,
        f"observed minus expected percentage of neighboring coacquisitions (max {cutoff} intervening genes)": (
            num_neighbors_by_intervening_genes
            * 100.0
            / this_method_num_cq_with_available_positions
        )
        - expected_fraction_of_neighbors * 100.0,
        "coacquisitions with known positions": this_method_num_cq_with_available_positions,
    }


def replace_zeros_with_nan(df):
    # replace zeros with NaNs for columns that should not have zeros
    for method in df["method"].unique():
        for column in df.columns:
            if column in ["method", "threshold"]:
                continue
            if not (df["method"] == method)[df[column] != 0.0].any():
                df.loc[df["method"] == method, column] = np.nan
    return df


def summarize_coacquisitions(
    coacquisitions_dfs,
    neighbor_genes_between_cutoffs,
    min_genome_size,
    min_coacquisitions=100,
    min_neighbors=10,
):
    """
    Summarize coacquisition data at log scale intervals.

    Arguments:
    coacquisitions_dfs (dict): A dictionary where keys are method names and values are DataFrames containing coacquisition data.
    neighbor_genes_between_cutoffs (list): A list of integers representing the number of genes between a pair of coacquired genes to be considered neighbors.
    min_genome_size (int): The minimum genome size to consider for filtering coacquisitions. Coacquisitions with genome sizes less than this value are excluded from the summary.
    min_coacquisitions (int): The minimum number of coacquisitions with known gene positions to consider for summarization.
                                If the number of coacquisitions is less than this value, the threshold is skipped.

    Returns:
    coacquisitions_summary_df (DataFrame): A DataFrame containing the summary records for each threshold, across all thresholds for all methods.
    """

    # summarize coacquisitions for each method and threshold
    coacquisitions_summary_records = []
    for method in sorted(list(coacquisitions_dfs.keys())):

        # get the coacquisitions dataframe for this method
        cq_df = coacquisitions_dfs[method]

        # keep only the coacquisitions with known pair distances
        cq_df = cq_df[cq_df["notes"] == "gene_positions_are_available"]
        if cq_df.shape[0] < min_coacquisitions:
            print(
                f"Method {method} has less than {min_coacquisitions} coacquisitions with known gene positions. Skipping."
            )
            continue

        min_transfers = 0.05  # for any coacquisition (pair of NOGs) to be considered, the minimum number of transfers to a common contig

        # if there is only one threshold, we don't need to calculate the coacq thresholds on log scale
        if cq_df["min_transfers"].nunique() > 1:
            # find the min and max num of coacquisitions for each method, which is the coacq at max and min threshold respectively
            # first the number of coacquisitions at the max threshold, with known pair distances
            min_coacq = cq_df[cq_df["min_transfers"] >= 1.0].shape[0]
            min_coacq = max(min_coacq, min_coacquisitions)

            # then the number of coacquisitions at the min threshold, with known pair distances
            max_coacq = cq_df[cq_df["min_transfers"] >= min_transfers].shape[0]

            # between the min and max coacq_threshold_indices, find which of 1eN, 2eN, 5eN are possible, where N is an integer. These would be our coacq_thresholds
            # we do this because we want to summarize the coacquisitions at log scale intervals
            N = 1
            coacq_thresholds = []
            log_multiples = [1, 2, 5, 10]
            while True:
                for i, multiple in enumerate(log_multiples):
                    if multiple * 10**N >= min_coacq and multiple * 10**N <= max_coacq:
                        coacq_thresholds.append(multiple * 10**N)
                    elif (
                        min_coacq > multiple * 10**N
                        and i != (len(log_multiples) - 1)
                        and (log_multiples[i + 1] + 1) * 10 ** (N) <= max_coacq
                    ):
                        # this means that the range of coacquisitions doesn't contain the multiple * 10**N value
                        # then we just append both the min_coacq and max_coacq values and 5 points in between
                        coacq_thresholds.append(min_coacq)
                        for j in range(1, 6):
                            coacq_thresholds.append(
                                min_coacq + j * (max_coacq - min_coacq) / 5
                            )
                        coacq_thresholds.append(max_coacq)
                N += 1
                # if the next multiple of 10**N is greater than the max_coacq, we don't need to check further on the log scale
                if 10**N > max_coacq:
                    break

            # sort the coacq_thresholds in ascending order
            coacq_thresholds = sorted(list(set(coacq_thresholds)))
        else:
            # if there is only one threshold, we take the whole dataframe
            min_coacq = max_coacq = cq_df.shape[0]
            coacq_thresholds = [min_coacq]

        # sort the cq_df by the min_transfers column in descending order
        cq_df = cq_df.sort_values(by="min_transfers", ascending=False)

        # summarize the coacquisitions for each threshold
        for i, coacq_threshold in enumerate(coacq_thresholds):
            # if it's the last threshold, use the entire dataframe
            if i == len(coacq_thresholds) - 1:
                cq_df_at_threshold = cq_df
            else:
                # extract the first coacq_threshold number of coacquisitions from the sorted cq_df
                cq_df_at_threshold = cq_df.head(int(coacq_threshold))
            min_transfer_threshold = cq_df_at_threshold[
                "min_transfers"
            ].min()  # min transfers required to get this threshold of coacquisitions

            coacq_threshold_summary_dict = process_coacq_threshold_df(
                cq_df_at_threshold,
                method,
                coacq_threshold,
                neighbor_genes_between_cutoffs,
                min_transfer_threshold,
            )

            # if no summary dict is returned, skip to the next threshold
            if coacq_threshold_summary_dict:
                coacquisitions_summary_records.append(coacq_threshold_summary_dict)

    # make df of coacquisitions_summary_records
    coacquisitions_summary_df = pd.DataFrame(coacquisitions_summary_records)

    # return the summary records for all methods across all thresholds
    return coacquisitions_summary_df


def summarize_coacquisitions_parallel_worker_fn(
    method, cq_df, neighbor_genes_between_cutoffs, min_genome_size, min_coacquisitions
):
    # keep only the coacquisitions with known pair distances
    cq_df = cq_df[cq_df["notes"] == "gene_positions_are_available"]
    if cq_df.shape[0] < min_coacquisitions:
        print(
            f"Method {method} has less than {min_coacquisitions} coacquisitions with known gene positions. Skipping."
        )
        return None

    # filter out coacquisitions with genome sizes less than min_genome_size
    cq_df.loc[:, "genome_size"] = (
        cq_df["contig:genome_size"].str.split(":").str[1].astype(int)
    )
    cq_df = cq_df[cq_df["genome_size"] >= min_genome_size]
    if cq_df.shape[0] == 0:
        print(
            f"No coacquisitions with genome size >= {min_genome_size} for method {method}"
        )
        return None

    min_transfers = 0.05  # for any coacquisition (pair of NOGs) to be considered, the minimum number of transfers to a common contig

    # if there is only one threshold, we don't need to calculate the coacq thresholds on log scale
    if cq_df["min_transfers"].nunique() > 1:
        # find the min and max num of coacquisitions for each method, which is the coacq at max and min threshold respectively
        # first the number of coacquisitions at the max threshold, with known pair distances
        min_coacq = cq_df[cq_df["min_transfers"] >= 1.0].shape[0]
        min_coacq = max(min_coacq, min_coacquisitions)

        # then the number of coacquisitions at the min threshold, with known pair distances
        max_coacq = cq_df[cq_df["min_transfers"] >= min_transfers].shape[0]

        # between the min and max coacq_threshold_indices, find which of 1eN, 2eN, 5eN are possible, where N is an integer. These would be our coacq_thresholds
        # we do this because we want to summarize the coacquisitions at log scale intervals
        N = 1
        coacq_thresholds = []
        log_multiples = [1, 2, 5, 10]
        while True:
            for i, multiple in enumerate(log_multiples):
                if multiple * 10**N >= min_coacq and multiple * 10**N <= max_coacq:
                    coacq_thresholds.append(multiple * 10**N)
                elif (
                    min_coacq > multiple * 10**N
                    and i != (len(log_multiples) - 1)
                    and (log_multiples[i + 1] + 1) * 10 ** (N) <= max_coacq
                ):
                    # this means that the range of coacquisitions doesn't contain the multiple * 10**N value
                    # then we just append both the min_coacq and max_coacq values and 5 points in between
                    coacq_thresholds.append(min_coacq)
                    for j in range(1, 6):
                        coacq_thresholds.append(
                            min_coacq + j * (max_coacq - min_coacq) / 5
                        )
                    coacq_thresholds.append(max_coacq)
            N += 1
            # if the next multiple of 10**N is greater than the max_coacq, we don't need to check further on the log scale
            if 10**N > max_coacq:
                break

        # sort the coacq_thresholds in ascending order
        coacq_thresholds = sorted(list(set(coacq_thresholds)))
    else:
        # if there is only one threshold, we take the whole dataframe
        min_coacq = max_coacq = cq_df.shape[0]
        coacq_thresholds = [min_coacq]

    # sort the cq_df by the min_transfers column in descending order
    cq_df = cq_df.sort_values(by="min_transfers", ascending=False)

    # summarize the coacquisitions for each threshold
    coacquisitions_summary_records = []
    for i, coacq_threshold in enumerate(coacq_thresholds):
        # if it's the last threshold, use the entire dataframe
        if i == len(coacq_thresholds) - 1:
            cq_df_at_threshold = cq_df
        else:
            # extract the first coacq_threshold number of coacquisitions from the sorted cq_df
            cq_df_at_threshold = cq_df.head(int(coacq_threshold))
        min_transfer_threshold = cq_df_at_threshold["min_transfers"].min()
        coacq_threshold_summary_dict = process_coacq_threshold_df(
            cq_df_at_threshold,
            method,  # method name
            coacq_threshold,  # threshold value
            neighbor_genes_between_cutoffs,  # list of neighbor genes between cutoffs
            min_transfer_threshold,  # minimum transfer threshold
        )

        # if no summary dict is returned, skip to the next threshold, else append to the list of summary records
        if coacq_threshold_summary_dict:
            coacquisitions_summary_records.append(coacq_threshold_summary_dict)

    # return the summary records for this method across all thresholds
    return coacquisitions_summary_records


def summarize_coacquisitions_parallel(
    coacquisitions_dfs,
    neighbor_genes_between_cutoffs,
    min_genome_size,
    min_coacquisitions,
    method_with_stringency_thresholds=None,
):
    """
    Summarize coacquisition data at log scale intervals, in parallel.

    Arguments:
    coacquisitions_dfs (dict): A dictionary where keys are method names and values are DataFrames containing coacquisition data.
    neighbor_genes_between_cutoffs (list): A list of integers representing the number of genes between a pair of coacquired genes to be considered neighbors.
    min_genome_size (int): The minimum genome size to consider for filtering coacquisitions. Coacquisitions with genome sizes less than this value are excluded from the summary.
    min_coacquisitions (int): The minimum number of coacquisitions with known gene positions to consider for summarization.
                                If the number of coacquisitions is less than this value, the threshold is skipped.
    method_with_stringency_thresholds (str): comma-separated string of method names for which the transfer thresholds are stringency based instead of being probabilistic.
                                            This means that a higher threshold

    Returns:
    coacquisitions_summary_df (DataFrame): A DataFrame containing the summary records for each threshold, across all thresholds for all methods.
    """

    # summarize coacquisitions for each method and threshold
    coacquisitions_summary_records = []
    with mp.Pool(len(coacquisitions_dfs)) as pool:
        for coacquisitions_summary_record in pool.starmap(
            summarize_coacquisitions_parallel_worker_fn,
            [
                (
                    method,
                    cq_df,
                    neighbor_genes_between_cutoffs,
                    min_genome_size,
                    min_coacquisitions,
                )
                for method, cq_df in coacquisitions_dfs.items()
            ],
        ):
            if coacquisitions_summary_record is not None:
                coacquisitions_summary_records.extend(coacquisitions_summary_record)
    coacquisitions_summary_records = [
        record for record in coacquisitions_summary_records if record is not None
    ]
    # make df of coacquisitions_summary_records
    coacquisitions_summary_df = pd.DataFrame(coacquisitions_summary_records)

    # return the summary records for all methods across all thresholds
    return coacquisitions_summary_df


def summarize_coacquisitions_manual_thresholds(
    coacquisitions_dfs,
    neighbor_genes_between_cutoffs,
    min_genome_size,
    method,
    min_coacquisitions=100,
):
    """
    This is similar to `summarize_coacquisitions`,
    but instead of calculating the thresholds on numbers of coacquisitions on a logscale resulting in a `cq_df_at_threshold` for each such threshold,
    each coacquisitions_df in `coacquisitions_dfs` is processed as a single cq_df_at_threshold.
    The fn processes these dfs in ascending order of size and provides a combined summary.
    This is useful for methods such as Count MP or GLOOME MP where there were separate runs resulting in different 'transfer thresholds, i.e. at different levels of stringency of the method.

    This fn should be run only for a single method at a time. The returned summary records can be combined with those from other methods using `combine_coacquisitions_summary_records`.

    Arguments:

    coacquisitions_dfs (dict): A dictionary where keys are threshold labels and values are DataFrames containing coacquisition data.
    neighbor_genes_between_cutoffs (list): A list of integers representing the number of genes between a pair of coacquired genes to be considered neighbors.
    min_genome_size (int): The minimum genome size to consider for filtering coacquisitions. Coacquisitions with genome sizes less than this value are excluded from the summary.
    min_coacquisitions (int): The minimum number of coacquisitions with known gene positions to consider for summarization.
                                If the number of coacquisitions is less than this value, the threshold is skipped.

    Returns:
    coacquisitions_summary_df (DataFrame): A DataFrame containing the summary records for each threshold, across all thresholds for the method.
    """

    # summarize coacquisitions for a single method and multiple thresholds

    # we process the coacquisitions_dfs in ascending order of size
    coacquisitions_summary_records = []
    for threshold_label, cq_df in sorted(
        coacquisitions_dfs.items(), key=lambda x: x[1].shape[0]
    ):
        # figure out method name. In these cases, the threshold label is of the form "method_name.modifier.threshold_value_float"

        # threshold value is everything after method name
        method_prefix = f"{method}." if not method.endswith(".") else method
        threshold_value = threshold_label.replace(method_prefix, "")

        # keep only the coacquisitions with known pair distances
        cq_df = cq_df[cq_df["notes"] == "gene_positions_are_available"]
        if cq_df.shape[0] < min_coacquisitions:
            print(
                f"Method {method} has less than {min_coacquisitions} coacquisitions with known gene positions. Skipping."
            )
            continue

        # filter out coacquisitions with genome sizes less than min_genome_size
        cq_df.loc[:, "genome_size"] = (
            cq_df["contig:genome_size"].str.split(":").str[1].astype(int)
        )
        cq_df = cq_df[cq_df["genome_size"] >= min_genome_size]
        if cq_df.shape[0] == 0:
            print(
                f"No coacquisitions with genome size >= {min_genome_size} for method {method} at threshold {threshold_value}"
            )
            continue

        coacq_threshold_summary_dict = process_coacq_threshold_df(
            cq_df,
            method,
            threshold_value,
            neighbor_genes_between_cutoffs,
            float(threshold_value),
        )

        # if no summary dict is returned, skip to the next threshold
        if coacq_threshold_summary_dict:
            coacquisitions_summary_records.append(coacq_threshold_summary_dict)

    # make df of coacquisitions_summary_records
    coacquisitions_summary_df = pd.DataFrame(coacquisitions_summary_records)

    # return the summary records for this method across all thresholds
    return coacquisitions_summary_df


def combine_coacquisitions_summary_records(coacquisitions_summary_records_list: list):
    """
    If there are multiple `coacquisitions_summary_records` (e.g. two from `summarize_coacquisitions_manual_thresholds` and one from `summarize_coacquisitions`),
    this function combines them into a single coacquisitions_summary_records.

    Arguments:
    coacquisitions_summar_records_list (list): A list of DataFrames containing the summary records for each threshold, across all thresholds for a method.

    Returns:
    coacquisitions_summary_df (DataFrame): A DataFrame containing the combined summary records for each threshold, across all thresholds for all methods.
    """

    # combine the coacquisitions_summary_records from multiple methods
    coacquisitions_summary_df = pd.concat(
        coacquisitions_summary_records_list,
        ignore_index=True,
        # throw error if certain columns are not present in all dfs
        verify_integrity=True,
    )

    # return the combined summary records for all methods across all thresholds
    return coacquisitions_summary_df


###############################################################################################
# EXAMPLE USAGE ###############################################################################
###############################################################################################
# data_dir = "../data/"
# res_dir = "../data/compiled_results/"
# neighbor_genes_between_cutoffs = [1]
# # read in the data: chromosome locations, and coacquisitions for each method
# coacquisitions_dfs = load_data(data_dir, res_dir)

# # for each method, find the minimum number of transfers under different transfer thresholds
# # between each pair of NOGs that are coacquired
# coacquisitions_dfs = calculate_min_transfers(coacquisitions_dfs)


# # summarize the number of coacquisitions by transfer threshold for each method
# coacquisitions_summary_records = summarize_coacquisitions(
#     coacquisitions_dfs, neighbor_genes_between_cutoffs, min_genome_size)


# # make a dataframe from the summary records
# coacquisitions_summary_df = pd.DataFrame(coacquisitions_summary_records)


# # replace zeros with NaNs for columns that should not have zeros
# coacquisitions_summary_df = replace_zeros_with_nan(
#     coacquisitions_summary_df)

# # save the summary to a file
# coacquisitions_summary_df.to_csv(f"{res_dir}/summary_coacquisitions_by_threshold_and_neighbors_new.tsv",
#                                     sep='\t', index=False, quoting=1, quotechar='"', float_format='%.2f')
# print(coacquisitions_summary_df)
# ###############################################################################################
