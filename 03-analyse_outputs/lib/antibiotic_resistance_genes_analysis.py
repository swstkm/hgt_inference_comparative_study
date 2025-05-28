import os
import pandas as pd
import numpy as np
from scipy import stats
from IPython.display import display
from tqdm import tqdm

def read_nogwise_data(compiled_res_dir):
    compiled_nogwise_filepaths = [
        os.path.join(root, fi)
        for root, _, files in os.walk(compiled_res_dir)
        for fi in files
        if fi.startswith("compiled_transfers.nogwise.") and "branchwise" not in fi
    ]
    compiled_nogwise_filepaths_dict = {
        fi.partition("compiled_transfers.nogwise.")[2].partition(".tsv")[0]: fi
        for fi in compiled_nogwise_filepaths
    }
    compiled_nogwise_dfs = {
        key: pd.read_csv(value, sep="\t")
        for key, value in compiled_nogwise_filepaths_dict.items()
    }
    # drop 'gloome_branch_name' column if it exists in any df
    for key, cn_df in compiled_nogwise_dfs.items():
        if "gloome_branch_name" in cn_df.columns:
            cn_df.drop(columns=["gloome_branch_name"], inplace=True)
        # sort dfs by 'transfers'
        try:
            compiled_nogwise_dfs[key] = cn_df.sort_values(by="transfer_threshold")
        except KeyError as e:
            raise KeyError(f"KeyError for {key}: {e}")
    return compiled_nogwise_dfs


def arg_thresholdwise_mwu(
        method: str,
        method_thresholded_df: pd.DataFrame,
):
    """
    This function takes a dataframe and performs a Mann-Whitney U test
    to compare the 'corrected_transfers' column between two groups:
    ARG NOGs and non-ARG NOGs.
    It returns a dictionary with the results of the MWU test, CLES, and mean values.
    """
    # perform MWU test
    arg_nogs = method_thresholded_df[method_thresholded_df["ARG"] == True]
    non_arg_nogs = method_thresholded_df[method_thresholded_df["ARG"] == False]
    # Check if both groups are non-empty
    if len(arg_nogs) < 5:
        # print("Warning: Not enough ARG NOGs for MWU test. Skipping method:", method,
        #         "and transfer threshold:", method_thresholded_df["transfer_threshold"].min())
        return {}
    if len(non_arg_nogs) < 5:
        # print("Warning: Not enough non-ARG NOGs for MWU test. Skipping method:", method,
        #         "and transfer threshold:", method_thresholded_df["transfer_threshold"].min())
        return {}
    # Perform MWU test
    mwu, comparison_p = stats.mannwhitneyu(
        arg_nogs["corrected_transfers"],
        non_arg_nogs["corrected_transfers"],
        alternative="two-sided",
    )
    # Find the median values and lengths
    arg_median = arg_nogs["corrected_transfers"].median()
    non_arg_median = non_arg_nogs["corrected_transfers"].median()
    arg_nogs_count = len(arg_nogs)
    non_arg_nogs_count = len(non_arg_nogs)
    # Calculate CLES
    cles = mwu / (arg_nogs_count * non_arg_nogs_count)
    # Create a dictionary to store the results and return it
    comparison_res_dict = {
        "mwu": mwu,
        "p_value": comparison_p,
        "arg_median": arg_median,
        "non_arg_median": non_arg_median,
        "difference_in_medians": arg_median - non_arg_median,
        "arg_nogs_count": arg_nogs_count,
        "non_arg_nogs_count": non_arg_nogs_count,
        "cles": cles,
    }
    return comparison_res_dict


def arg_thresholdwise_fisher_exact(
        method: str,
        method_thresholded_df: pd.DataFrame,
        arg_ratio: tuple = (), # (#arg, #non-arg): only if "fisher_exact"
):
    """
    This function takes a dataframe and performs a Fisher's exact test
    to compare the number of NOGs with non-zero corrected transfers
    between two groups: ARG NOGs and non-ARG NOGs, as compared to the
    arg_ratio (the ratio of ARG NOGs to non-ARG NOGs in the entire dataset).
    It returns a dictionary with the results of the Fisher's exact test.
    """
    # Check if the arg_ratio is valid
    if len(arg_ratio) != 2 or arg_ratio[0] <= 0 or arg_ratio[1] <= 0:
        raise ValueError("arg_ratio must be a tuple of two positive integers: (#arg, #non-arg)")
    # Check if the dataframe is empty
    if method_thresholded_df.empty:
        print("Warning: No data for method:", method, "and transfer threshold:", method_thresholded_df["transfer_threshold"].min())
        return {}
    # Create a contingency table
    contingency_table = pd.DataFrame({
        "ARG": [
            method_thresholded_df[(method_thresholded_df["ARG"] == True) & (method_thresholded_df["corrected_transfers"] > 0)]["nog_id"].nunique(),
            arg_ratio[0]
        ],
        "non-ARG": [
            method_thresholded_df[(method_thresholded_df["ARG"] == False) & (method_thresholded_df["corrected_transfers"] > 0)]["nog_id"].nunique(),
            arg_ratio[1]
        ]
    }, index=["observed", "expected"])
    # Perform Fisher's exact test
    odds_ratio, comparison_p = stats.fisher_exact(
        contingency_table,
        alternative="greater", # greater: ratio of ARG NOGs to non-ARG NOGs is greater than expected
    )
    # Create a dictionary to store the results and return it
    comparison_res_dict = {
        "odds_ratio": odds_ratio,
        "p_value": comparison_p,
        "arg_nogs_count": contingency_table.loc["observed", "ARG"],
        "non_arg_nogs_count": contingency_table.loc["observed", "non-ARG"],
        "arg_nogs_expected": contingency_table.loc["expected", "ARG"],
        "non_arg_nogs_expected": contingency_table.loc["expected", "non-ARG"],
        "total_nogs_count": contingency_table.loc["observed"].sum()
    }
    return comparison_res_dict


def arg_thresholdwise_test(
        method: str,
        method_thresholded_df: pd.DataFrame,
        arg_ratio: tuple = (), # (#arg, #non-arg): only if "fisher_exact"
        comparison_method: str = "mwu", # "mwu", "fisher_exact"
):
    valid_comparison_methods = ["mwu", "fisher_exact"]
    # depending on the comparison_method, we call the appropriate function
    if comparison_method == "mwu":
        # perform MWU test
        comparison_res_dict = arg_thresholdwise_mwu(
            method=method,
            method_thresholded_df=method_thresholded_df,
        )
    elif comparison_method == "fisher_exact":
        # perform Fisher's exact test
        comparison_res_dict = arg_thresholdwise_fisher_exact(
            method=method,
            method_thresholded_df=method_thresholded_df,
            arg_ratio=arg_ratio,
        )
    else:
        raise ValueError(f"Comparison method must be one of {valid_comparison_methods}")
    
    # Check if the comparison_res_dict is empty
    if not comparison_res_dict:
        # print("Warning: No data for method:", method, "and transfer threshold:", method_thresholded_df["transfer_threshold"].min())
        return {}
    
    return comparison_res_dict

def arg_transferrability_analysis(
        compiled_nogwise_dfs: dict,
        correction_var: str = "#genes",
        comparison_method: str = "mwu", # "mwu", "fisher_exact"
        arg_ratio: tuple = (), # (#arg, #non-arg): only if "fisher_exact"
):
    """
    This function takes a dictionary of dataframes and a correction variable.
    For each method,df in the dictionary, it normalizes the 'transfers' column by the correction variable.
    Then at transfer_threshold in the 'transfer_threshold' column, 
        it performs a test to compare the 'corrected_transfers' column between two groups, if 
        is significantly different between the two groups of ARG NOGs and non-ARG NOGs.
        Assuming that if any of the genes in a NOG is ARG, then the NOG is considered ARG.
    Comparison methods are:
        - "mwu": Mann-Whitney U test comparing distributions of corrected transfers between the two groups
        - "fisher_exact": Fisher's exact test comparing the number of NOGs with non-zero corrected transfers
    The function returns a dataframe with the results of the tests.
    """
    # Check if arg_ratio is not None if comparison_method is "fisher_exact"
    if comparison_method == "fisher_exact" and arg_ratio == ():
        raise ValueError("valid arg_ratio must be provided if comparison_method is 'fisher_exact'")

    all_method_comparison_res_records = []
    for method, method_df in tqdm(compiled_nogwise_dfs.items(), 
            desc="Processing methods", unit=" method"):
        # Check if the correction variable exists in the dataframe
        if correction_var not in method_df.columns:
            raise ValueError(f"Correction variable '{correction_var}' not found in dataframe for method '{method}'")

        # Normalize the 'transfers' column by the correction variable
        method_df["corrected_transfers"] = method_df["transfers"] / method_df[correction_var]
        # remove nans
        method_df.dropna(inplace=True)
        # Check if the transfer_threshold column is numeric
        if not pd.api.types.is_numeric_dtype(method_df["transfer_threshold"]):
            raise ValueError(f"Transfer threshold column is not numeric for method '{method}'")
        # Perform MWU test, separately for each transfer threshold
        if method_df["transfer_threshold"].unique().size > 1:
            # if integer/integer-reciprocal transfer thresholds, these are gain-loss penalties
            if all((value.is_integer() or round((1/value)).is_integer()) for value in method_df["transfer_threshold"].unique() if value > 0):
                print(f"Processing method: {method} with integer/integer-reciprocal transfer thresholds")
                # Get the unique transfer thresholds and sort them
                transfer_thresholds = method_df["transfer_threshold"].unique()
                transfer_thresholds.sort()
                for transfer_threshold in transfer_thresholds:
                    threshold_df = method_df[method_df["transfer_threshold"] == transfer_threshold].copy()
                    # display(threshold_df)
                    # the total num transfers is the sum of corrected_transfers for all NOGs
                    num_transfers = threshold_df["corrected_transfers"].sum()
                    # Check if the threshold_df is empty
                    if threshold_df.empty:
                        print(f"Warning: No data for transfer threshold {transfer_threshold} in method {method}. Skipping.")
                        continue
                    threshold_comparison_res_dict = arg_thresholdwise_test(
                        method=method,
                        method_thresholded_df=threshold_df,
                        arg_ratio=arg_ratio,
                        comparison_method=comparison_method,
                    )
                    # Store the results in the dictionary
                    if threshold_comparison_res_dict:
                        # add the method, transfer_threshold (two separate keys) and returned dict items to the results list
                        # so the list is structured as:
                        # [{'method': method, 'transfer_threshold': transfer_threshold, 'mwu': mwu, ...}, {'method': method, ...}, ...]
                        threshold_comparison_res_dict["method"] = method
                        threshold_comparison_res_dict["transfer_threshold"] = transfer_threshold
                        threshold_comparison_res_dict["num_transfers"] = num_transfers
                        all_method_comparison_res_records.append(threshold_comparison_res_dict)
            else:   # these are not integer/integer-reciprocal transfer thresholds
                print(f"Processing method: {method} with transfer thresholds that aren't integer/integer-reciprocal")
                
                skipped_thresholds = []
                if comparison_method == "mwu":
                    # we split the transfers themselves into 10 cumulative bins
                    # and find the min transfer_threshold for each bin
                    min_transfers = 1
                    max_transfers = method_df["corrected_transfers"].shape[0]
                    # create 10 cumulative bins that fit better on a log-scale
                    # first split the range into 20 points and store in a list
                    edge_range = 21 # 20 points
                    transfers_edges = [max_transfers - (
                        (max_transfers - min_transfers) * (0.5 ** i)
                    ) for i in range(edge_range)]
                    print(f"For method {method}, the transfer edges are: {transfers_edges}")
                    # now for each of these edges e, take the threshold_df as the top e of the data
                    method_df = method_df.sort_values(
                        by="transfer_threshold", ascending=False
                    )
                    for i in range(edge_range):
                        # get the top e rows of the dataframe
                        threshold_df = method_df.head(int(np.floor(transfers_edges[i]))).copy()
                        transfer_threshold = threshold_df["transfer_threshold"].min()
                        # the number of transfers is the sum of corrected_transfers for all NOGs
                        # but note that generally for the methods in this section
                            # the 'transfer_threshold' is 
                            # not cumulative but only an expectation for each NOG
                        num_transfers = threshold_df["corrected_transfers"].sum()
                        # Check if the threshold_df is empty
                        if threshold_df.empty:
                            print(f"Warning: No data for transfer threshold {transfer_threshold} in method {method}."
                                f"Number of NOGs: {len(threshold_df)} for edge {transfers_edges[i]}. Skipping.")
                            continue
                        threshold_comparison_res_dict = arg_thresholdwise_test(
                            method=method,
                            method_thresholded_df=threshold_df,
                            arg_ratio=arg_ratio,
                            comparison_method=comparison_method,
                        )
                        # Store the results in the dictionary
                        if threshold_comparison_res_dict:
                            # add the method, transfer_threshold (two separate keys) and returned dict items to the results list
                            # so the list is structured as:
                            # [{'method': method, 'transfer_threshold': transfer_threshold, 'mwu': mwu, ...}, {'method': method, ...}, ...]
                            threshold_comparison_res_dict["method"] = method
                            threshold_comparison_res_dict["transfer_threshold"] = transfer_threshold
                            all_method_comparison_res_records.append(threshold_comparison_res_dict)
                        else:
                            skipped_thresholds.append(transfer_threshold)
                    print(f"Warning: For method {method}, skipped transfer thresholds: {skipped_thresholds}")
                elif comparison_method == "fisher_exact":
                    # we split the #nogs into 10 cumulative bins
                    # and find the min transfer_threshold for each bin
                    # first split the range into 20 points and store in a list
                    edge_range = 21 # 20 points
                    num_nogs_edges = [
                        method_df["nog_id"].nunique() - (
                            (method_df["nog_id"].nunique() - 1) * (0.5 ** i)
                        ) for i in range(edge_range)
                    ]
                    print(f"For method {method}, the NOG edges are: {num_nogs_edges}")
                    # now for each of these edges e, take the threshold_df as the top e number of NOGs from the data
                    method_df = method_df.sort_values(
                        by="transfer_threshold", ascending=False
                    )
                    for i in range(edge_range):
                        # get the rows corresponding to the first e nogs from the dataframe
                        selected_nog_ids = method_df["nog_id"].unique()[:int(np.floor(num_nogs_edges[i]))]
                        threshold_df = method_df[
                            method_df["nog_id"].isin(selected_nog_ids)
                        ].copy()
                        transfer_threshold = threshold_df["transfer_threshold"].min()
                        # Check if the threshold_df is empty
                        if threshold_df.empty:
                            print(f"Warning: No data for transfer threshold {transfer_threshold} in method {method}."
                                f"Number of NOGs: {len(threshold_df)} for edge {num_nogs_edges[i]}. Skipping.")
                            continue
                        threshold_comparison_res_dict = arg_thresholdwise_test(
                            method=method,
                            method_thresholded_df=threshold_df,
                            arg_ratio=arg_ratio,
                            comparison_method=comparison_method,
                        )
                        # Store the results in the dictionary
                        if threshold_comparison_res_dict:
                            # add the method, transfer_threshold (two separate keys) and returned dict items to the results list
                            # so the list is structured as:
                            # [{'method': method, 'transfer_threshold': transfer_threshold, 'mwu': mwu, ...}, {'method': method, ...}, ...]
                            threshold_comparison_res_dict["method"] = method
                            threshold_comparison_res_dict["transfer_threshold"] = transfer_threshold
                            all_method_comparison_res_records.append(threshold_comparison_res_dict)
                        else:
                            skipped_thresholds.append(transfer_threshold)
                    print(f"Warning: For method {method}, skipped transfer thresholds: {skipped_thresholds}")
        elif method_df["transfer_threshold"].unique().size == 1:
            # if there is only one transfer threshold, we can just use the whole dataframe
            threshold_df = method_df.copy()
            # test 
            threshold_comparison_res_dict = arg_thresholdwise_test(
                method=method,
                method_thresholded_df=threshold_df,
                arg_ratio=arg_ratio,
                comparison_method=comparison_method,
            )
            # Store the results in the dictionary
            if threshold_comparison_res_dict:
                # add the method, transfer_threshold (two separate keys) and returned dict items to the results list
                # so the list is structured as:
                # [{'method': method, 'transfer_threshold': transfer_threshold, 'mwu': mwu, ...}, {'method': method, ...}, ...]
                threshold_comparison_res_dict["method"] = method
                threshold_comparison_res_dict["transfer_threshold"] = method_df["transfer_threshold"].min()
                all_method_comparison_res_records.append(threshold_comparison_res_dict)
        else: # if method_df["transfer_threshold"].unique().size == 0:
            print(f"Warning: No data for transfer threshold in method {method}. Skipping.")
            continue
    # Create a dataframe from the results list
    all_method_comparison_res_df = pd.DataFrame(all_method_comparison_res_records)
    # Sort the dataframe by method and transfer_threshold
    all_method_comparison_res_df.sort_values(
        by=["method", "transfer_threshold"], inplace=True
    )
    # Reset the index
    all_method_comparison_res_df.reset_index(drop=True, inplace=True)
    
    # return the dataframe
    return all_method_comparison_res_df
                    
