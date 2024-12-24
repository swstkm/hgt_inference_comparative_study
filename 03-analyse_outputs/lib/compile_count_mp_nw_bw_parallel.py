import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

# to suppress warning from ete3 because it's not up to date with py3.12
import warnings

# ignore SyntaxWarning
warnings.filterwarnings("ignore", category=SyntaxWarning)

import ete3


def calculate_count_inferred_transfers_for_branch(args):
    nog_id, branch_name, genes_in_branch, genes_in_ancestor = args
    transfers, losses, duplications, reductions = 0, 0, 0, 0
    if genes_in_ancestor == 0 and genes_in_branch > 0:
        transfers = genes_in_branch
    elif genes_in_ancestor > 0 and genes_in_branch == 0:
        losses = genes_in_ancestor
    elif genes_in_ancestor > 0 and genes_in_branch > genes_in_ancestor:
        duplications = genes_in_branch - genes_in_ancestor
    elif genes_in_ancestor > 0 and genes_in_branch < genes_in_ancestor:
        reductions = genes_in_ancestor - genes_in_branch
    return (nog_id, branch_name, transfers, losses, duplications, reductions)


def process_count_branches_in_parallel(df, input_tree, max_processes):
    results = []

    with Pool(processes=max_processes) as pool:
        nogid_branch_pairs = [
            (n, b.name, df.loc[n][b.name], df.loc[n][b.up.name])
            for n in df.index
            for b in input_tree.get_descendants()
        ]
        pbar_process = tqdm(
            total=len(nogid_branch_pairs),
            desc=f"Processing nog-branch pairs",
            leave=False,
        )
        print(f"Processing {len(nogid_branch_pairs)} nog-branch pairs")

        # # Process in chunks to reduce overhead
        chunksize = 10  # Adjust based on your workload
        for pool_result in pool.imap_unordered(
            calculate_count_inferred_transfers_for_branch,
            nogid_branch_pairs,
            chunksize=chunksize,
        ):
            results.append(pool_result)
            pbar_process.update(chunksize)

    return results


def calculate_count_nogwise_branchwise_transfers(
    count_nogwise_transfers_filepath, input_tree_filepath, max_processes=500
):
    """
    Calculate the number of transfers, losses, duplications and reductions for each NOG in each branch of the input tree.
    Parameters
    ----------
    count_nogwise_transfers_filepath : str
        Path to the output file of `Count`, but only the lines starting with #FAMILY
    input_tree_filepath : str
        Path to the input tree file in Newick format

    Returns
    -------
    list
        A list of tuples, each containing the NOG ID, branch name, number of transfers, losses, duplications and reductions
    """

    df = pd.read_csv(count_nogwise_transfers_filepath, sep="\t").sort_values("name")
    # remove the #FAMILY column
    df.drop("# FAMILY", axis=1, inplace=True)
    # set the index to 'name' column
    df.set_index("name", inplace=True)
    input_tree = ete3.Tree(input_tree_filepath, format=1)
    return process_count_branches_in_parallel(df, input_tree, max_processes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--count_nogwise_transfers_filepath",
        type=str,
        help="Path to the output file of `Count`, but only the lines starting with #FAMILY",
    )
    parser.add_argument(
        "-t",
        "--input_tree_filepath",
        type=str,
        required=True,
        help="Path to the input tree file in Newick format",
    )
    parser.add_argument(
        "-p",
        "--max_processes",
        type=int,
        required=False,
        help="Number of processes to run in parallel (default: 500)",
        default=500,
    )
    parser.add_argument(
        "-o",
        "--output_filepath",
        type=str,
        help="Output file path to write the results",
        required=True,
    )
    args = parser.parse_args()

    results = calculate_count_nogwise_branchwise_transfers(
        args.count_nogwise_transfers_filepath,
        args.input_tree_filepath,
        args.max_processes,
    )

    nogwise_branchwise_df = pd.DataFrame.from_records(
        results,
        columns=[
            "nog_id",
            "recipient_branch",
            "transfers",
            "losses",
            "duplications",
            "reductions",
        ],
    )
    # if all values are zero, remove the row
    nogwise_branchwise_df = nogwise_branchwise_df[
        nogwise_branchwise_df[
            ["transfers", "losses", "duplications", "reductions"]
        ].sum(axis=1)
        > 0
    ]

    # write out this dataframe
    nogwise_branchwise_df.to_csv(args.output_filepath, sep="\t", index=False)
