import os
from multiprocessing import Pool
from typing import Union, List, Optional

# to suppress warning from ete3 because it's not up to date with py3.12
import warnings

# filter SyntaxWarning
warnings.filterwarnings("ignore", category=SyntaxWarning)

import ete3
import pandas as pd
from IPython.display import display
import numpy as np
from tqdm.notebook import tqdm


def map_output_to_input_nodes(input_tree, output_tree, ale=False):
    """
    Given two ete3 trees, this function returns a dictionary that maps the nodes of the output tree to the nodes of the input tree.
    """
    # first we create a dictionary for each tree. The keys are the descendant leaves of each internal node, and the values are the nodes themselves.
    # The keys are strings of the form "leaf1,leaf2,leaf3" where the leaf names are sorted lexicographically.
    input_tree_dict = {}
    for input_node in input_tree.traverse(strategy="postorder"):
        if not input_node.is_leaf():  # internal node
            descendants = input_node.get_leaf_names()
            descendants.sort()
            input_tree_dict[",".join(descendants)] = input_node.name
            if input_node.name == "":
                print(
                    "Empty name found. Number of descendants: {}.".format(
                        len(descendants)
                    )
                )
        else:
            input_tree_dict[input_node.name] = input_node.name
    output_tree_dict = {}
    for output_node in output_tree.traverse(
        strategy="postorder"
    ):  # we traverse the tree in postorder
        if not output_node.is_leaf():  # internal node
            descendants = output_node.get_leaf_names()
            descendants.sort()
            if ale:
                output_tree_dict[",".join(descendants)] = f"N{output_node.name}"
            else:
                output_tree_dict[",".join(descendants)] = output_node.name
        else:
            output_tree_dict[output_node.name] = output_node.name

    # now we iterate over the keys of the input tree dictionary, and for each key we check if it is also a key in the output tree dictionary
    # if it is, we add an entry to the mapping dictionary (output_node -> input_node)
    # if not, we add an entry to the mapping dictionary (input_node -> None), and raise an error
    node_mapping = {}
    node_not_found_list = []
    for output_key in output_tree_dict.keys():
        if (
            output_key in input_tree_dict
        ):  # if the output key is found in the input tree, add the mapping
            node_mapping[output_tree_dict[output_key]] = input_tree_dict[output_key]
        else:
            node_mapping[output_tree_dict[output_key]] = None
            node_not_found_list.append(output_key)
    # if node_not_found_list is not empty, print length and contents of it
    if node_not_found_list:
        print(f"Nodes not found in input tree: {len(node_not_found_list)}")
        print(node_not_found_list)

    return node_mapping


def map_angst_to_input_nodes(input_tree, angst_internal_nodes_list):

    # each angst internal node is of the form leaf1-leaf2-leaf3-... where leafx are the leaves of the subtree of the internal node

    # first we create a dictionary for the input_tree
    # mapping is leaf1,leaf2,leaf3 -> internal_node_name based on the internal nodes of the input tree
    input_tree_dict = {}
    for input_node in input_tree.traverse():
        if not input_node.is_leaf():
            descendants = input_node.get_leaf_names()
            descendants.sort()
            input_tree_dict[",".join(descendants)] = input_node.name
        else:
            input_tree_dict[input_node.name] = input_node.name

    # now, we do similar for the angst internal nodes
    # mapping is leaf1-leaf2-leaf3 -> internal_node_name based on the internal nodes of angst output
    output_tree_dict = {}
    for angst_int_node in angst_internal_nodes_list:
        # if node is not leaf, it has "-" delimited string of leaf names
        if "-" in angst_int_node:
            descendants = angst_int_node.split("-")
            descendants.sort()
            output_tree_dict[",".join(descendants)] = angst_int_node
        else:
            output_tree_dict[angst_int_node] = angst_int_node

    # now we create node_mapping between angst internal nodes and input tree internal nodes
    node_mapping = {}
    for concatenated_descendants in input_tree_dict.keys():
        if concatenated_descendants in output_tree_dict:
            node_mapping[output_tree_dict[concatenated_descendants]] = input_tree_dict[
                concatenated_descendants
            ]
        else:
            # print(f"Internal node with descendants {concatenated_descendants} not found in angst tree.")
            # possibly because it doesn't exist among recipient or source branches for angst
            # in which case just map it to "-" delimited version of itself (or not if it's a single leaf name)
            node_mapping[concatenated_descendants] = "-".join(concatenated_descendants)

    return node_mapping


def prepare_nogwise_transfer_thresholds_df(nogwise_transfers_df, threshold_column=True):
    """
    Given a nogwise branchwise dataframe with columns ['nog_id, 'source', 'recipient', 'transfers', 'transfer_threshold'],
    this function returns a dataframe with columns ['nog_id', 'transfer_threshold', 'transfers'].

    For each unique value of 'transfer_threshold' in the input dataframe,
    the corresponding 'transfers' values for all rows with 'transfer_threshold' >= the unique value are summed up.

    If 'threshold_column' is False, that 'transfer_threshold' column is not included in the output dataframe.
    """
    # first, sort the dataframe by 'transfer_threshold' in descending order
    nogwise_transfers_df = nogwise_transfers_df.sort_values(
        by="transfer_threshold", ascending=False
    )

    # assert that required columns are present
    assert "nog_id" in nogwise_transfers_df.columns
    assert "transfer_threshold" in nogwise_transfers_df.columns
    assert "transfers" in nogwise_transfers_df.columns

    # create a new dataframe with columns ['nog_id', 'transfer_threshold', 'transfers']
    nogwise_transfer_thresholds_df = []

    # iterate over the unique values of 'transfer_threshold' in the input dataframe
    transfer_thresholds = nogwise_transfers_df["transfer_threshold"].unique()
    # there's a chance that there are too many unique values of 'transfer_threshold'
    # in which case, we can reduce the number of unique values by coarse-graining between the min and max values
    if len(transfer_thresholds) > 100:
        transfer_thresholds = np.linspace(
            nogwise_transfers_df["transfer_threshold"].min(),
            nogwise_transfers_df["transfer_threshold"].max(),
            100,
        )
        # make sure the last value is the max value
        transfer_thresholds[-1] = nogwise_transfers_df["transfer_threshold"].max()
    for transfer_threshold in tqdm(
        transfer_thresholds, desc="Processing transfer thresholds"
    ):
        # for each value of 'transfer_threshold', sum up the 'transfers' values for all rows with 'transfer_threshold' >= the current value
        this_nogwise_transfers_df = nogwise_transfers_df[
            nogwise_transfers_df["transfer_threshold"] >= transfer_threshold
        ]
        this_nogwise_transfers = this_nogwise_transfers_df.groupby("nog_id")[
            "transfers"
        ].sum()  # sum up the 'transfers' values for each 'nog_id'
        this_nogwise_transfers = (
            this_nogwise_transfers.reset_index()
        )  # convert the groupby object to a dataframe
        this_nogwise_transfers["transfer_threshold"] = (
            transfer_threshold  # add the 'transfer_threshold' column
        )
        nogwise_transfer_thresholds_df.append(this_nogwise_transfers)

    nogwise_transfer_thresholds_df = pd.concat(nogwise_transfer_thresholds_df)
    # remove rows where the 'transfers' value is 0
    nogwise_transfer_thresholds_df = nogwise_transfer_thresholds_df[
        nogwise_transfer_thresholds_df["transfers"] > 0
    ]

    if not threshold_column:
        nogwise_transfer_thresholds_df = (
            nogwise_transfers_df.groupby("nog_id")["transfers"].sum().reset_index()
        )

    return nogwise_transfer_thresholds_df


def compile_angst_results(output_dir, input_tree_filepath):
    # for each nog_id/ read in the .events file inside it and store it
    nogwise_hgt_list = []
    for nog_id in [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]:
        # read in the AnGST.events file inside it, but only lines that start with "[hgt]: "
        with open(os.path.join(output_dir, nog_id, "AnGST.events"), "r") as nog_fo:
            # lines look like `[hgt]: source_branch --> target_branch`
            hgt_events = [
                (nog_id, l.split()[1], l.split()[3])
                for l in nog_fo.readlines()
                if l.startswith("[hgt]")
            ]
        # add this list of rows to the total list of rows
        nogwise_hgt_list.extend(hgt_events)

    # make a df out of this list of 3-tuples
    nogwise_branchwise_df = pd.DataFrame.from_records(
        nogwise_hgt_list, columns=["nog_id", "source_branch", "recipient_branch"]
    )
    print(f"Original transfers DF looks like:")
    display(nogwise_branchwise_df)

    # make a list of internal nodes that angst has, in both source_branch and recipient_branch
    angst_internal_nodes = nogwise_branchwise_df["source_branch"].unique()
    # add recipient_branch names to this
    angst_internal_nodes = set(angst_internal_nodes).union(
        set(nogwise_branchwise_df["recipient_branch"].unique())
    )

    # note that angst writes internal nodes in terms of leaves of the subtree for that internal node
    # e.g. leaf1-leaf2-leaf3
    # we need to map these to the input tree internal node names
    input_tree = ete3.Tree(input_tree_filepath, format=1)
    angst_node_mapping = map_angst_to_input_nodes(input_tree, angst_internal_nodes)
    # write this mapping to a tsv file
    with open(os.path.join(output_dir, "angst_node_mapping.tsv"), "w") as angst_node_mapping_fo:
        angst_node_mapping_fo.write("angst_node\tinput_tree_node\n")
        for angst_node, input_tree_node in angst_node_mapping.items():
            angst_node_mapping_fo.write(f"{angst_node}\t{input_tree_node}\n")

    # now replace in the nogwise df, the names, based on this mapping
    nogwise_branchwise_df["source_branch"] = nogwise_branchwise_df["source_branch"].map(
        lambda x: angst_node_mapping[x]
    )
    nogwise_branchwise_df["recipient_branch"] = nogwise_branchwise_df[
        "recipient_branch"
    ].map(lambda x: angst_node_mapping[x])
    # add a single 'transfers' column with just 1s
    nogwise_branchwise_df["transfers"] = 1

    # this is the nogwise.branchwise df
    nogwise_df = nogwise_branchwise_df.copy()

    # the original nogwise_hgt_df is actually nogwise.branchwise. We can count the number of transfers for each nog_id
    # first add the threshold column
    nogwise_df["transfer_threshold"] = (
        1  # since all 'transfers' values are 1, there are no multiple thresholds in case of AnGST
    )
    nogwise_df = prepare_nogwise_transfer_thresholds_df(nogwise_df)

    return nogwise_branchwise_df, nogwise_df


def compile_ale_outputs(output_dir, input_tree):
    """
    Given the output_dir of an ALE run, and the input_tree, this function compiles the outputs of the ALE run into a set of files:
    - compiled_transfers.nogwise.branchwise.ale.tsv
    - compiled_transfers.nogwise.ale
    - compiled_transfers.branchwise.ale
    """

    # extract the specific ALE dir name from the output_dir
    ale_dir = os.path.basename(os.path.dirname(output_dir))
    # convert to smaller case
    ale_dir = ale_dir.lower()
    print(f"Files to be written with {ale_dir}:")

    # read in the first uml file you can find in output_dir.
    uml_file = [f for f in os.listdir(output_dir) if f.endswith(".uml_rec")][0]
    # Split the third line and take the last element as the newick string
    with open(os.path.join(output_dir, uml_file), "r") as uml_fo:
        ale_tree_string = uml_fo.readlines()[2].split()[-1].strip()

    ale_tree = ete3.Tree(ale_tree_string, format=1)
    ale_node_mapping = map_output_to_input_nodes(input_tree, ale_tree, ale=True)
    # write this mapping to a tsv file
    with open(os.path.join(output_dir, "ale_node_mapping.tsv"), "w") as ale_node_mapping_fo:
        ale_node_mapping_fo.write("ale_node\tinput_tree_node\n")
        for ale_node, input_tree_node in ale_node_mapping.items():
            ale_node_mapping_fo.write(f"{ale_node}\t{input_tree_node}\n")

    # now read in all the *uTs files in the output_dir, and process them.
    # each file contains the columns: 'source_branch', 'recipient_branch', and 'freq' (of transfer).
    # for each tuple of branches (from, to),  take the sum of the freqs across all the files

    # first, get the list of all the .uTs files (not .uml_rec)
    all_uts_files = [f for f in os.listdir(output_dir) if f.endswith(".uTs")]

    # first we create a dict of dfs, mapping nog to file
    nogwise_hgt_dfs_dict = {}
    for uts_file in all_uts_files:
        nog_id = uts_file.split("_")[-1].split(".")[0]
        # read in the file
        with open(os.path.join(output_dir, uts_file), "r") as uts_fo:
            # split each line and take last 3 elements, join the first 2 of them for the from-to tuple, and create a dict between that and the last element
            uts_lines = [
                tuple(l.split()[-3:])
                for l in uts_fo.readlines()
                if not l.startswith("#")
            ]
            new_uts_lines = []
            for i, uts_line in enumerate(uts_lines):
                from_branch, to_branch, freq = uts_line
                if "(" in from_branch:
                    new_from_branch = from_branch.split("(")[0]
                else:
                    new_from_branch = f"N{from_branch}"
                if "(" in to_branch:
                    new_to_branch = to_branch.split("(")[0]
                else:
                    new_to_branch = f"N{to_branch}"
                new_uts_lines.append((new_from_branch, new_to_branch, freq))
            uts_records = [
                (
                    ale_node_mapping[from_branch],
                    ale_node_mapping[to_branch],
                    float(freq),
                )
                for from_branch, to_branch, freq in new_uts_lines
            ]
            # create a df from this list of records, with columns 'source_branch'(str), 'recipient_branch'(str), 'transfers'(float)
            uts_df = pd.DataFrame.from_records(
                uts_records, columns=["source_branch", "recipient_branch", "transfers"]
            )
            # make sure the from and to columns are string type only and transfers is float type
            uts_df["source_branch"] = uts_df["source_branch"].astype(str)
            uts_df["recipient_branch"] = uts_df["recipient_branch"].astype(str)
            uts_df["transfers"] = uts_df["transfers"].astype(float)
            # add nog_id column
            uts_df["nog_id"] = nog_id
            # add this df to the dict
            nogwise_hgt_dfs_dict[nog_id] = uts_df

    # then we can create a nogwise.branchwise file by concatenating the dfs.
    nogwise_branchwise_hgt_df = pd.concat(
        nogwise_hgt_dfs_dict.values(), ignore_index=True
    )

    # columns should be 'nog_id', 'source_branch', 'recipient_branch', 'transfers' in that order
    nogwise_branchwise_hgt_df = nogwise_branchwise_hgt_df[
        ["nog_id", "source_branch", "recipient_branch", "transfers"]
    ]

    nogwise_hgt_df = nogwise_branchwise_hgt_df.copy()

    # create a nogwise transfers file by taking the sum of the transfers for each nog_id
    # first add a 'transfer_threshold' column to each df
    nogwise_hgt_df["transfer_threshold"] = nogwise_hgt_df["transfers"]
    # then group by nog_id and sum the transfers and add the transfer_threshold column properly
    nogwise_hgt_df = prepare_nogwise_transfer_thresholds_df(nogwise_hgt_df)

    return nogwise_hgt_df, nogwise_branchwise_hgt_df


def compile_ranger_results(ranger_dir, input_tree_filepath):
    """
    Compile the results from a given ranger output directory into two dataframes:
    1. nogwise_branchwise_hgt_df: DataFrame containing transfer events for each nog_id, source_branch, and recipient_branch.
    2. nogwise_hgt_df: DataFrame containing the sum of transfers for each nog_id.

    Parameters:
    ranger_dir (str): Path to the ranger output directory.
    input_tree_filepath (str): Path to the input tree file.

    Returns:
    tuple: (nogwise_branchwise_hgt_df, nogwise_hgt_df)
    """
    # the ranger_dir contains a dir for each nog_id, with name tmp_{nog_ID}_results
    ranger_nog_results_dirs = [
        d for d in os.listdir(ranger_dir) if os.path.isdir(os.path.join(ranger_dir, d))
    ]

    # each of these dirs contains files with names in the form: `recon*`
    # we want all the transfer events across all the branches in all the recon files
    # these are lines in the files that end with "Mapping --> {source_branch}, Recipient --> {recipient_branch}"
    # we want to count the number of times each of these events occurs across all the recon files for each nog_id
    nogwise_branchwise_hgt_freq_list = []

    for nog_results_dir in tqdm(ranger_nog_results_dirs, desc="Processing NOGs"):
        nog_id = nog_results_dir.split("_")[1]
        # get all the recon files in this dir
        recon_files = [
            f
            for f in os.listdir(os.path.join(ranger_dir, nog_results_dir))
            if f.startswith("recon")
        ]
        # create a list to store the events
        nog_hgt_list = []
        # for each recon file, read in the lines that end with "Mapping --> {source_branch}, Recipient --> {recipient_branch}"
        for recon_file in recon_files:
            with open(
                os.path.join(ranger_dir, nog_results_dir, recon_file), "r"
            ) as recon_fo:
                # lines if they contain "Transfer"
                hgt_events = [
                    l.split(":")[1].split()
                    for l in recon_fo.readlines()
                    if "Transfer," in l
                ]
                # if there are no transfers in this file, skip it
                if len(hgt_events) == 0:
                    continue
                # add these to the list as just the source and recipient branches
                hgt_events = [(l[3].strip(", "), l[-1].strip()) for l in hgt_events]
                nog_hgt_list.extend(hgt_events)
        # for each tuple of branches (from, to), count take the freq (count of the tuple, divided by total number of files) and add it to the list
        this_nog_transfer_edges = set(nog_hgt_list)
        num_recon_files = len(recon_files)
        nog_hgt_list = [
            (
                nog_id,
                from_branch,
                to_branch,
                nog_hgt_list.count((from_branch, to_branch)) / num_recon_files,
            )
            for from_branch, to_branch in this_nog_transfer_edges
        ]
        # add this list of rows to the total list of rows
        nogwise_branchwise_hgt_freq_list.extend(nog_hgt_list)

    # make a df out of the list of nogwise_branchwise_hgt_freq_list
    nogwise_branchwise_hgt_df = pd.DataFrame.from_records(
        nogwise_branchwise_hgt_freq_list,
        columns=["nog_id", "source_branch", "recipient_branch", "transfers"],
    )

    # now we need to translate the branch names to the input tree branch names
    # first, get one of the species trees that ranger used. This is there in line #5 of any of the recon files
    with open(
        os.path.join(ranger_dir, ranger_nog_results_dirs[0], "recon1"), "r"
    ) as recon_fo:
        ranger_tree_string = recon_fo.readlines()[4].strip()
    ranger_tree = ete3.Tree(ranger_tree_string, format=1)
    input_tree = ete3.Tree(input_tree_filepath, format=1)
    ranger_node_mapping = map_output_to_input_nodes(input_tree, ranger_tree)
    # write this mapping to a tsv file
    with open(os.path.join(ranger_dir, "ranger_node_mapping.tsv"), "w") as ranger_node_mapping_fo:
        ranger_node_mapping_fo.write("ranger_node\tinput_tree_node\n")
        for ranger_node, input_tree_node in ranger_node_mapping.items():
            ranger_node_mapping_fo.write(f"{ranger_node}\t{input_tree_node}\n")

    # now replace in the nogwise_branchwise df, the names, based on this mapping
    nogwise_branchwise_hgt_df["source_branch"] = nogwise_branchwise_hgt_df[
        "source_branch"
    ].map(lambda x: ranger_node_mapping[x])
    nogwise_branchwise_hgt_df["recipient_branch"] = nogwise_branchwise_hgt_df[
        "recipient_branch"
    ].map(lambda x: ranger_node_mapping[x])

    # create a nogwise transfers file by taking the sum of the transfers for each nog_id
    nogwise_hgt_df = nogwise_branchwise_hgt_df.copy()
    # drop the zero transfers rows
    nogwise_hgt_df = nogwise_hgt_df[nogwise_hgt_df["transfers"] > 0.0]
    nogwise_hgt_df["transfer_threshold"] = nogwise_hgt_df["transfers"]
    nogwise_hgt_df = prepare_nogwise_transfer_thresholds_df(nogwise_hgt_df)

    return nogwise_branchwise_hgt_df, nogwise_hgt_df


def compile_wn_results(wn_hgt_genes_filepath, members_filepath):
    # read in the HGT_genes.tsv file from the results dir
    hgt_genes_df = pd.read_csv(wn_hgt_genes_filepath, sep="\t")
    # column gene_id is {taxon_id}.{locus_tag} columns
    hgt_genes_df["gene_id"] = (
        hgt_genes_df["taxon_id"].astype(str) + "." + hgt_genes_df["locus_tag"]
    )

    # read in the members.tsv file from the data dir
    members_df = pd.read_csv(
        members_filepath, sep="\t", usecols=[1, 5], header=None
    ).rename({1: "nog_id", 5: "gene_ids"}, axis=1)
    # split the gene_ids column by ',' and expand into multiple rows
    members_df = members_df.assign(
        gene_ids=members_df["gene_ids"].str.split(",")
    ).explode("gene_ids")
    # make a dict from gene_id values to nog_id values as list
    gene_nog_dict = members_df.set_index("gene_ids")["nog_id"].to_dict()
    # show some of this map
    print(f"Gene to NOG map for some genes:")
    for gene_id, nog_id in list(gene_nog_dict.items())[:5]:
        print(f"{gene_id}: {nog_id}")

    # make a new column nog_ids in hgt_genes_df by mapping gene_id to nog_id using gene_nog_dict
    hgt_genes_df["nog_id"] = hgt_genes_df["gene_id"].map(gene_nog_dict)

    # the hgt_genes_df has HGT columns (bool) marked with stringency levels, with names of form 'HGT (stringency {stringency_level})'
    # we use these to calculate the 'transfers' and the 'transfer_threshold' columns
    # the 'transfers' column is the number of the stringency levels where HGT is True
    # the 'transfer_threshold' column is a list of the stringency levels where HGT is True
    # first, get the columns that have 'HGT (stringency' in them: these columns mark the HGT events across different stringency levels
    hgt_columns = [c for c in hgt_genes_df.columns if "HGT (stringency " in c]
    # make sure they are boolean type
    hgt_genes_df[hgt_columns] = hgt_genes_df[hgt_columns].astype(bool)
    # create a new column 'transfers' in hgt_genes_df as just 1, since each row is a single transfer event
    hgt_genes_df["transfers"] = 1
    # create a new column 'transfer_threshold' in hgt_genes_df as the max stringency level where HGT is True
    hgt_genes_df["transfer_threshold"] = hgt_genes_df[hgt_columns].apply(
        lambda x: max([float(c.split("=")[1].strip(")")) for c in hgt_columns if x[c]]),
        axis=1,
    )
    # drop the hgt_columns
    hgt_genes_df = hgt_genes_df.drop(columns=hgt_columns)
    # we explode the 'transfer_threshold' column into multiple rows
    hgt_genes_df = hgt_genes_df.explode("transfer_threshold")

    # this is now converted to a proper nogwise_hgt_df by grouping by nog_id and summing the 'transfers' column across each transfer_threshold
    nogwise_hgt_df = prepare_nogwise_transfer_thresholds_df(hgt_genes_df)

    # create a nogwise.branchwise transfers file by taking the max of the 'transfer_threshold' column for each nog_id, taxon_id pair
    # therefore at this max stringency level, the transfer is guaranteed to have happened,
    # even if one of the transfers in the group is also there at a lower stringency level
    nogwise_branchwise_hgt_df = (
        hgt_genes_df.groupby(["nog_id", "taxon_id"])["transfer_threshold"]
        .max()
        .reset_index()
    )
    nogwise_branchwise_hgt_df = nogwise_branchwise_hgt_df.rename(
        {"taxon_id": "recipient_branch", "transfer_threshold": "transfers"}, axis=1
    )
    # add a source_branch column
    nogwise_branchwise_hgt_df["source_branch"] = "unknown"
    # reorder the columns
    nogwise_branchwise_hgt_df = nogwise_branchwise_hgt_df[
        ["nog_id", "source_branch", "recipient_branch", "transfers"]
    ]

    return nogwise_hgt_df, nogwise_branchwise_hgt_df


def compile_count_mp_nogwise_transfers(count_MP_output_dir, taxonomic_id, res_dir):
    """
    Compiles nogwise transfers from Count MP output directory.

    Parameters:
    count_MP_output_dir (str): Directory containing Count MP output files.
    taxonomic_id (str): Taxonomic ID to filter files.
    res_dir (str): Directory to save the compiled nogwise transfers file.

    Returns:
    pd.DataFrame: Compiled nogwise transfers DataFrame.
    """
    # Find the files in this dir that end with `_families.tsv`
    count_MP_nogwise_transfers_filepaths = [
        os.path.join(count_MP_output_dir, f)
        for f in os.listdir(count_MP_output_dir)
        if f.endswith("_families.tsv") and f.startswith(taxonomic_id)
    ]

    # Read in the familywise files as DataFrames as a list of tuples
    # The first element is the gain penalty ratio for that specific file
    # The second element is the DataFrame
    count_MP_nogwise_changes_transfers_dfs = [
        (f.split("_")[-2], pd.read_csv(f, sep="\t"))
        for f in count_MP_nogwise_transfers_filepaths
    ]

    # Sort by 'name' column
    count_MP_nogwise_transfers_dfs = [
        (i, df.sort_values("name")) for i, df in count_MP_nogwise_changes_transfers_dfs
    ]

    # For each DataFrame, keep only columns we need, and rename them
    count_MP_nogwise_transfers_dfs = [
        (
            i,
            df[["name", "Gains"]].rename(
                {"name": "nog_id", "Gains": "transfers"}, axis=1
            ),
        )
        for i, df in count_MP_nogwise_transfers_dfs
    ]

    # Add a column 'transfer_threshold' as the gain penalty ratio for that specific file
    count_MP_nogwise_transfers_dfs = [
        (i, df.assign(transfer_threshold=i)) for i, df in count_MP_nogwise_transfers_dfs
    ]

    # Create a single nogwise DataFrame
    # First, create a dict of nog_id to DataFrame
    count_MP_nogwise_transfers_dfs_dict = dict(count_MP_nogwise_transfers_dfs)

    # Then concatenate the DataFrames
    count_MP_nogwise_transfers_df = pd.concat(
        count_MP_nogwise_transfers_dfs_dict.values(), ignore_index=True
    )

    # Group by nog_id and sum the 'transfers' column
    count_MP_nogwise_transfers_df = prepare_nogwise_transfer_thresholds_df(
        count_MP_nogwise_transfers_df
    )

    # Write it
    count_MP_nogwise_transfers_df.to_csv(
        f"{res_dir}/compiled_transfers.nogwise.count.mp.tsv",
        index=False,
        header=True,
        sep="\t",
    )

    return count_MP_nogwise_transfers_df


def combine_count_mp_nw_bw_transfers(count_mp_files):
    """
    Combines compiled nogwise branchwise transfers from a list of Count MP files.

    Parameters:
    count_mp_files (list): List of Count MP file paths with basenames of the form `compiled_transfers.nogwise.branchwise.count.mp.{gain_penalty_ratio}.tsv`.
    These should be compiled nogwise branchwise transfer files for each gain_penalty_ratio

    Returns:
    pd.DataFrame: Compiled nogwise branchwise transfers DataFrame.
    """
    # Read in the files
    count_mp_nogwise_branchwise_dfs = [
        (
            f.partition("count.mp.")[2].rpartition(".tsv")[0],
            pd.read_csv(f, sep="\t", usecols=["nog_id", "recipient_branch"]),
        )
        for f in count_mp_files
    ]

    # Print gain_penalty_ratios
    print("Gain penalty ratios:")
    display([i[0] for i in count_mp_nogwise_branchwise_dfs])

    # Sort by the gain_penalty_ratio
    count_mp_nogwise_branchwise_dfs = sorted(
        count_mp_nogwise_branchwise_dfs, key=lambda x: float(x[0])
    )

    # For each df, add a column 'gain_penalty_ratio' with the gain penalty ratio
    count_mp_nogwise_branchwise_dfs = [
        (i, df.assign(transfers=i)) for i, df in count_mp_nogwise_branchwise_dfs
    ]

    # Add a column 'source_branch' with value 'unknown'
    count_mp_nogwise_branchwise_dfs = [
        (i, df.assign(source_branch="unknown"))
        for i, df in count_mp_nogwise_branchwise_dfs
    ]

    # Concatenate the dfs
    count_mp_nogwise_branchwise_df = pd.concat(
        [df for i, df in count_mp_nogwise_branchwise_dfs], ignore_index=True
    )

    return count_mp_nogwise_branchwise_df


def process_count_ml_output(count_ML_output_file: str):

    # read in the file as TSV, except commented lines
    count_ml_output_df = pd.read_csv(
        count_ML_output_file, sep="\t", comment="#", header=0
    ).rename(columns={"Family": "nog_id"})

    # drop the row where nog_id is ABSENT
    count_ml_output_df = count_ml_output_df[count_ml_output_df["nog_id"] != "ABSENT"]

    # first column is the nog_id, second is the numerical distribution (concatenated), third onwards are of the form 'branch:gain/loss/expansion/reduction'
    # new df with only nog_id and the columns ending with 'gain'
    count_ml_gains_df = count_ml_output_df[
        ["nog_id"] + [c for c in count_ml_output_df.columns if c.endswith("gain")]
    ]

    # melt this df to get 'nog_id', 'recipient_branch', 'transfers'
    count_ml_gains_df = count_ml_gains_df.melt(
        id_vars="nog_id", var_name="recipient_branch", value_name="transfers"
    )

    # add a source_branch column
    count_ml_gains_df["source_branch"] = "unknown"
    count_ml_gains_df = count_ml_gains_df[count_ml_gains_df["transfers"] > 0]

    # remove the ':gain' from the recipient_branch column entries
    count_ml_gains_df["recipient_branch"] = count_ml_gains_df[
        "recipient_branch"
    ].str.replace(":gain", "")

    # create a nogwise df by summing the transfers for each nog_id
    count_ml_nogwise_gains_df = count_ml_gains_df.copy()
    count_ml_nogwise_gains_df.loc[:, "transfer_threshold"] = count_ml_nogwise_gains_df[
        "transfers"
    ]
    count_ml_nogwise_gains_df = prepare_nogwise_transfer_thresholds_df(
        count_ml_nogwise_gains_df
    )

    # similarly for losses
    count_ml_losses_df = count_ml_output_df[
        ["nog_id"] + [c for c in count_ml_output_df.columns if c.endswith("loss")]
    ]
    count_ml_losses_df = count_ml_losses_df.melt(
        id_vars="nog_id", var_name="recipient_branch", value_name="losses"
    )
    count_ml_losses_df = count_ml_losses_df[count_ml_losses_df["losses"] > 0]
    count_ml_losses_df = count_ml_losses_df.rename(
        columns={"recipient_branch": "branch"}
    )
    count_ml_losses_df["branch"] = count_ml_losses_df["branch"].str.replace(":loss", "")

    return count_ml_gains_df, count_ml_nogwise_gains_df, count_ml_losses_df


def read_and_compile_gloome_results(
    gloome_output_dir: Union[str, List[str]], # can be a list of gloome output dirs
    pa_matrix_tsv_filepath: str,
    ml_mp: str, # either 'ml' or 'mp'
    input_tree_filepath: str,
):
    """
    Read and compile GLOOME results from the specified output directory.

    Parameters:
    gloome_output_dir: Path to the GLOOME output directory.
                       This can be a list of gloome output dirs (in case of 'mp' mode)
    pa_matrix_tsv_filepath (str): Path to the PA matrix TSV file.
    ml_mp (str): Specify whether to use 'ml' or 'mp' for maximum-likelihood or maximum-parsimony.
                 In case of 'mp', the
    input_tree_filepath (str): Path to the input tree file.
                               This is required if GLOOME was run with a species tree.
                               If not available, set it as None, 
                               and the function will not map the branch names to the input tree.
    
    Returns:
    dict: A dictionary containing compiled GLOOME results.
          The keys are the file names and the values are the corresponding DataFrames.
    """

    # if input_tree_filepath is not None, read in the tree
    input_tree = None
    if input_tree_filepath is not None:
        input_tree = ete3.Tree(input_tree_filepath, format=1)
    else:
        input_tree = None
    # make sure pa_matrix_tsv_filepath is a valid file
    if not os.path.isfile(pa_matrix_tsv_filepath):
        raise ValueError(
            f"PA matrix TSV file {pa_matrix_tsv_filepath} does not exist. Please provide a valid file."
        )

    # if it's mp, make sure gloome_output_dir is a list of dirs
    if ml_mp == "mp":
        if isinstance(gloome_output_dir, str):
            raise ValueError(
                "For maximum-parsimony mode, gloome_output_dir must be a list of directories."
                "These correspond to the different gain penalty ratios used in GLOOME."
            )
        # if gloome_output_dir is a list, make sure it has at least 2 elements
        if len(gloome_output_dir) < 2:
            raise ValueError(
                "For maximum-parsimony mode, gloome_output_dir must be a list of directories."
                "These correspond to the different gain penalty ratios used in GLOOME."
            )
        gloome_results_dict = read_and_compile_mp_gloome_results(
            gloome_output_dir, pa_matrix_tsv_filepath, input_tree
        )
    elif ml_mp == "ml":
        if isinstance(gloome_output_dir, list):
            raise ValueError(
                "For maximum-likelihood mode, gloome_output_dir must be a single directory."
            )
        gloome_results_dict = read_and_compile_ml_gloome_results(
            gloome_output_dir, pa_matrix_tsv_filepath, input_tree
        )
    else:
        raise ValueError(
            "ml_mp must be either 'ml' or 'mp', for maximum-likelihood or maximum-parsimony"
        )
        
    # return the dictionary
    return gloome_results_dict


def read_and_compile_mp_gloome_results(
        gloome_output_dirs: List[str],
        pa_matrix_tsv_filepath: str,
        input_tree: Optional[ete3.Tree] = None,
):
    """
    Read and compile GLOOME results for maximum-parsimony mode.

    Parameters:
    gloome_output_dirs (List[str]): List of paths to GLOOME output directories.
    pa_matrix_tsv_filepath (str): Path to the PA matrix TSV file.
    input_tree (ete3.Tree): ETE3 Tree object representing the input tree.

    Returns:
    dict: A dictionary containing compiled GLOOME results.
          The keys are the file names and the values are the corresponding DataFrames.
    """
    # read in the PA matrix TSV file
    pa_matrix_df = pd.read_csv(pa_matrix_tsv_filepath, sep="\t")
    # create a dict of row number to NOG IDs in pa_matrix_df
    pos_nog_dict = {i + 1: nog for i, nog in enumerate(pa_matrix_df.iloc[:, 0])}

    # create a dict to store the compiled results
    gloome_results_dict = {}
    nogwise_gain_dfs = []
    nogwise_branchwise_gain_dfs = {}

    # process each gloome output dir with corresponding gain penalty ratio
    for gloome_output_dir in gloome_output_dirs:
        gloome_tree = ete3.Tree(
            os.path.join(gloome_output_dir, "TheTree.INodes.ph"), format=1
        )
        if input_tree is not None:
            # if input tree is provided, map the branch names to the input tree
            gloome_node_mapping = map_output_to_input_nodes(input_tree, gloome_tree)
            # write this mapping to a tsv file
            with open(
                os.path.join(gloome_output_dir, "gloome_mp_node_mapping.tsv"),
                "w",
            ) as gloome_node_mapping_fo:
                gloome_node_mapping_fo.write("gloome_node\tinput_tree_node\n")
                for gloome_node, input_tree_node in gloome_node_mapping.items():
                    gloome_node_mapping_fo.write(
                        f"{gloome_node}\t{input_tree_node}\n"
                    )
        else:
            gloome_node_mapping = {}
        # read in the per-position-per-branch expectation files
        all_res_files = [
            f
            for f in os.listdir(gloome_output_dir)
            if f.startswith("gainLossMP") and f.endswith(".txt")
        ]
        per_pos_per_branch_expectations_file_path = [
            os.path.join(gloome_output_dir, f)
            for f in all_res_files
            if f.endswith(".PerPosPerBranch.txt")
        ][0]
        # read in the file: skip commented (#) lines
        per_pos_per_branch_expectations_df = pd.read_csv(
            per_pos_per_branch_expectations_file_path, comment="#", sep="\t"
        ).rename(columns={"branch": "gloome_branch_name"})
        # if input tree is provided, map the gloome branch names to the input tree branch names
        if input_tree is not None:
            per_pos_per_branch_expectations_df["recipient_branch"] = (
                per_pos_per_branch_expectations_df["gloome_branch_name"]
                .map(gloome_node_mapping)
                .astype(str)
            )
        else:
            per_pos_per_branch_expectations_df["recipient_branch"] = (
                per_pos_per_branch_expectations_df["gloome_branch_name"]
            )
        # use the pos_nog_dict to replace the POS column with NOG IDs
        per_pos_per_branch_expectations_df["POS"] = per_pos_per_branch_expectations_df[
            "POS"
        ].map(pos_nog_dict)
        # rename the POS column to nog_id and exp01 column to transfers
        per_pos_per_branch_expectations_df.rename(
            columns={
                "POS": "nog_id",
                "expectation": "transfers",
            },
            inplace=True,
        )
        # retain only rows where G/L is gain
        per_pos_per_branch_expectations_df = per_pos_per_branch_expectations_df[
            per_pos_per_branch_expectations_df["G/L"] == "gain"
        ]
        # add a source_branch column
        per_pos_per_branch_expectations_df["source_branch"] = "unknown"
        # the transfer threshold column is the gain penalty ratio
        # this is contained in the 3rd line of the per_pos_per_branch_expectations_file
        # read the file again to get the gain penalty ratio
        with open(
            per_pos_per_branch_expectations_file_path, "r"
        ) as f:
            gain_penalty_ratio = f.readlines()[2].strip().split("=")[1].strip()
        # add the gain penalty ratio to the df
        per_pos_per_branch_expectations_df["transfer_threshold"] = gain_penalty_ratio
        # retain only the columns nog_id, source_branch, recipient_branch, gloome_branch_name, transfers, transfer_threshold
        nogwise_branchwise_gains_df = per_pos_per_branch_expectations_df[
            [
                "nog_id",
                "source_branch",
                "recipient_branch",
                "gloome_branch_name",
                "transfers",
                "transfer_threshold",
            ]
        ]
        # add this df to the list
        nogwise_branchwise_gain_dfs[gain_penalty_ratio] = nogwise_branchwise_gains_df

        # group by nog_id and sum the transfers, 
        nogwise_gains_df = nogwise_branchwise_gains_df.copy()
        nogwise_gains_df = nogwise_gains_df.groupby("nog_id").sum().reset_index()
        # add the transfer_threshold column to the nogwise_gains_df
        nogwise_gains_df["transfer_threshold"] = gain_penalty_ratio
        nogwise_gains_df = nogwise_gains_df[
            ["nog_id", "transfers", "transfer_threshold"]
        ]
        # add the nogwise_gains_df to the list
        nogwise_gain_dfs.append(nogwise_gains_df)
    
    # concatenate the nogwise_gain_dfs
    nogwise_gains_df = pd.concat(nogwise_gain_dfs, ignore_index=True)
    gloome_results_dict[
        f"compiled_transfers.nogwise.gloome.mp."] = nogwise_gains_df
    
    # for nogwise branchwise, we create a file for each gain penalty ratio
    for gain_penalty_ratio, nogwise_branchwise_gains_df in nogwise_branchwise_gain_dfs.items():
        gloome_results_dict[
            f"compiled_transfers.nogwise.branchwise.gloome.mp.{gain_penalty_ratio}."
        ] = nogwise_branchwise_gains_df
    # return the dictionary
    return gloome_results_dict


def read_and_compile_ml_gloome_results(
    gloome_output_dir: str,
    pa_matrix_tsv_filepath: str,
    input_tree: Optional[ete3.Tree] = None,
):
    """
    Read and compile GLOOME results for maximum-likelihood mode.

    Parameters:
    gloome_output_dir (str): Path to the GLOOME output directory.
    pa_matrix_tsv_filepath (str): Path to the PA matrix TSV file.
    input_tree (ete3.Tree): ETE3 Tree object representing the input tree.

    Returns:
    dict: A dictionary containing compiled GLOOME results.
          The keys are the file names and the values are the corresponding DataFrames.
    """
    # read in the PA matrix TSV file
    pa_matrix_df = pd.read_csv(pa_matrix_tsv_filepath, sep="\t")
    # create a dict of row number to NOG IDs in pa_matrix_df
    pos_nog_dict = {i + 1: nog for i, nog in enumerate(pa_matrix_df.iloc[:, 0])}

    # read in the per-position-per-branch expectation file
    per_pos_per_branch_expectations_file_path = os.path.join(
        gloome_output_dir, "gainLossProbExpPerPosPerBranch.txt"
    )
    # read in the file: skip commented (#) lines
    per_pos_per_branch_expectations_df = pd.read_csv(
        per_pos_per_branch_expectations_file_path, comment="#", sep="\t"
    ).rename(columns={"branch": "gloome_branch_name"})
    # if input tree is provided, map the branch names to the input tree
    if input_tree is not None:
        gloome_tree = ete3.Tree(
            os.path.join(gloome_output_dir, "TheTree.INodes.ph"), format=1
        )
        gloome_node_mapping = map_output_to_input_nodes(input_tree, gloome_tree)
        # write this mapping to a tsv file
        with open(
            os.path.join(gloome_output_dir, "gloome_ml_node_mapping.tsv"),
            "w",
        ) as gloome_node_mapping_fo:
            gloome_node_mapping_fo.write("gloome_node\tinput_tree_node\n")
            for gloome_node, input_tree_node in gloome_node_mapping.items():
                gloome_node_mapping_fo.write(
                    f"{gloome_node}\t{input_tree_node}\n"
                )
        # replace the gloome branch names with the input tree branch names
        per_pos_per_branch_expectations_df["recipient_branch"] = (
            per_pos_per_branch_expectations_df["gloome_branch_name"]
            .map(gloome_node_mapping)
            .astype(str)
        )
    else:
        per_pos_per_branch_expectations_df['recipient_branch'] = per_pos_per_branch_expectations_df[
            "gloome_branch_name"
        ]
    # use the pos_nog_dict to replace the POS column with NOG IDs
    per_pos_per_branch_expectations_df["POS"] = per_pos_per_branch_expectations_df[
        "POS"
    ].map(pos_nog_dict)
    # rename the POS column to nog_id and expectation column to transfers
    per_pos_per_branch_expectations_df.rename(
        columns={
            "POS": "nog_id",
            "expectation": "transfers",
            "probability": "transfer_threshold",
        },
        inplace=True,
    )
    # retain only rows where G/L is gain
    per_pos_per_branch_expectations_df = per_pos_per_branch_expectations_df[
        per_pos_per_branch_expectations_df["G/L"] == "gain"
    ]
    # add a source_branch column
    per_pos_per_branch_expectations_df["source_branch"] = "unknown"
    # retain only the columns nog_id, source_branch, recipient_branch, gloome_branch_name, transfers, transfer_threshold
    nogwise_branchwise_gains_df = per_pos_per_branch_expectations_df[
        [
            "nog_id",
            "source_branch",
            "recipient_branch",
            "gloome_branch_name",
            "transfers",
            "transfer_threshold",
        ]
    ]
    # add this df to dictionary
    gloome_results_dict = {
        f"compiled_transfers.nogwise.branchwise.gloome.ml.": nogwise_branchwise_gains_df
    }

    # concatenate the nogwise_branchwise_gains_df and use prepare_nogwise_transfer_thresholds_df to get nogwise_gains_df
    nogwise_gains_df = nogwise_branchwise_gains_df.copy()
    # Group by nog_id and sum the transfers
    nogwise_gains_df = prepare_nogwise_transfer_thresholds_df(nogwise_gains_df)
    # add the nogwise_gains_df to the dictionary
    gloome_results_dict[
        f"compiled_transfers.nogwise.gloome.ml."] = nogwise_gains_df
    # return the dictionary
    return gloome_results_dict
