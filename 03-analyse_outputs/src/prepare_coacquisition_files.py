import argparse
import itertools
import multiprocessing as mp
from multiprocessing import process
from operator import call
import os
import sys
import time
from collections import defaultdict
from datetime import timedelta
from turtle import st
from unittest import result

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import traceback


def process_transfers_file(
    transfers: str,
    taxon_nog_gene_df: pd.DataFrame,
    gene_features_df: pd.DataFrame,
    taxon_contigs_dict: dict,
    #    members: str, features: str,
    min_threshold: float,
    debug: bool,
):
    """
    Process the compiled_transfers.nogwise.branchwise.{method}{modifier}.tsv file
    to get a DataFrame with columns nog_id, recipient_branch, source_branch, transfers
    """

    # read in the compiled transfers (columns: nog_id, recipient_branch, source_branch, transfers)
    compiled_transfers_df = pd.read_csv(transfers, sep="\t", header=0)
    compiled_transfers_df = compiled_transfers_df[
        compiled_transfers_df["transfers"] >= min_threshold
    ]

    # remove internal recipient branches (they start with 'N')
    compiled_transfers_df["recipient_branch"] = compiled_transfers_df[
        "recipient_branch"
    ].astype(str)
    compiled_transfers_df = compiled_transfers_df[
        ~compiled_transfers_df["recipient_branch"].str.startswith("N")
    ]
    compiled_transfers_df = compiled_transfers_df.sort_values(
        by=["recipient_branch", "nog_id"]
    )
    if debug:
        logger.debug(f"compiled_transfers_df looks like:\n{compiled_transfers_df}")
    # and those that have only one NOG mapped to them (i.e. no coacquisitions)
    compiled_transfers_nog_counts = compiled_transfers_df.groupby("recipient_branch")[
        "nog_id"
    ].nunique()
    compiled_transfers_df = compiled_transfers_df[
        compiled_transfers_df["recipient_branch"].isin(
            compiled_transfers_nog_counts[compiled_transfers_nog_counts > 1].index
        )
    ]
    if debug:
        logger.debug(f"compiled_transfers_df looks like:\n{compiled_transfers_df}")

    # keep only those transfers in compiled_transfers_df that are in nog_id column of taxon_nog_gene_df
    nogs_to_keep = taxon_nog_gene_df["nog_id"].unique()
    compiled_transfers_df = compiled_transfers_df[
        compiled_transfers_df["nog_id"].isin(nogs_to_keep)
    ]

    # keep only those transfers where recipient_branch is in keys of taxon_contigs_dict
    taxa_to_keep = list(taxon_contigs_dict.keys())
    compiled_transfers_df = compiled_transfers_df[
        compiled_transfers_df["recipient_branch"].isin(taxa_to_keep)
    ]
    logger.debug(
        f"taxa to keep include: {taxa_to_keep[:5]}, of type: {type(taxa_to_keep[0])}\
                 and compiled_transfers_df looks like:\n {compiled_transfers_df.head()}"
    )

    return compiled_transfers_df


def process_features_file(features_filepath: str):

    # read in the gene features file
    gene_features_df = pd.read_csv(features_filepath, sep="\t", header=0)
    gene_features_df["taxon_id"] = gene_features_df["taxon_id"].astype(str)
    gene_features_df["gene_id"] = (
        gene_features_df["taxon_id"] + "." + gene_features_df["locus_tag"]
    )

    gene_features_df.rename(columns={"seqid": "contig"}, inplace=True)

    # set up a contigwise_index column that indexes the gene_id within each contig
    gene_features_df["contigwise_index"] = gene_features_df.groupby("contig").cumcount()
    # another column 'contig_genome_size'. This is the size of the contig
    contig_genome_sizes = (
        gene_features_df.groupby("contig")["gene_id"].count().to_dict()
    )
    gene_features_df["contig_genome_size"] = gene_features_df["contig"].map(
        contig_genome_sizes
    )

    # create a dict of which contigs are in each taxon
    taxon_contigs_dict = (
        gene_features_df.groupby("taxon_id")["contig"].apply(set).to_dict()
    )

    return gene_features_df, taxon_contigs_dict


def process_members_file(members_filepath: str):
    """
    Process the members.tsv file from EggNOG, to get a df with columns nog_id, taxonomic_id, genes_csv,
    """

    members_df = pd.read_csv(
        members_filepath,
        sep="\t",
        usecols=[1, 5],
        names=["nog_id", "genes_csv"],
        dtype={"nog_id": str, "genes_csv": str},
    )
    # split the genes_csv column into a list of genes
    members_df.loc[:, "genes"] = members_df["genes_csv"].str.split(",")
    members_df = members_df.drop(columns=["genes_csv"]).explode("genes")
    members_df.loc[:, "taxon"] = (
        members_df["genes"].str.split(".").str[0]
    )  # get the taxon_id from the gene

    # we want a df that maps, for every 'taxon', and every 'nog_id', a csv of genes
    taxon_nog_gene_df = (
        members_df.groupby(["taxon", "nog_id"])["genes"].apply(",".join).reset_index()
    )
    taxon_nog_gene_df.to_csv(
        f"{os.path.dirname(
        members_filepath)}/taxon_nog_genes_map.tsv",
        sep="\t",
        index=False,
    )

    return taxon_nog_gene_df


def process_gene_pair(gene_pair_args):

    gene1, gene2, gene1_location, gene2_location = gene_pair_args
    try:
        upstream_gene_location = (
            gene1_location
            if gene1_location["start"] < gene2_location["start"]
            else gene2_location
        )
        downstream_gene_location = (
            gene2_location
            if gene1_location["start"] < gene2_location["start"]
            else gene1_location
        )
        pair_distance = (
            downstream_gene_location["start"] - upstream_gene_location["end"]
        )
        # since the gene locations df already contains the contigwise_index
        # we can just subtract the contigwise_index of the downstream gene from the upstream gene
        num_genes_between = (
            downstream_gene_location["contigwise_index"]
            - upstream_gene_location["contigwise_index"]
            - 1
        )

    except Exception as e:
        logger.error(
            f"Error processing gene pair {gene1} and {gene2} on contig {gene1_location['contig']}:\n{e}"
        )
        raise e

    return (num_genes_between, pair_distance)


def process_nog_pair(nog_pair_args):

    nog_pair_coacquisitions_records_list = []

    # first check if we have locations available for both genes
    if not nog_pair_args["genes"][0] or not nog_pair_args["genes"][1]:
        # this is case 1: gene positions are unknown for one or both genes
        nog_pair_coacquisitions_records_list.append(
            (
                nog_pair_args["recipient_branch"],  # str
                # nog1, nog2
                nog_pair_args["nog_pair"][0],
                nog_pair_args["nog_pair"][1],
                "NA",
                "NA",  # gene1, gene2
                "NA",
                "NA",  # num_genes_between, pair_distances
                # source_overlap, cotransfers
                nog_pair_args["source_overlap_str"],
                nog_pair_args["cotransfers"],
                "NA",
                "NA",  # pair_distances_string, contig:genome_size
                nog_pair_args["nog1_transfers"],
                nog_pair_args["nog2_transfers"],
                "gene_positions_not_available",
            )
        )
        nog_pair_coacquisitions_records_list = [
            [str(i) for i in record] for record in nog_pair_coacquisitions_records_list
        ]
        return nog_pair_coacquisitions_records_list

    # find all pairs of genes, one from each nog, that have been coacquired
    gene_pairs = list(
        itertools.product(nog_pair_args["genes"][0], nog_pair_args["genes"][1])
    )
    # filter out the genes that are not in the same contig
    gene_pairs = [
        gene_pair
        for gene_pair in gene_pairs
        if nog_pair_args["gene_locations_df"].loc[gene_pair[0], "contig"]
        == nog_pair_args["gene_locations_df"].loc[gene_pair[1], "contig"]
    ]

    # if there are no gene pairs in the list of genes in the same contig, this is case 2
    if not gene_pairs:
        nog_pair_coacquisitions_records_list.append(
            (
                nog_pair_args["recipient_branch"],  # str
                # nog1, nog2
                nog_pair_args["nog_pair"][0],
                nog_pair_args["nog_pair"][1],
                "NA",
                "NA",  # gene1, gene2
                "NA",
                "NA",  # num_genes_between, pair_distances
                # source_overlap, cotransfers
                nog_pair_args["source_overlap_str"],
                nog_pair_args["cotransfers"],
                "NA",
                "NA",  # pair_distances_string, contig:genome_size
                nog_pair_args["nog1_transfers"],
                nog_pair_args["nog2_transfers"],
                "genes_on_different_contigs",
            )
        )
    else:
        # this is case 3: gene positions are known for both genes and they are on the same contig
        for gene_pair in gene_pairs:
            gene1_location = nog_pair_args["gene_locations_df"].loc[gene_pair[0]]
            gene2_location = nog_pair_args["gene_locations_df"].loc[gene_pair[1]]
            # the gene locations df for this contig
            this_contig = gene1_location["contig"]
            this_contig_genome_size = gene1_location["contig_genome_size"]
            this_contig_genome_size_str = f"{
                this_contig}:{this_contig_genome_size}"
            num_genes_between, pair_distance = process_gene_pair(
                (gene_pair[0], gene_pair[1], gene1_location, gene2_location)
            )

            nog_pair_coacquisitions_records_list.append(
                (
                    nog_pair_args["recipient_branch"],  # str
                    # nog1, nog2
                    nog_pair_args["nog_pair"][0],
                    nog_pair_args["nog_pair"][1],
                    # gene1, gene2
                    gene_pair[0],
                    gene_pair[1],
                    num_genes_between,
                    pair_distance,  # num_genes_between, pair_distances
                    # source_overlap, cotransfers
                    nog_pair_args["source_overlap_str"],
                    nog_pair_args["cotransfers"],
                    f"{gene_pair[0]}:{gene_pair[1]}:{
                pair_distance}",
                    this_contig_genome_size_str,
                    nog_pair_args["nog1_transfers"],
                    nog_pair_args["nog2_transfers"],
                    "gene_positions_are_available",
                )
            )

    # convert all the results to a list of strings
    nog_pair_coacquisitions_records_list = [
        [str(i) for i in record] for record in nog_pair_coacquisitions_records_list
    ]

    return nog_pair_coacquisitions_records_list


def prepare_coacquisition_files(
    method_name: str,
    compiled_transfers_df: pd.DataFrame,
    compiled_results_dir: str,
    gene_locations_df: pd.DataFrame,
    taxon_nog_gene_df: pd.DataFrame,
    # num_intervening_genes_dict: dict,
    min_threshold: float,
    sources_available: bool,
    max_processes: int,
    debug: bool,
):
    """
    Prepare coacquisition files for a given method, using the compiled_transfers.nogwise.branchwise.{method}{modifier}.tsv file.
    """

    # set up coacquisition file to write to
    coacquisition_file_header = [
        "recipient_branch",
        "nog1",
        "nog2",
        "gene1",
        "gene2",
        "num_genes_between",
        "pair_distances",
        "source_overlap",
        "cotransfers",
        "pair_distances_string",
        "contig:genome_size",
        "nog1_transfers",
        "nog2_transfers",
        "notes",
    ]
    coacquisition_filename = f"{
        compiled_results_dir}/coacquisitions.{method_name}.tsv"
    with open(coacquisition_filename, "w") as f:
        f.write("\t".join(coacquisition_file_header) + "\n")
    logger.info(f"Writing coacquisitions to {coacquisition_filename}")

    # get the recipient branches
    recipient_branches = compiled_transfers_df["recipient_branch"].unique()

    if debug:
        logger.debug(
            f"Debug mode: only processing first 5 recipient branches out of {len(recipient_branches)}"
        )
        recipient_branches = recipient_branches[:5]
        compiled_transfers_df = compiled_transfers_df[
            compiled_transfers_df["recipient_branch"].isin(recipient_branches)
        ]

    # create a map from recipient branch to transfers df for that branch, as a key-value pair
    recipient_branchwise_dfs_dict = {
        recipient_branch: group
        for recipient_branch, group in compiled_transfers_df.groupby("recipient_branch")
    }
    logger.info(
        f"Number of recipient branches to process: {
                len(recipient_branches)}"
    )

    # first we go through all the items in this dict
    # find all the gene pairs across all the nog pairs that have been coacquired
    # one of three cases can happen:
    # 1. gene positions are unknown for one or both genes
    # 2. gene positions are known for both genes but they are on different contigs
    # 3. gene positions are known for both genes and they are on the same contig
    # for case 1 and 2 we can't do anything, so we note the coacquisition with a note on which case it is and move on
    # for case 3 we calculate the distance and number of genes between the two genes in the pair
    # additionally, if sources are available for the method,
    # we note down whether the coacquisition is a 'cotransfer' (i.e. the list of 'source_branch' entries across both nog pairs overlaps)

    # Index dataframes
    taxon_nog_gene_dict = {
        (row["taxon"], row["nog_id"]): row["genes"].split(",")
        for _, row in taxon_nog_gene_df.iterrows()
    }
    # taxa_nogs_set = set(taxon_nog_gene_dict.keys())
    gene_locations_df = gene_locations_df.set_index("gene_id").copy()
    genes_with_locations_set = set(gene_locations_df.index)
    logger.debug(
        f"taxon_nog_gene_dict looks like:\n{
        '\n'.join([f'{k}: {v}' for k, v in list(taxon_nog_gene_dict.items())[:10]])}"
    )
    logger.debug(
        f"genes_with_locations_set looks like: {
        list(genes_with_locations_set)[:10]}"
    )
    logger.debug(f"gene_locations_df looks like:\n{gene_locations_df.head()}")

    # what fraction of the genes in the taxon_nog_gene_df have locations?
    genes_in_taxon_nog_gene_df = set(taxon_nog_gene_df["genes"].explode().unique())
    genes_with_locations_set = genes_with_locations_set & genes_in_taxon_nog_gene_df
    logger.debug(
        f"Fraction of genes in taxon_nog_gene_df that have locations is the ratio of\
                 genes_with_locations_set that looks like {list(genes_with_locations_set)[:10]} to genes_in_taxon_nog_gene_df that looks like {list(genes_in_taxon_nog_gene_df)[:10]}\
                 This fraction is {len(genes_with_locations_set) / len(genes_in_taxon_nog_gene_df)}. Some of those genes are:\n {list(genes_with_locations_set)[:20]}"
    )

    # find and display location records for
    # ['1005057.BUAMB_125', '1005057.BUAMB_234', '1005057.BUAMB_274', '1005057.BUAMB_432', '1005057.BUAMB_398',
    # '1005057.BUAMB_028', '1005057.BUAMB_061', '1005057.BUAMB_257', '1005057.BUAMB_545', '1005057.BUAMB_530']
    # logger.debug(f"Gene locations for some specific genes: {gene_locations_df.loc[['1005057.BUAMB_125', '1005057.BUAMB_234', '1005057.BUAMB_274',
    #              '1005057.BUAMB_432', '1005057.BUAMB_398', '1005057.BUAMB_028', '1005057.BUAMB_061', '1005057.BUAMB_257', '1005057.BUAMB_545', '1005057.BUAMB_530']]}")
    # logger.debug(f"These specific genes are in genes_with_locations_set: {set(['1005057.BUAMB_125', '1005057.BUAMB_234', '1005057.BUAMB_274', '1005057.BUAMB_432',
    #              '1005057.BUAMB_398', '1005057.BUAMB_028', '1005057.BUAMB_061', '1005057.BUAMB_257', '1005057.BUAMB_545', '1005057.BUAMB_530']) & genes_with_locations_set}")

    # for each recipient branch
    for recipient_branch, recipient_branch_df in tqdm(
        recipient_branchwise_dfs_dict.items(),
        desc=f"Processing {method_name} coacquisitions",
    ):
        # nogs acquired by this branch
        acquired_nogs = set(recipient_branch_df["nog_id"].unique())
        acquired_nog_pairs = list(
            itertools.combinations(acquired_nogs, 2)
        )  # all possible pairs
        logger.debug(
            f"Number of acquired nog pairs: {
                     len(acquired_nog_pairs)} for recipient branch {recipient_branch}"
        )
        if sources_available:
            nogwise_sources = (
                recipient_branch_df.groupby("nog_id")["source_branch"]
                .apply(set)
                .to_dict()
            )
        else:
            nogwise_sources = {}
        if debug:
            # get all genes for this branch
            dbg_this_branch_genes = {
                k: v for k, v in taxon_nog_gene_dict.items() if k[0] == recipient_branch
            }
            logger.debug(
                f"For recipient branch {
                         recipient_branch}, the genes are: {dbg_this_branch_genes}"
            )
        # we process all the nog pairs in parallel
        nog_pair_process_results = []
        with mp.Manager() as coacquisitions_records_list_manager:
            coacquisitions_records_list = coacquisitions_records_list_manager.list()
            with mp.Pool(processes=max_processes) as pool:
                if debug:
                    pbar_queue = tqdm(
                        total=len(acquired_nog_pairs),
                        desc=f"Queueing {
                        method_name} nog pairs",
                    )
                    pbar_process = tqdm(
                        total=len(nog_pair_process_results),
                        desc=f"Processing {
                        method_name} nog pairs",
                        position=1,
                    )  # position 1 is below the queueing progress bar

                # loop over the nog pairs, and process them in parallel by using `apply_async`
                for nog_pair in acquired_nog_pairs:
                    try:
                        # retrieve the genes for the nog pair. taxon_nog_gene_df was prepared from the members.tsv file
                        # if there are no genes for this nog pair, skip it
                        nog1_genes = taxon_nog_gene_dict.get(
                            (recipient_branch, nog_pair[0]), []
                        )
                        nog2_genes = taxon_nog_gene_dict.get(
                            (recipient_branch, nog_pair[1]), []
                        )
                    except KeyError as ke:
                        logger.error(
                            f"KeyError for recipient branch {recipient_branch} and nog pair {nog_pair} as {ke}"
                        )
                        traceback.print_exc()
                        continue
                    # only keep the genes for which we have gene locations: set intersection of nogx_genes and genes_with_locations_set
                    nog1_genes_set, nog2_genes_set = set(nog1_genes), set(nog2_genes)
                    nog1_genes = list(nog1_genes_set & genes_with_locations_set)
                    nog2_genes = list(nog2_genes_set & genes_with_locations_set)
                    if debug:
                        if not nog1_genes or not nog2_genes:
                            continue  # debug:skip this nog pair if we don't have gene locations for genes in either nog

                    nog1_transfers = recipient_branch_df.loc[
                        recipient_branch_df["nog_id"] == nog_pair[0], "transfers"
                    ].values[  # type: ignore
                        0
                    ]
                    nog2_transfers = recipient_branch_df.loc[
                        recipient_branch_df["nog_id"] == nog_pair[1], "transfers"
                    ].values[  # type: ignore
                        0
                    ]
                    # debug
                    if debug:
                        logger.debug(
                            f"For recipient branch {
                                     recipient_branch}, nog pair {nog_pair}, nog1_transfers: {nog1_transfers}, nog2_transfers: {nog2_transfers}"
                        )
                    if sources_available:
                        nog1_sources, nog2_sources = (
                            nogwise_sources[nog_pair[0]],
                            nogwise_sources[nog_pair[1]],
                        )
                        source_overlap = nog1_sources & nog2_sources
                        cotransfers = len(source_overlap)
                        source_overlap_str = (
                            ",".join(source_overlap) if cotransfers > 0 else "None"
                        )
                    else:
                        source_overlap_str = "NA"
                        cotransfers = "NA"
                    nog_pair_args = {
                        "nog_pair": nog_pair,
                        "recipient_branch": recipient_branch,
                        "genes": (nog1_genes, nog2_genes),
                        "recipient_branch_df": recipient_branch_df.loc[
                            recipient_branch_df["nog_id"].isin(nog_pair)
                        ],
                        # only provide the subset of the gene_locations_df that contains the genes in this nog pair (i.e. nog1_genes+ nog2_genes)
                        "gene_locations_df": gene_locations_df.loc[
                            nog1_genes + nog2_genes
                        ],
                        "source_overlap_str": source_overlap_str,
                        "cotransfers": cotransfers,
                        "nog1_transfers": nog1_transfers,
                        "nog2_transfers": nog2_transfers,
                    }

                    result = pool.apply_async(process_nog_pair, args=(nog_pair_args,))
                    nog_pair_process_results.append(result)
                    if debug:
                        pbar_queue.update(1)  # type: ignore

                # get the results from the pool
                for result in nog_pair_process_results:
                    result.wait()
                    coacquisitions_records_list.extend(result.get())
                    if debug:
                        pbar_process.update(1)  # type: ignore

                if debug:
                    pbar_queue.close()  # type: ignore
                    pbar_process.close()  # type: ignore

            # write coacquisitions_records_list to the coacquisition file
            with open(coacquisition_filename, "a") as f:
                f.writelines(
                    "\t".join(record) + "\n" for record in coacquisitions_records_list
                )

    logger.info(
        f"Finished processing {method_name}. Output written to {
                coacquisition_filename}"
    )

    return 0


if __name__ == "__main__":
    # start tracking time
    start_time = time.time()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        "-f",
        type=str,
        required=False,
        default="../data/1236_gene_features.tsv",
        help="(str) Default: ../data/1236_gene_features.tsv\
                        Path to the file containing the gene features. This should include columns for locus_tag, taxon_id, seqid (i.e. contig ID), start, end, and strand.",
    )
    parser.add_argument(
        "--members",
        "-m",
        type=str,
        required=False,
        default="../data/1236_nog_members.tsv",
        help="(str) Default: ../data/1236_nog_members.tsv\
                        Path to the file containing the members.tsv file from EggNOG. This should include columns taxonomic_id, nog_id, #taxa, #genes, taxa_csv, genes_csv.",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        required=False,
        default=100,
        help="(int) Default: 100\
                        Number of processes at max to use for parallel processing.",
    )
    parser.add_argument(
        "--compiled_results_dir",
        "-c",
        type=str,
        required=False,
        default="../data/compiled_results/",
        help="(str) Default: ../data/compiled_results/\
                        Path to the directory containing the compiled results. Files with names of the form 'compiled_transfers.nogwise.branchwise.{method}{modifier}.tsv' should be present in this directory.",
    )
    parser.add_argument(
        "--methods_with_sources",
        "-ms",
        type=str,
        default="ale,angst,ranger",
        help="(str) Default: ale,angst,ranger \
            Comma-separated list of HGT inference methods for which sources are available. \
            Taken from compiled_transfers.nogwise.branchwise.{method}{modifier}.tsv filenames in compiled_results_dir",
    )
    parser.add_argument(
        # don't skip making existing coacquisitions files by default
        "--skip_existing",
        "-s",
        action="store_true",
        default=False,
        help="(flag) Default: False \
            Skip making coacquisitions files that already exist in the output directory.",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="(flag) Default: False \
                            Enable debug mode. Makes program run more verbose, \
                            and only the first 5 recipient branches are processed, for the first 3 compiled_transfers files.",
    )

    args = parser.parse_args()

    # get timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # set up logger, also based on debug mode or not
    logger.remove()
    if args.debug:
        logger_level = "DEBUG"
        # add a stream handler to print logs to stdout
        logger.add(sys.stdout, level=logger_level)
    else:
        logger_level = "INFO"
    logger.add(
        f"tmp_prepare_coacquisition_files_{
               timestamp}.log",
        level=logger_level,
    )
    logger.info(
        f"Starting prepare_coacquisition_files.py with args: {
        "\n".join([f'{k}: {v}' for k, v in vars(args).items()])}"
    )

    min_branch_transfer_threshold = 0.1

    # all the compiled_transfers.nogwise.branchwise.{method}{modifier}.tsv files in compiled_results_dir
    compiled_transfers_filepaths = [
        f"{args.compiled_results_dir}/{f}"
        for f in os.listdir(args.compiled_results_dir)
        if f.startswith("compiled_transfers.nogwise.branchwise.") and f.endswith(".tsv")
    ]
    # sort the files by name
    compiled_transfers_filepaths = sorted(compiled_transfers_filepaths)
    if args.debug:
        compiled_transfers_filepaths = compiled_transfers_filepaths[:3]
        # compiled_transfers_filepaths = [
        #     i for i in compiled_transfers_filepaths if 'count' in i]
        logger.debug(f"Debug mode: only processing first 3 files")
    logger.info(f"compiled_transfers_filepaths: {compiled_transfers_filepaths}")

    # process members file and features file
    processed_taxon_nog_gene_df = process_members_file(args.members)
    gene_features_df, taxon_contigs_dict = process_features_file(args.features)
    logger.debug(
        f"processed_taxon_nog_gene_df looks like:\n{processed_taxon_nog_gene_df}"
    )
    logger.debug(f"gene_features_df looks like:\n{gene_features_df}")

    for transfers_filepath in compiled_transfers_filepaths:
        method = transfers_filepath[
            transfers_filepath.find("compiled_transfers.nogwise.branchwise.")
            + len("compiled_transfers.nogwise.branchwise.") : transfers_filepath.find(
                ".tsv"
            )
        ]
        logger.info(f"Processing {method} file {transfers_filepath}")

        # skip this method if coacquisitions file already exists
        coacquisition_files = [
            f"{args.compiled_results_dir}/{g}"
            for g in os.listdir(args.compiled_results_dir)
            if g.startswith(f"coacquisitions.{method}.") and g.endswith(".tsv")
        ]
        if args.skip_existing and coacquisition_files:
            logger.info(
                f"Skipping {method} because the following files already exist:\n{coacquisition_files}"
            )
            continue

        transfers_df = process_transfers_file(
            transfers=transfers_filepath,
            taxon_nog_gene_df=processed_taxon_nog_gene_df,
            gene_features_df=gene_features_df,
            taxon_contigs_dict=taxon_contigs_dict,
            min_threshold=min_branch_transfer_threshold,
            debug=args.debug,
        )
        logger.debug(f"For {method}, transfers_df:\n{transfers_df.head()}")

        prepare_coacquisition_files(
            method_name=method,
            compiled_transfers_df=transfers_df,
            compiled_results_dir=args.compiled_results_dir,
            gene_locations_df=gene_features_df,
            taxon_nog_gene_df=processed_taxon_nog_gene_df,
            # num_intervening_genes_dict=processed_intervening_genes_dict,
            min_threshold=min_branch_transfer_threshold,
            # if method startswith any that exist in methods_with_sources, then sources are available
            sources_available=bool(
                any(method.startswith(m) for m in args.methods_with_sources.split(","))
            ),
            max_processes=args.processes,
            debug=args.debug,
        )
        logger.info(
            f"Finished processing {transfers_filepath} in {
                    timedelta(seconds=time.time() - start_time)}"
        )

    logger.info(
        f"Finished preparing all coacquisition files in {
                timedelta(seconds=time.time() - start_time)}"
    )
