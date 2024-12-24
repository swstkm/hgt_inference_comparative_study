#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from scipy.signal import savgol_filter
from scipy import stats
import pandas as pd
from Bio import SeqIO
import multiprocessing as mp
from datetime import datetime
import subprocess
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import shutil
import sys
from scipy.ndimage import label


def extract_gene_sequences(gene_positions, genome_path):

    # note  that genome in this case is a multi-sequence fasta file, with one entry for each chromosome/contig
    # gene_positions contains 'seqid' column which is the chromosome/contig name

    # read the genome file
    genome = SeqIO.index(genome_path, "fasta")

    # extract the gene sequences from the genome
    gene_seqs_dict = {}
    for locus_tag, gene in gene_positions.iterrows():
        if gene["seqid"] in genome:
            if gene["strand"] == "+":
                gene_seq = genome[gene["seqid"]].seq[gene["start"]:gene["end"]]
            elif gene["strand"] == "-":
                gene_seq = genome[gene["seqid"]].seq[gene["start"]:gene["end"]].reverse_complement()
            else:
                raise ValueError(
                    "Strand must be + or -. Check the gene_positions file for gene_id: {gene_id}")
            gene_seqs_dict[locus_tag] = gene_seq
        else:
            raise ValueError(f"For locus_tag: {locus_tag}, seqid: {
                             gene['seqid']} not found in the genome file {genome_path}")

    return gene_seqs_dict


def calculate_gene_kmer_counts(locus_tag, gene_seqs_dict, genome_basename, template_size, tmp_dir, kmc_bin):

    # write the gene_seq to a temporary FASTA file with the locus_tag and genome_basename as identifiers
    gene_fasta_path = f"{tmp_dir}/{genome_basename}_{locus_tag}.fna"
    # write the gene sequence to the temporary fasta file, with an EOL character at the end
    with open(gene_fasta_path, "w") as f:
        f.write(f">{genome_basename} {locus_tag}\n{gene_seqs_dict}\n")

    # find the frequency of each n-oligonucleotide in the gene using kmc
    kmc_command = f"{kmc_bin}/kmc -hp -k{template_size} -m2 -ci1 -cs1000000 -fm -hp -r {
        gene_fasta_path} {tmp_dir}/kmc_db_{genome_basename}_{locus_tag} {tmp_dir} > /dev/null"
    subprocess.run(kmc_command, shell=True, check=True)
    kmc_dump_command = f"{kmc_bin}/kmc_tools -hp transform {tmp_dir}/kmc_db_{genome_basename}_{
        locus_tag} dump {tmp_dir}/{genome_basename}_{locus_tag}.gene_kmer_counts.txt > /dev/null"
    subprocess.run(kmc_dump_command, shell=True, check=True)
    gene_kmc = pd.read_csv(f"{tmp_dir}/{genome_basename}_{locus_tag}.gene_kmer_counts.txt",
                           sep="\t", header=None, names=["kmer", locus_tag]).set_index("kmer")
    gene_kmc[locus_tag] = gene_kmc[locus_tag].astype(int)

    return locus_tag, gene_kmc


def plot_derivatives(Typicality_df, tmp_dir, genome_basename):

    try:
        # Select the first frac of the DataFrame
        frac = 0.2
        to_plot_df = Typicality_df.head(int(frac*Typicality_df.shape[0]))
        # Create a new figure
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        alpha = 0.3

        # Plot the derivatives
        ax1.plot(to_plot_df["Typicality_rank"], to_plot_df["Typicality_derivative"],
                 label="Derivative", color="blue", alpha=alpha)
        ax2.plot(to_plot_df["Typicality_rank"], to_plot_df["Typicality_double_derivative"],
                 label="Double Derivative", color="red")
        ax1.plot(to_plot_df["Typicality_rank"], to_plot_df["Typicality_derivative_smoothed"],
                 label="Smoothed Derivative", color="green", alpha=alpha)
        ax2.plot(to_plot_df["Typicality_rank"], to_plot_df["Typicality_smoothed_double_derivative"],
                 label="Smoothed Double Derivative", color="purple")

        # highlight the HGT genes in the plot, for the lowest stringency level
        highest_stringency_col = [
            col for col in to_plot_df.columns if "HGT (stringency level=" in col][0]
        highest_stringency_level = float(highest_stringency_col.split(
            "HGT (stringency level=")[1].split(")")[0])
        hgt_genes = to_plot_df[to_plot_df[highest_stringency_col] == True]
        ax2.plot(hgt_genes["Typicality_rank"],
                 hgt_genes["Typicality_smoothed_double_derivative"], color="black")
        ax1.axvline(x=to_plot_df[to_plot_df[highest_stringency_col] == True]["Typicality_rank"].iloc[-1],
                    color="black", linestyle="--", label=f"HGT threshold (stringency level={highest_stringency_level})")

        # Set labels
        ax1.set_xlabel("Rank (sorted by Typicality)")
        ax1.set_ylabel("Derivative", color="blue")
        ax2.set_ylabel("Double Derivative", color="red")

        # Enable grid and add legend
        ax1.grid()
        ax2.grid()
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        fig.tight_layout()

        # Save the figure
        plt.savefig(f"{tmp_dir}/{genome_basename}_Typicality.png")
        logger.info(f"Saved the derivatives plot for genome {genome_basename} to {
                    tmp_dir}/tmp_{genome_basename}_Typicality.png")

    except Exception as e:
        logger.error(f"Error plotting derivatives: {
                     e}, for genome {genome_basename}")
    finally:
        # Close the figure
        plt.close()


def plot_threshold(half_typicality_df, threshold, threshold_index, genome_basename, tmp_dir, logger, stringency):

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(half_typicality_df["Typicality_rank"],
               half_typicality_df["Typicality_smoothed_double_derivative"], label="Double Derivative", color="red")
    ax[0].axhline(y=threshold, color="black",
                  linestyle="--", label=f"Threshold")
    ax[0].axhline(y=-threshold, color="black", linestyle="--")
    ax[0].axvline(x=threshold_index, color="green",
                  linestyle="--", label="Threshold")
    hgt_genes = half_typicality_df[half_typicality_df[f"HGT (stringency level={
        stringency})"] == True]
    ax[0].plot(hgt_genes["Typicality_rank"],
               hgt_genes["Typicality_smoothed_double_derivative"], color="black", label="HGT genes")

    ax[0].set_xlabel("Rank (sorted by Typicality)")
    ax[0].set_ylabel("Double Derivative")
    ax[0].legend(loc="upper right")
    ax[0].grid()

    ax[1].hist(half_typicality_df["Typicality_smoothed_double_derivative"],
               bins=50, color="red", alpha=0.5)
    ax[1].axvline(x=threshold, color="black",
                  linestyle="--", label="Threshold")
    ax[1].axvline(x=-threshold, color="black", linestyle="--")
    ax[1].set_xlabel("Double Derivative")
    ax[1].set_ylabel("Frequency")
    ax[1].legend(loc="upper right")
    ax[1].grid()

    plt.savefig(f"{tmp_dir}/{genome_basename}_HGT_threshold_{stringency}.png")
    logger.info(f"Saved the HGT threshold plot for genome {genome_basename} to {
                tmp_dir}/{genome_basename}_HGT_threshold_{stringency}.png for stringency level {stringency}")
    plt.close()


def find_threshold(Typicality_df, logger, genome_basename, stringency, tmp_dir, debug):
    """
    Find the threshold on the derivative such that everything before it is atypical HGT genes.

    Parameters:
    Typicality_df (DataFrame): DataFrame with calculated typicality for each gene.
    logger (Logger): Logger for logging information and warnings.
    genome_basename (str): Base name of the genome.
    stringency (array): Array of stringency values for the threshold.
                        We calculate a new threshold, for each stringency value.

    tmp_dir (str): Output directory for saving plots, or temporary files.
    debug (bool): If True, generate debug plots.

    Returns:
    DataFrame: DataFrame with calculated typicality for each gene.

    Notes:
    The threshold is found by taking the median absolute deviation (MAD) of the double derivative of the smoothed typicality values.
    The threshold is then set as `stringency` times the MAD. This stringency value is recommended to be at least 2.5.
    The threshold is the point after which the double derivative becomes approximately zero, i.e. the derivative becomes approximately constant.
    This means that the typicality values are no longer increasing rapidly, but are increasing at a constant rate w.r.t the background, which is the genome.

    """

    Typicality_df = Typicality_df.reset_index()

    # since the genes are sorted by typicality, we can take the first half of the genes
    # there is too much noise at the end
    # take the first half of the genes since we are concerned with phase 1 and 2 only
    half_typicality_df = Typicality_df.iloc[:int(Typicality_df.shape[0]/2)]

    # for each stringency value, find the threshold, and mark the HGT genes that are below the threshold
    for s in stringency:
        threshold = s * \
            np.median(
                np.abs(half_typicality_df["Typicality_smoothed_double_derivative"]))
        logger.debug(f"Threshold for genome {
                     genome_basename} at stringency {s} is {threshold}")

        # we find the threshold by finding the first region where the double derivative is approximately zero
        label_array, num_features = label(
            half_typicality_df["Typicality_smoothed_double_derivative"] < threshold)
        region_sizes = np.bincount(label_array)
        # if there are more than one regions, we take the longest region as the threshold
        if len(region_sizes) > 1:
            longest_region = np.argmax(region_sizes[1:]) + 1
            threshold_index = half_typicality_df[label_array ==
                                                 longest_region].index[0]
            Typicality_df.loc[:, f"HGT (stringency level={s})"] = False
            Typicality_df.loc[Typicality_df.index < threshold_index, f"HGT (stringency level={
                s})"] = True
            half_typicality_df.loc[:, f"HGT (stringency level={s})"] = False
            half_typicality_df.loc[half_typicality_df.index <
                                   threshold_index, f"HGT (stringency level={s})"] = True
            logger.info(f"Found {Typicality_df[Typicality_df[f'HGT (stringency level={s})']].shape[0]} HGT genes out of {
                        Typicality_df.shape[0]} genes, for genome {genome_basename} at stringency {s}")
        else:
            logger.error(f"No threshold found for genome {genome_basename} at stringency {
                         s}. All genes are marked as non-HGT.")
            Typicality_df.loc[:, f"HGT (stringency level={s})"] = False
            half_typicality_df.loc[:, f"HGT (stringency level={s})"] = False

        # if debug is enabled, plot the derivatives, and save the plot
        if debug:
            logger.debug(f"Plotting threshold for genome {
                         genome_basename} at stringency {s}")
            plot_threshold(half_typicality_df, threshold,
                           threshold_index, genome_basename, tmp_dir, logger, s)

    return Typicality_df


def identify_hgt(gene_seqs_dict, kmer_counts, gene_positions, logger, debug, tmp_dir, genome_basename, stringency):
    """
    Identify HGT genes based on typicality values, by using the derivative of the rank-ordered typicality values.

    Parameters:
    gene_seqs_dict (dict): Dictionary with gene sequences.
    kmer_counts (dict): Dictionary with kmer counts for each gene.
    gene_positions (DataFrame): DataFrame with gene positions.
    logger (Logger): Logger for logging information and warnings.
    debug (bool): If True, debug information will be printed.
    tmp_dir (str): Output directory for saving plots, or temporary files.
    genome_basename (str): Base name of the genome.
    stringency (float): The stringency of the threshold for marking HGT genes.

    Returns:
    DataFrame: DataFrame with calculated typicality for each gene.
    """
    # Initialize dictionary for storing typicality
    Typicality_dict = {}

    # Iterate over all genes
    for locus_tag in gene_seqs_dict.keys():
        # Check for inf values in kmer counts and replace them with 0
        if np.isinf(kmer_counts[locus_tag]).any():
            logger.warning(f"Gene {locus_tag} has inf values in kmer counts. Those look like:\n\
                {kmer_counts[locus_tag][kmer_counts[locus_tag] == np.inf]}")
            kmer_counts[locus_tag] = kmer_counts[locus_tag].replace(np.inf, 0)
        # Check for nan values in kmer counts and replace them with 0
        elif np.isnan(kmer_counts[locus_tag]).any():
            logger.warning(f"Gene {locus_tag} has nan values in kmer counts. Those look like:\n\
                {kmer_counts[locus_tag][kmer_counts[locus_tag].isna()]}")
            kmer_counts[locus_tag] = kmer_counts[locus_tag].replace(np.nan, 0)
        # Calculate typicality for the gene
        Typicality_dict[locus_tag] = np.cov(
            kmer_counts[locus_tag], kmer_counts["genome"])[0, 1]

    # Convert the dictionary to a DataFrame
    Typicality_df = pd.DataFrame.from_dict(
        Typicality_dict, orient="index", columns=["Typicality"])
    Typicality_df.index.name = "locus_tag"
    # Join the gene positions to the DataFrame
    Typicality_df = Typicality_df.join(gene_positions, on="locus_tag")

    # Sort the DataFrame by typicality
    Typicality_df = Typicality_df.sort_values("Typicality", ascending=True)
    # Rank the typicality values
    Typicality_df["Typicality_rank"] = Typicality_df["Typicality"].rank(
        ascending=True, method="average")

    # Calculate the derivative of the typicality ranks
    Typicality_df["Typicality_derivative"] = np.gradient(
        Typicality_df["Typicality"])

    # Smooth the derivative using Savitzky-Golay filter
    savgol_order = 2
    # determine the minimum savgol window len based on the number of genes
    savgol_window_len = max(int(Typicality_df.shape[0]//100), savgol_order+1)
    # make sure this is an odd number
    if savgol_window_len % 2 == 0:
        savgol_window_len += 1
    savgol_max_window_len = 2*savgol_window_len + 1
    min_rmse = np.inf
    savgol_window_lens = np.arange(savgol_window_len, savgol_max_window_len, 2)
    for savgol_window_len in savgol_window_lens:
        Typicality_df["Typicality_derivative_smoothed"] = savgol_filter(
            Typicality_df["Typicality_derivative"], savgol_window_len, savgol_order)
        # Calculate the root mean squared error of the smoothed derivative w.r.t the original derivative
        rmse = np.sqrt(np.mean((Typicality_df["Typicality_derivative"] -
                                Typicality_df["Typicality_derivative_smoothed"])**2))
        if rmse < min_rmse:
            min_rmse = rmse
            best_savgol_window_len = savgol_window_len
        else:
            break

    logger.info(f"Optimal Savitzky-Golay filter window length: {savgol_window_len} genes out of {Typicality_df.shape[0]} genes \
        for genome {genome_basename}. The RMSE is {min_rmse} and σ is {np.std(Typicality_df['Typicality_derivative_smoothed'])}.")

    # Apply the best Savitzky-Golay filter
    Typicality_df["Typicality_derivative_smoothed"] = savgol_filter(
        Typicality_df["Typicality_derivative"], best_savgol_window_len, savgol_order)

    # Calculate the double derivative
    Typicality_df["Typicality_double_derivative"] = np.gradient(
        Typicality_df["Typicality_derivative"])
    # Calculate the double derivative of the smoothed derivative
    Typicality_df["Typicality_smoothed_double_derivative"] = np.gradient(
        Typicality_df["Typicality_derivative_smoothed"])

    # Find the threshold on the derivative such that everything before it is atypical HGT genes
    # Typicality_df = find_threshold_at_double_derivative_root(Typicality_df, logger, genome_basename)
    Typicality_df = find_threshold(
        Typicality_df, logger, genome_basename, stringency, tmp_dir, debug)

    # index is locus_tag
    Typicality_df.set_index("locus_tag", inplace=True)

    # if debug is enabled, plot the derivatives, and save the plot
    if debug:
        logger.debug(f"Plotting derivatives for genome {genome_basename}")
        plot_derivatives(Typicality_df, tmp_dir, genome_basename)

    return Typicality_df


def calculate_genome_Typicality(genome_path, gene_positions, template_size, tmp_dir, threads, kmc_bin, debug, stringency):

    # extract the gene sequences from the genome
    gene_seqs_dict = extract_gene_sequences(gene_positions, genome_path)
    logger.info(f"Extracted {len(gene_seqs_dict)
                             } gene sequences from the genome {genome_path}")
    logger.debug(f"The first 5 entries of gene_positions look like this:\n{
                 gene_positions.head(5)}")

    genome_basename = os.path.basename(genome_path)

    # find the frequency of each n-oligonucleotide in the 'genome' using kmc
    kmc_command = f"{kmc_bin}/kmc -hp -k{template_size} -m2 -ci1 -cs1000000 -fm -hp -r {
        genome_path} {tmp_dir}/kmc_db_{genome_basename} {tmp_dir} > /dev/null"
    subprocess.run(kmc_command, shell=True, check=True,
                   capture_output=True, text=True)
    kmc_dump_command = f"{kmc_bin}/kmc_tools -hp transform {tmp_dir}/kmc_db_{
        genome_basename} dump {tmp_dir}/{genome_basename}.genome_kmer_counts.txt > /dev/null"
    subprocess.run(kmc_dump_command, shell=True, check=True,
                   capture_output=True, text=True)
    kmer_counts = pd.read_csv(f"{tmp_dir}/{genome_basename}.genome_kmer_counts.txt",
                              sep="\t", header=None, names=["kmer", "genome"]).set_index("kmer")
    kmer_counts["genome"] = kmer_counts["genome"].astype(int)

    # initialise one column for each gene in the gene_positions df to store their kmer counts
    zero_columns = pd.DataFrame(
        0, index=kmer_counts.index, columns=gene_positions.index)
    kmer_counts = pd.concat([kmer_counts, zero_columns], axis=1)
    logger.debug(f"Initialised kmer_counts df with {kmer_counts.shape[0]} kmers and {kmer_counts.shape[1]} columns\n\
        The first 5 rows and columns look like this:\n{kmer_counts.iloc[:5, :5]}")

    # find the frequency of each n-oligonucleotide in each of the genes via a multiprocessing pool
    # in the kmer_counts df, add one column for each gene with the frequency of each n-oligonucleotide in that gene. For kmers not present in a gene, the frequency should be 0
    with mp.Pool(threads) as pool:
        results = pool.starmap(calculate_gene_kmer_counts, [(
            locus_tag, gene_seq, genome_basename, template_size, tmp_dir, kmc_bin) for locus_tag, gene_seq in gene_seqs_dict.items()])
        # results is a tuple of locus_tag and gene kmer counts (kmc). kmc is a df with kmers as indices and counts as values
        # using the indices of kmer_counts df, add a column for each gene in the kmer_counts df mapped with the counts of each kmer in that gene
        for locus_tag, gene_kmc in results:
            # if gene_kmc is empty, raise an error and log the locus_tag and gene_kmc
            if gene_kmc.empty:
                raise ValueError(
                    f"Gene {locus_tag} has no kmer counts. gene_kmc: {gene_kmc}")
            else:
                # logger.debug(f"Gene {locus_tag} has {gene_kmc.shape[0]} kmers and counts look like this:\n{gene_kmc["count"].head()}")
                # add the gene_kmc to the kmer_counts df, and for indices not present in gene_kmc, fill with 0
                kmer_counts[locus_tag] = gene_kmc[locus_tag].reindex(
                    kmer_counts.index, fill_value=0)

    kmer_counts.fillna(0, inplace=True)
    kmer_counts = kmer_counts.loc[:, (kmer_counts != 0).any(axis=0)]

    # show the first 5 rows and first 5 columns of the kmer_counts df
    logger.debug(f"First 5 rows and first 5 columns of the kmer_counts df look like this:\n{
                 kmer_counts.iloc[:5, :5]}")

    # calculate the typicality for each gene
    Typicality_df = identify_hgt(gene_seqs_dict, kmer_counts, gene_positions,
                                 logger, debug, tmp_dir, genome_basename, stringency)
    logger.debug(f"Calculated Typicality for {
                 Typicality_df.shape[0]} genes. The first 5 rows look like this:\n{Typicality_df.head()}")

    return Typicality_df


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-positions", "-g", type=str, required=False, default="/root/work/projects/hgt_inference_comparative_study/data/1236_gene_features.tsv",
                        help="(str) Path to gene positions TSV file, with column headers: locus_tag, gene_id, start, end, strand, seqid, genome_accession, taxon_id. Default: /root/work/projects/hgt_inference_comparative_study/data/1236_gene_features.tsv")
    parser.add_argument("--genome-paths", "-p", type=str, required=False, default="/root/work/projects/hgt_inference_comparative_study/data/1236_genome_fna_filepaths.tsv",
                        help="(str) Path to genome paths TSV file, with columns: genome_accession, genome_path. Default: /root/work/projects/hgt_inference_comparative_study/data/1236_genome_fna_filepaths.tsv")
    parser.add_argument("--output-dir", "-o", type=str, required=False, default="Results/",
                        help="(str) Path to output directory. Default: Results/")
    parser.add_argument("--threads", "-t", type=int, default=100,
                        help="(int) Number of threads to use for parallelization. Default: 100")
    parser.add_argument("--template-size", "-s", type=int, default=8,
                        help="(int) Size of the template, i.e. n in n-oligonucleotide. Default: 8")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug mode. This will test only the first 3 genomes. \
                            It will also not delete the temporary directory which may contain intermediate files.\
                            Note that gene and genome kmer count files will still be concatenated and original files will be deleted.")
    parser.add_argument("--kmc_bin", "-k", type=str, default="/root/largeFilesThatShouldNotBeSynced/bin/KMC/bin/",
                        help="(str) Path to the directory containing the kmc binary files: kmc and kmc_tools. Default: /root/largeFilesThatShouldNotBeSynced/bin/KMC/bin/")
    parser.add_argument("--stringency_min_max", "-m", type=str, default="4,13",
                        help="(str) The minimum and maximum stringency values for the threshold. Default: 4,13.\
                            The threshold is calculated as the stringency times the median absolute deviation of the double derivative of the smoothed typicality values.\
                            The minimum stringency value is recommended to be at least 2.5.")
    parser.add_argument("--stringency_steps", "-ss", type=int, default=10,
                        help="(int) The number of steps between the minimum and maximum stringency values, both inclusive. Default: 10")

    args = parser.parse_args()
    # parse args
    gene_positions = args.gene_positions
    genome_paths = args.genome_paths
    output_dir = args.output_dir
    threads = args.threads
    template_size = args.template_size
    debug = args.debug
    kmc_bin = args.kmc_bin

    stringency_min, stringency_max = map(
        float, args.stringency_min_max.split(","))
    stringency = np.linspace(
        stringency_min, stringency_max, args.stringency_steps)

    # Get the current date and time
    start_time = datetime.now()
    # Format the current date and time as a string
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    # Create a temporary directory path with the timestamp
    tmp_dir = f"tmp_Wn_{timestamp}"
    # Create the temporary directory and output directory
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    # set up logger, also based on debug mode or not
    logger.remove()
    if debug:
        logger_level = "DEBUG"
        # add a stream handler to print logs to stdout
        logger.add(sys.stdout, level=logger_level)
    else:
        logger_level = "INFO"
    logger.add(f"tmp_Wn_{timestamp}.log", level=logger_level)

    # log this with time, along with location of the tmp folder
    logger.info(f'Created the tmp folder at location: {tmp_dir}')

    # log arguments as key value pairs
    logger.info(f"Arguments: {"\n".join(
        [f"{k}: {v}" for k, v in vars(args).items()])}")

    # read the gene_positions file
    gene_positions = pd.read_csv(
        gene_positions, sep="\t", header=0, index_col="locus_tag")
    # for each of the genomes, calculate Typicality for each gene and write them to a file
    genome_paths = pd.read_csv(genome_paths, sep="\t", header=0, names=[
                               "genome_accession", "genome_path"])
    # keep only the genomes for which gene_positions are available
    genome_paths = genome_paths[genome_paths["genome_accession"].isin(
        gene_positions["genome_accession"].unique())]
    if args.debug:
        genome_paths = genome_paths.head(3)

    Typicality_dfs = []
    for genome_acc, genome_path in genome_paths.itertuples(index=False):
        Typicality_df = calculate_genome_Typicality(genome_path, gene_positions[gene_positions["genome_accession"] == genome_acc],
                                                    template_size, tmp_dir, threads,
                                                    kmc_bin, debug, stringency)
        logger.debug(f"Calculated Typicality for genome {genome_acc} and found {Typicality_df.shape[0]} genes\
            with Typicality values. The first 5 rows look like this:\n{Typicality_df.head()}")
        Typicality_dfs.append(Typicality_df)
    Typicality_df = pd.concat(Typicality_dfs)
    Typicality_df.to_csv(f"{output_dir}/Wn.tsv",
                         sep="\t", header=True, index=True)

    # make a new df that contains only rows where at least one of the HGT columns is True
    hgt_cols = [
        col for col in Typicality_df.columns if "HGT (stringency level=" in col]
    hgt_df = Typicality_df[Typicality_df[hgt_cols].any(axis=1)]
    # write each of the HGT (stringency={s}) columns together a separate file
    hgt_df.to_csv(f"{output_dir}/HGT_genes.tsv",
                  sep="\t", header=True, index=True)
    logger.info(f"Written the HGT genes to {output_dir}/HGT_genes.tsv")

    # log the number of HGT genes at each threshold
    hgt_counts = hgt_df[hgt_cols].sum()
    logger.info(f"Number of HGT genes at each threshold:\n{hgt_counts}")

    # remove the kmc binary files and .fna files from the temporary directory
    # Remove the kmc binary files and .fna files from the temporary directory
    kmc_rm_command = f"find {tmp_dir} -type f -name 'kmc_db*' -delete"
    subprocess.run(kmc_rm_command, shell=True, check=True)
    fna_rm_command = f"find {tmp_dir} -type f -name '*.fna' -delete"
    subprocess.run(fna_rm_command, shell=True, check=True)
    logger.info(
        f"Removed the kmc binary files and .fna files from the temporary directory {tmp_dir}")

    # combine all the gene kmer counts files into one file, with an additional column for locus_tag
    # gene_kmer_counts_files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".gene_kmer_counts.txt")]
    # gene_kmer_counts = pd.concat([pd.read_csv(f, sep="\t", header=None,
    #                                           names=["kmer", "count"]).assign(locus_tag=f.split("/")[-1].split("genomic.fna_")[1].split(".")[0],
    #                                                                           taxon_id=f.split("/")[-1].split("_")[0]) for f in gene_kmer_counts_files])

    # gene_kmer_counts.to_csv(f"{output_dir}/gene_kmer_counts.tsv", sep="\t", header=True, index=False)
    # logger.info(f"Combined all the gene kmer counts files into one file: {output_dir}/gene_kmer_counts.tsv")
    # # similar for genome kmer counts files
    # genome_kmer_counts_files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".genome_kmer_counts.txt")]
    # genome_kmer_counts = pd.concat([pd.read_csv(f, sep="\t", header=None,
    #                                             names=["kmer", "count"]).assign(taxon_id=f.split("/")[-1].split("_")[0]) for f in genome_kmer_counts_files])
    # genome_kmer_counts.to_csv(f"{output_dir}/genome_kmer_counts.tsv", sep="\t", header=True, index=False)
    # logger.info(f"Combined all the genome kmer counts files into one file: {output_dir}/genome_kmer_counts.tsv")
    # remove the gene kmer counts files and genome kmer counts files from the temporary directory using shutil
    gene_genome_rm_command = f"find {
        tmp_dir} -type f -name '*.gene_kmer_counts.txt' -delete"
    subprocess.run(gene_genome_rm_command, shell=True, check=True)
    logger.info(
        f"Removed the gene kmer counts files from the temporary directory {tmp_dir}")
    genome_genome_rm_command = f"find {
        tmp_dir} -type f -name '*.genome_kmer_counts.txt' -delete"
    subprocess.run(genome_genome_rm_command, shell=True, check=True)
    logger.info(
        f"Removed the genome kmer counts files from the temporary directory {tmp_dir}")

    # remove the temporary directory
    if not args.debug:
        try:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
                logger.info(f"Removed the temporary directory {tmp_dir}")
        except Exception as e:
            logger.error(f"Error removing directory {tmp_dir}: {e}")
    else:
        logger.info(f"Debug mode enabled. Temporary directory {
                    tmp_dir} not removed.")

    logger.info(f"Finished running Wn for {len(
        genome_paths)} genomes and {
            Typicality_df.shape[0]} genes. \nThe results are written to {
                output_dir}/Wn.tsv \nTotal runtime: {datetime.now() - start_time}")
