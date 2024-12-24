import subprocess
import os
from multiprocessing import Pool
from loguru import logger
import time
import shutil
from datetime import timedelta


def run_RANGERDTL_on_NOG(NOG_ID, tmp_dir, bin_dir, n_runs, ranger_bin):
    """
    This function is used to run RangerDTL in a multithreaded procedure. 
    For each thread/worker process, the function is called with a different NOG_ID.
    Args:
        NOG_ID (str): NOG ID to run RangerDTL on
        tmp_dir (str): path to temporary directory to store input files
        bin_dir (str): path to Ranger bin directory
        n_runs (int): number of runs to perform
    Returns:
        NOG_ID (str): NOG ID that was run, if successful
        If unsuccessful, returns None

    The output of RangerDTL's AggregateRanger is a file with the same name as the input file, but with the ending "_aggregateoutput.txt". 
    This file contains the reconciliation results of n_runs runs of RangerDTL.

    All the results are stored in the folder "Ranger/Results". The input files are stored in "Ranger/tmp_{NOG_ID}_results". 

    The input files are (optionally) deleted after the program has been run. 
    """

    # Make a folder for the results of each tree
    os.makedirs(f"./Results/tmp_{NOG_ID}_results", exist_ok=True)
    # use the NOG_ID to define the filename of the input file
    input_file = f"{tmp_dir}/{NOG_ID}.input"
    # Run Ranger to perform n_runs reconciliations
    for i in range(1, n_runs+1):
        # Run RangerDTL. Output files need to have prefix "recon{index}", with index starting at 1.
        try:
            subprocess.run(f"{bin_dir}/{ranger_bin} -i {input_file} -o  ./Results/tmp_{
                NOG_ID}_results/recon{i} >> ./Results/tmp_{NOG_ID}_results/RANGERDTL.log", shell=True)
            # subprocess.run(f"{bin_dir}/SupplementaryPrograms/Ranger-DTL-Fast.linux -i {input_file} -o  ./Results/tmp_{
            # NOG_ID}_results/recon{i} >> ./Results/tmp_{NOG_ID}_results/RANGERDTL.log", shell=True)
            # subprocess.run(f"{bin_dir}/CorePrograms/Ranger-DTL.linux -i {input_file} -o  ./Results/tmp_{
            # NOG_ID}_results/recon{i} >> ./Results/tmp_{NOG_ID}_results/RANGERDTL.log", shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"RangerDTL failed for NOG_ID {
                         NOG_ID} with error\n {e}")
            return None
    logger.info(f"Running RangerDTL finished for NOG_ID {NOG_ID}")
    # Aggregate the results of the runs into one file.
    try:
        subprocess.run(f"{bin_dir}/CorePrograms/AggregateRanger.linux ./Results/tmp_{
                       NOG_ID}_results/recon > ./Results/{NOG_ID}_aggregateoutput.txt", shell=True)
        logger.info(f"Aggregating results finished for NOG_ID {NOG_ID}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Aggregating RangerDTL results failed for NOG_ID {
                     NOG_ID} with error\n {e}")
        return None

    # Optional: Delete the input file because it is not needed anymore
    if os.path.exists(input_file):
        os.remove(input_file)
        logger.info(f"Deleted input file {input_file}")
    # Optional: remove tmp_{NOG_ID}_results folder
    # shutil.rmtree(f"./Results/tmp_{NOG_ID}_results")
    return NOG_ID


def run_RANGERDTL_on_NOG_wrapper(args):
    """
    Wrapper function for running RangerDTL on a single NOG_ID
    Args:
        args (tuple): Tuple containing (NOG_ID, species_tree_filepath, gene_trees_filepath, bin_dir)
    Returns:
        NOG_ID (str): NOG ID that was run, if successful
        If unsuccessful, returns None
    """
    return run_RANGERDTL_on_NOG(*args)


if __name__ == '__main__':
    # log time
    logger.info(f"Started running {__file__} at {time.asctime()}")

    start_time = time.time()

    # import argparse and read in the arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--speciestree", "-s", help="Path to the species tree (default: ../../data/1236_wol_tree_pruned_with_internal_labels.nwk)",
                        type=str, default="../../data/1236_wol_tree_pruned_with_internal_labels.nwk")
    parser.add_argument("--genetrees", "-g", help="Path to the gene trees (default: ../../data/1236_pruned_gene_trees.tsv.rooted.underscored)", type=str,
                        default="../../data/1236_pruned_gene_trees.tsv.rooted.underscored")
    parser.add_argument("--threads", "-t",
                        help="Number of threads to use (default: 50)", type=int, default=50)
    parser.add_argument("--bin", "-b", help="Path to the Ranger bin directory (default: /root/bin/RANGERDTL_LINUX/)",
                        type=str, default="/root/bin/RANGERDTL_LINUX/")
    parser.add_argument(
        "--runs", "-r", help="Number of runs to perform (default: 100)", type=int, default=100)
    parser.add_argument(
        "--fast", "-f", help="Use RANGER-DTL-Fast instead of RANGER-DTL (default: False)", default=False, action="store_true")
    args = parser.parse_args()
    # log arguments
    logger.info(f"Running {__file__} with arguments: {args}")
    # Read in the arguments
    species_tree_filepath = args.speciestree
    gene_trees_filepath = args.genetrees
    max_threads = args.threads
    bin_dir = args.bin
    n_runs = args.runs
    if args.fast:
        ranger_bin = "SupplementaryPrograms/Ranger-DTL-Fast.linux"
    else:
        ranger_bin = "CorePrograms/Ranger-DTL.linux"

    # create temporary dir for the input files to RANGER, using tempfile with date-time stamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    tmp_dir = f"./tmp_ranger_{timestamp}"
    os.makedirs(tmp_dir, exist_ok=True)

    # Read in the newick string of the species tree
    if not os.path.exists(species_tree_filepath):
        logger.error(f"Species tree file {
                     species_tree_filepath} does not exist")
        raise FileNotFoundError(f"Species tree file {
                                species_tree_filepath} does not exist")
    else:
        logger.info(f"Reading species tree from {species_tree_filepath}")
        with open(species_tree_filepath, "r") as species_tree:
            species_tree_newick = species_tree.read()
            # make sure the newick string ends with a newline character. If not, add it.
            if not species_tree_newick.endswith("\n"):
                species_tree_newick += "\n"

    # Read in the gene trees and create a list of eggNOG IDs
    # additionally, write out the species tree and each gene tree to a file as first and second lines,
    # with the ID of gene tree as the filename, in the tmp folder
    if not os.path.exists(gene_trees_filepath):
        logger.error(f"Gene trees file {gene_trees_filepath} does not exist")
        raise FileNotFoundError(
            f"Gene trees file {gene_trees_filepath} does not exist")
    else:
        logger.info(f"Reading gene trees from {gene_trees_filepath}")
        with open(gene_trees_filepath, "r") as gene_trees_fo:
            gene_trees = gene_trees_fo.readlines()
            NOG_ID_list = [tree.split("\t")[0] for tree in gene_trees]
            for gene_tree in gene_trees:
                nog_id, nog_newick = gene_tree.split("\t")
                # add newline character to newick string if it doesn't end with it
                if not nog_newick.endswith("\n"):
                    nog_newick += "\n"
                with open(f"{tmp_dir}/{nog_id}.input", "w") as gene_tree_fo:
                    gene_tree_fo.write(species_tree_newick)
                    gene_tree_fo.write(nog_newick)

    # Debugging: Run RangerDTL on last 10 NOGs only
    # NOG_ID_list = NOG_ID_list[-2:]; max_threads=2;

    logger.info(f"Running RangerDTL with {max_threads} threads on {
                len(NOG_ID_list)} NOGs, performing {n_runs} runs for each NOG.")
    with Pool(processes=max_threads) as pool:
        # iterate through what has been created via pool.imap()
        for pool_out in pool.imap(run_RANGERDTL_on_NOG_wrapper, [(NOG_ID, tmp_dir, bin_dir, n_runs, ranger_bin) for NOG_ID in NOG_ID_list]):
            if pool_out is not None:
                logger.info(f"RangerDTL finished for {pool_out}")

    # log time taken
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Finished running {__file__} at {time.asctime()}, took {
                str(timedelta(seconds=elapsed_time))}")
