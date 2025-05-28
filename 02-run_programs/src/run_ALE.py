import os
import subprocess
from multiprocessing import Pool
from loguru import logger
import glob
import shutil
import time


def run_ALE_on_NOG(NOG_ID, species_tree_filepath, gene_trees_filepath, bin_dir, n_samples, separator, max_retries):
    """Run ALE on a single NOG_ID
    Args:
        NOG_ID (str): NOG ID to run ALE on
        species_tree_filepath (str): Path to the species tree file
        gene_trees_filepath (str): Path to the gene trees file
        bin_dir (str): Path to the ALE bin directory
        n_samples (int): Number of reconciliation samples for ALEml_undated
        separator (str): Separator for ALEml_undated

    Returns:
        NOG_ID (str): NOG ID that was run, if successful
        If unsuccessful, returns None
    """
    base_dir = os.getcwd()  # this is the directory where the script is run from

    genetree_filepath = os.path.join(base_dir, f"{NOG_ID}.tree")
    # write this specific genetree to a file by grepping for the NOG_ID
    # and then writing the second column to a file
    try:
        with open(genetree_filepath, 'w') as tree_file_object:
            p1 = subprocess.Popen(
                ["grep", NOG_ID, gene_trees_filepath], stdout=subprocess.PIPE)
            p2 = subprocess.Popen(
                ["awk", "{print $2}"], stdin=p1.stdout, stdout=tree_file_object)
            p2.communicate()
            if p1.poll() is None:
                p1.communicate()
        logger.info(f"Successfully wrote gene tree {
                    NOG_ID} to file {genetree_filepath}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to write gene tree to file with error {e}")
        return None
    # run ALE on the genetree
    # first run ALEobserve to create the .ale file
    n_sleep = 2  # sleep for n_sleep seconds before retrying
    for i in range(max_retries + 1):
        try:
            subprocess.run(
                [f"{bin_dir}ALEobserve", genetree_filepath], check=True)
            break
        except subprocess.CalledProcessError as e:
            if i < max_retries:  # i is zero indexed
                logger.warning(f"ALEobserve failed with error {
                               e}, retrying ({i+1}/{max_retries})")
                # sleep for n_sleep seconds before retrying
                time.sleep(n_sleep)
                continue
            else:
                logger.error(f"ALEobserve failed with error {
                             e}, no more retries left")
                return None

    # then run ALEml_undated with retry mechanism
    for i in range(max_retries + 1):
        try:
            aleml_undated_command = [f"{bin_dir}ALEml_undated", species_tree_filepath, f"{
                genetree_filepath}.ale", f'separators="{separator}"', f"sample={n_samples}"]
            logger.info(f"Running ALEml_undated with command: {
                        aleml_undated_command}")
            subprocess.run(aleml_undated_command, check=True)
            return NOG_ID  # If ALEml_undated succeeds, return the NOG_ID
        except subprocess.CalledProcessError as e:
            if i < max_retries:  # i is zero indexed
                logger.warning(f"ALEml_undated failed with error {
                               e}, retrying ({i+1}/{max_retries})")
                time.sleep(n_sleep)
                continue
            else:
                logger.error(f"ALEml_undated failed with error {
                             e}, no more retries left")
                return None

    return NOG_ID


def run_ALE_on_NOG_wrapper(args):
    return run_ALE_on_NOG(*args)


if __name__ == '__main__':
    # calculate total time taken
    from datetime import timedelta
    import time
    start_time = time.time()
    # log using start time
    logger.info("Starting ALE run")

    # write everything in terms of argparse instead of hardcoding
    import argparse
    parser = argparse.ArgumentParser(description="Run ALE on all gene trees")
    parser.add_argument("--species", "-s", type=str, default="../../1236_wol_tree_pruned_no_internal_labels.nwk",
                        help="Path to species tree file (default: ../../1236_wol_tree_pruned_no_internal_labels.nwk)")
    parser.add_argument("--gene", "-g", type=str, default="../../1236_pruned_gene_trees.tsv.rooted",
                        help="Path to gene trees file (default: ../../1236_pruned_gene_trees.tsv.rooted)")
    parser.add_argument("--threads", "-t", type=int, default=50,
                        help="Number of threads to use for parallelization (default: 50)")
    parser.add_argument("--bin", "-b", type=str,
                        default="/root/bin/ALE/bin/", help="Path to ALE bin directory (default: /root/bin/ALE/bin/)")
    parser.add_argument("--samples", "-n", type=int, default=100,
                        help="Number of reconciliation samples for ALEml_undated (default: 100)")
    parser.add_argument("--separator", "-sep", type=str, default=".",
                        help="Separator for ALEml_undated (default: '.')")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug mode (run ALE on first 5 input trees only)")
    parser.add_argument("--output-dir", "-o", type=str, default="./Results/",
                        help="Path to output directory (default: ./Results/)")
    parser.add_argument("--retries", "-r", type=int, default=10,
                        help="Max number of retries for ALEml_undated and ALEobserve (default: 10)")
    args = parser.parse_args()

    # parse the arguments
    species_tree_filepath = args.species
    gene_trees_filepath = args.gene
    max_threads = args.threads
    bin_dir = args.bin
    n_samples = args.samples
    separator = args.separator
    output_dir = args.output_dir
    max_retries = args.retries

    # log the arguments
    logger.info(f"\nArguments passed: \
                \nSpecies tree file: {species_tree_filepath} \
                \nGene trees file: {gene_trees_filepath} \
                \nNumber of threads: {max_threads} \
                \nALE bin directory: {bin_dir} \
                \nNumber of samples: {n_samples} \
                \nSeparator: {separator} \
                \nOutput directory: {output_dir} \
                \nNumber of retries: {max_retries}")

    # read and extract first column as list of NOG_IDs
    with open(gene_trees_filepath, "r") as treeIDs:
        NOG_ID_list = [line.rstrip().split("\t")[0]
                       for line in treeIDs.readlines()]

    # if debug
    if args.debug:
        NOG_ID_list = NOG_ID_list[:5]
        logger.warning(
            "Debug mode enabled. Running ALE on first 5 gene trees only")
        max_threads = 5

    # log the number of NOGs to run ALE on, with time
    logger.info(f"ALE will be run on {(len(NOG_ID_list))} NOGs, using {
                str(max_threads)} threads")

    with Pool(processes=max_threads) as pool:  # run ALE on each NOG in parallel
        for pool_out in pool.imap(run_ALE_on_NOG_wrapper, [(NOG_ID, species_tree_filepath, gene_trees_filepath, bin_dir, n_samples, separator, max_retries) for NOG_ID in NOG_ID_list]):
            if pool_out is not None:                                # if ALE run was successful
                # log time when you finished this NOG_ID
                print(f"{time.strftime('%X %x %Z')}: Finished {pool_out}\n")

    # move all files that end with `ale.uTs` to the outputs directory
    # check if the outputs directory exists, if not create it. If creating new, raise warning
    if os.path.exists(output_dir):
        logger.warning(
            "The outputs directory already exists. Overwriting files in the outputs directory")
        # remove all files in the outputs directory
        for file in glob.glob(f"./{output_dir}/*"):
            os.remove(file)
    else:
        os.makedirs(output_dir)
    # remove all `*.ale.uml_rec` files
    # for file in glob.glob(f"./*.ale.uml_rec"):
        # os.remove(file)
    # or move them to the outputs directory
    for file in glob.glob(f"./*.ale.uml_rec"):
        shutil.move(file, output_dir)
    # get the filename of the species tree
    species_tree_filename = os.path.basename(species_tree_filepath)
    for file in glob.glob(f"./*.ale.uTs"):
        shutil.move(file, output_dir)
    # remove all files that end with .ale
    for file in glob.glob(f"./*.ale"):
        os.remove(file)
    # move all files that end with .tree to the trees directory
    # create the directory if it doesn't exist
    os.makedirs("./tmp_trees/", exist_ok=True)
    for file in glob.glob(f"./*.tree"):
        shutil.move(file, "./tmp_trees/")

    # calculate total time taken
    end_time = time.time()
    total_time = end_time - start_time
    # log the time taken in a readable format
    logger.info(f"Total time taken: {timedelta(seconds=total_time)}")
