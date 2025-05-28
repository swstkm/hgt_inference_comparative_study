# Run AnGST with multiprocessing. This script requires python2 to run AnGST.
import os
import subprocess
from multiprocessing import Pool
import time
import logging
import shutil
from datetime import timedelta
from ete3 import Tree
import traceback

# Create or get the logger
logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def run_ANGST_on_NOG(pool_args):
    """
    This function is used to run AnGST in a multithreaded procedure.
    For each thread/worker process, the function is called with a different NOG_ID.
    Args:
        pool_args (tuple): Tuple containing the following elements:
            nog_id (str): NOG ID to run AnGST on
            species_tree (str): Path to species tree file
            gene_tree_filepath (str): Path to gene tree file
            output_dir (str): Path to output directory
            penalty_file (str): Path to penalty file
    Returns:
        nog_id (str): NOG ID that was run, if successful
        If unsuccessful, returns None
    """

    nog_id, species_tree, gene_tree_filepath, output_dir, penalty_file, bin_dir, tmp_dir = pool_args

    # create input file for AnGST in tmp folder
    input_path = os.path.join(tmp_dir, "{}.input".format(nog_id))
    angst_script = "{}/angst_lib/AnGST.py".format(bin_dir)
    # output directory for AnGST results for this NOG
    output_dir = os.path.join(output_dir, nog_id)

    input_content = [
        "species={}\n".format(species_tree),
        "gene={}\n".format(gene_tree_filepath),
        "output={}/\n".format(output_dir),
        "penalties={}".format(penalty_file)
    ]

    # write input file
    with open(input_path, "w") as infile:
        infile.writelines(input_content)

    # Run AnGST
    try:
        logger.info("Running AnGST for NOG_ID {}".format(nog_id))
        logger.info("Command: python2 {} {}".format(
            angst_script, input_path))
        # make sure python2 is available
        py2_available = False
        for path in os.environ["PATH"].split(os.pathsep):
            full_path = os.path.join(path, "python2")
            if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                py2_available = True
                break
        if not py2_available:
            logger.error("python2 is not available in PATH")
            raise EnvironmentError("python2 is not available in PATH. Make sure you are in the correct conda environment.")
                
        subprocess.check_call(["python2", angst_script, input_path])
    except subprocess.CalledProcessError as e:
        logger.error(
            "AnGST failed for NOG_ID {} with error\n {}".format(nog_id, e))
        return None
    except Exception as e:
        logger.error(
            "AnGST failed for NOG_ID {} with error\n {}".format(nog_id, e))
        logger.error(traceback.format_exc())
        return None

    return nog_id


def unroot(newick_string, outfile_path):
    """
    This function takes a newick string and writes it to a file after unrooting it.
    Args:
        newick_string (str): Newick string to be written to file
        outfile_path (str): Path to output file
    Returns:
        None
    """
    # read in the newick string
    t = Tree(newick_string, format=1)
    # unroot the tree
    t.unroot()
    # write the tree to file
    t.write(outfile=outfile_path, format=1)
    logger.info("Unrooted gene tree and wrote it to {}".format(outfile_path))
    return None


if __name__ == '__main__':
    import argparse
    # track time
    start_time = time.time()
    logger.info("Started running {} at {}".format(__file__, time.asctime()))

    # use argparse for reading in the number of threads and the input trees file
    parser = argparse.ArgumentParser(
        description="Run AnGST on a set of gene trees")
    parser.add_argument("--species", "-s", type=str, default="../../1236_wol_tree_pruned_angst.nwk",
                        help="Path to species tree file (default: ../../1236_wol_tree_pruned_angst.nwk)")
    parser.add_argument("--gene", "-g", type=str, default="../../1236_pruned_gene_trees.tsv",
                        help="Path to gene trees file (default: ../../1236_pruned_gene_trees.tsv)")
    parser.add_argument("--threads", "-t", type=int, default=50,
                        help="Number of threads to use for parallelization (default: 50)")
    parser.add_argument("--output", "-o", type=str, default="./Results",
                        help="Output directory for AnGST results (default: ./Results)")
    parser.add_argument("--penalties", "-p", type=str, default="../src/angst_penalty_file.txt",
                        help="Path to penalty file for AnGST (default: ../src/angst_penalty_file.txt)")
    parser.add_argument("--bin", "-b", type=str, default="/root/bin/angst/",
                        help="Path to AnGST bin directory (default: /root/bin/angst/)")

    # parse args
    args = parser.parse_args()
    species_tree = args.species
    gene_trees = args.gene
    max_threads = args.threads
    output_dir = args.output
    penalty_file = args.penalties
    bin_dir = args.bin
    # log arguments
    logger.info("Running {} with arguments: {}".format(__file__, args))

    # convert all the paths to absolute paths
    species_tree = os.path.abspath(species_tree)
    gene_trees = os.path.abspath(gene_trees)
    output_dir = os.path.abspath(output_dir)
    penalty_file = os.path.abspath(penalty_file)
    bin_dir = os.path.abspath(bin_dir)

    # create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("Created output directory {}".format(output_dir))
    else:
        logger.info("Output directory {} already exists".format(output_dir))
        # delete the output dir and recreate it
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    # create tmp directory for gene trees and input files, with a timestamp, with abs path
    timestamp = time.strftime("%Y%m%d%H%M%S")
    tmp_dir = os.path.join(os.getcwd(), "tmp_{}".format(timestamp))
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    logger.info("Created temporary directory {}".format(tmp_dir))

    # Read in the gene trees and create a list of eggNOG IDs
    # additionally, write out the species tree and each gene tree to a file as first and second lines,
    # with the ID of gene tree as the filename, in the tmp folder
    if not os.path.exists(gene_trees):
        logger.error("Gene trees file {} does not exist".format(gene_trees))
        raise IOError("Gene trees file {} does not exist".format(gene_trees))
    else:
        logger.info("Reading gene trees from {}".format(gene_trees))
        with open(gene_trees, "r") as gene_trees_fo:
            gene_trees = gene_trees_fo.readlines()
            nog_id_list = [tree.split("\t")[0] for tree in gene_trees]

    # Debugging: run AnGST on last n_debug NOGs only
    # n_debug=2; nog_id_list = nog_id_list[-n_debug:]; max_threads=n_debug; gene_trees=gene_trees[-n_debug:]

    # unroot the gene trees and write them to file
    for gene_tree in gene_trees:
        nog_id, nog_newick = gene_tree.split("\t")
        # first read in the tree using ete3 and then write it out after unrooting it
        try:
            unroot(nog_newick, "{}/{}.nwk".format(tmp_dir, nog_id))
        except Exception as e:
            logger.error(
                "Error unrooting gene tree for NOG_ID {}\n{}".format(nog_id, e))
            continue

    # prepare list of tuples to be fed into each process for parallelization
    pool_args = [(nog_id, species_tree, "{}/{}.nwk".format(tmp_dir, nog_id),
                  output_dir, penalty_file, bin_dir, tmp_dir) for nog_id in nog_id_list]

    # create the pool. Note that this is python2, so we can't use `with Pool() as pool:`
    pool = Pool(processes=max_threads)

    # use the pool
    for pool_out in pool.imap(run_ANGST_on_NOG, pool_args):
        if pool_out is not None:
            logger.info(
                "AnGST run for {} completed successfully".format(pool_out))

    # close and join the pool
    pool.close()
    pool.join()

    # Optional: remove the temporary directory
    # shutil.rmtree(tmp_dir); logger.info("Removed temporary directory {}".format(tmp_dir))

    # track time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("Finished running {} at {}, took {}".format(
        __file__, time.asctime(), str(timedelta(seconds=elapsed_time))))
