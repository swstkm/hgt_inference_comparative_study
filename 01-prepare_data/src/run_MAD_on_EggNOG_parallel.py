#!/usr/bin/env python3

from subprocess import CalledProcessError, Popen
import os
import time
from datetime import timedelta, datetime
import argparse
import glob
from multiprocessing import Pool


def run_command(params):
    """
    Inputs:
        params: tuple of (shell_command, logFileObject)
    Function:
        Runs the shell command, and writes the output to the log file object
    """
    cmd, log_file_path = params
    with open(log_file_path, 'w') as log_FO:
        process = Popen(cmd, stdout=log_FO, universal_newlines=True)
        if process.wait() != 0:
            raise CalledProcessError(process.returncode, cmd)


def MAD_roots_NOG_trees_parallel(input_file_path, mad_executable, n_lines, num_subprocesses):
    """
    Inputs:
        input_file_path: TSV format input file. Columns are: treeID(str) and tree(str)
        mad_executable: path to MAD executable
        n_lines: number of lines that the input file will be split into for each split file. 
    Outputs:
        Writes out a file with the same name as the input file, but with a `.rooted` at the end of the filename.
    Additional info:
        This function splits the input file into multiple files, and runs MAD on each of these split files in parallel.
        The number of processes that will be run in parallel is equal to the number of split files.
        The number of lines in each split file is equal to the `n_lines` argument.
        The output file will contain the same number of lines as the input file, but the second column will contain the rooted trees.
    """
    # Check if the input file exists
    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"The input file {
                                input_file_path} does not seem to exist. Check the filepath provided.")
    # Find the full path of the input file and extract the dir name
    input_file_path = os.path.abspath(input_file_path)
    input_dir = os.path.dirname(input_file_path)

    # Get the current date and time
    now = datetime.now()
    # Format the current date and time as a string
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # Create a temporary directory path with the timestamp
    tmp_dir = f"{input_dir}/tmp_{timestamp}"
    # Create the temporary directory
    os.mkdir(tmp_dir)
    # log this with time, along with location of the tmp folder
    print(f'Created the tmp folder at: {
          datetime.now()} at location: {tmp_dir}')

    # Open the input file and read it in chunks of `n_lines` lines
    # track the filenames of split files containing the second column (tree)
    nwk_split_files = []
    # track the filenames of split files containing the first column (treeID)
    info_split_files = []

    def write_split_files(chunk, chunk_index, nwk_split_files, info_split_files):
        """
        Helper function to write out the split files for a given chunk.
        """
        split_chunk = [line.split() for line in chunk]
        nwk_split_filepath = f"{tmp_dir}/tmp_tree_split.nwk.{chunk_index}"
        info_split_filepath = f"{tmp_dir}/tmp_tree_split.info.{chunk_index}"
        with open(nwk_split_filepath, 'w') as split_out_fo:
            split_out_fo.write('\n'.join(line[1]
                               for line in split_chunk) + '\n')
        with open(info_split_filepath, 'w') as split_out_fo:
            split_out_fo.write('\n'.join(line[0]
                               for line in split_chunk) + '\n')
        nwk_split_files.append(nwk_split_filepath)
        info_split_files.append(info_split_filepath)
        return nwk_split_files, info_split_files

    # open the input file
    with open(input_file_path, 'r') as treesFO:
        chunk = []
        for i, line in enumerate(treesFO):
            chunk.append(line)
            if (i + 1) % n_lines == 0:
                nwk_split_files, info_split_files = write_split_files(
                    chunk, (i // n_lines) + 1, nwk_split_files, info_split_files)
                chunk = []

        # Write the last chunk if it's not empty
        if chunk:
            nwk_split_files, info_split_files = write_split_files(
                chunk, (len(nwk_split_files) // n_lines) + 2, nwk_split_files, info_split_files)

    # using subprocess.Popen inside the run_command function,
    # call MAD on all of these split newick files. Prepare the commands.
    # `mad -n trees.nwk.1`
    commands = [([mad_executable, '-t', '-n', split_file_path], split_file_path+'.stdout')
                for split_index, split_file_path in enumerate(nwk_split_files)]

    mad_start_time = time.time()
    # Log timestamp for the start of the MAD rooting process
    print(f'MAD rooting starting at: {datetime.now()}')
    # Create a multiprocessing Pool with the defined maximum number of subprocesses
    with Pool(num_subprocesses) as pool:
        # Use pool.map to run the commands in parallel
        pool.map(run_command, commands)
        # Indicate the completion of the MAD rooting process
        print(f'MAD rooting completed at: {datetime.now()}. \
              Total time taken for MAD rooting: {timedelta(seconds=time.time() - mad_start_time)}')
    # Log that the split files are being combined, along with the timestamp
    print(f'{datetime.now()}: Combining the split files into one file...')

    # Note: mad adds a `.rooted` at the end of the input filename, as the output filename
    # combine the columns from the info files with the nwk.rooted files to prepare the final output file
    with open(input_file_path + '.rooted', 'w') as rootedOutputFO:
        for i, nwk_split_filepath in enumerate(nwk_split_files):
            # read in the corresponding split nwk.rooted file
            rooted_nwk_split_filepath = f'{nwk_split_filepath}.rooted'
            info_split_filepath = info_split_files[i]
            with open(rooted_nwk_split_filepath, 'r') as rooted_nwk_split_FO:
                rooted_nwk_trees = [
                    i.strip() for i in rooted_nwk_split_FO.readlines()
                ]   # remove trailing newlines. This is a list of rooted trees

            # combine the info and rooted trees
            with open(info_split_filepath, 'r') as info_file:
                rooted_info_plus_tree_chunks = [
                    '\t'.join(
                        [info_line.strip(), rooted_nwk_trees[line_index]])
                    for line_index, info_line in enumerate(info_file)
                ]

            # write out the new tree line chunks to the output file
            rootedOutputFO.write(
                '\n'.join(rooted_info_plus_tree_chunks) + '\n')

            # Clear memory
            del rooted_info_plus_tree_chunks
            del rooted_nwk_trees

    # # delete the split nwk files in the tmp folder
    for split_file_name in nwk_split_files:
        os.remove(split_file_name)
    # combine all the log files into one log file using `tail -n +1`, and write in the input_dir with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    os.system('tail -n +1 '+tmp_dir+'/*.stdout > '+input_dir +
              '/tmp_mad_rooting_splits_'+timestamp+'.log')
    # # delete the tmp folder
    # os.system('rm -rf '+tmp_dir)


if __name__ == '__main__':

    start_time = time.time()
    # Catches and interprets the user input from the command line.
    parser = argparse.ArgumentParser()
    # General stuff
    parser.add_argument('--Input', '-i', required=True, type=str,
                        help="path to input file which should be tsv format with two columns: first column is the tree ID and the second column is the tree in newick")
    parser.add_argument('--mad', '-m', required=False, type=str, default='mad',
                        help="path to MAD executable. Assumes mad is within $PATH by default, if this argument is not provided.")
    parser.add_argument('--nlines', '-n', required=False, type=int, default=50,
                        help="(Int) default=50. \
                        Number of lines that the input file will be split into for each split\
                        file. Smaller number implies less trees being processed by one MAD process.")
    parser.add_argument('--num_subprocesses', '-p', required=False, type=int, default=50,
                        help="(Int) default=50. Number of processes to run in parallel. \
                        The larger this number, the more number of processes that will \
                        be run in parallel, since each split file chunk gets its own process")

    # parser.add_argument('--Output', '-o', required=True, type=str,
    #                     help="path to output files")

    args = parser.parse_args()

    MAD_roots_NOG_trees_parallel(
        args.Input, args.mad, args.nlines, args.num_subprocesses)

    print('Process time taken from start to finish: ' +
          str(timedelta(seconds=time.time() - start_time)))
