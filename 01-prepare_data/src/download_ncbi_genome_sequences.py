import time
import os
import subprocess
from loguru import logger
import pandas as pd
from multiprocessing import Pool
import shutil
import traceback
import io
import zipfile


# Uses ncbi `datasets` to download each genome assembly sequence
def download_genome_sequences(args):
    """
    Download genome sequences from NCBI using ncbi `datasets` CLI program
    Args:
        acc_id (str): assembly accession ID
        output_dir (str): path to the output directory
        n_retries (int): number of retries to perform
    Returns:
        None
    """
    # unpack the arguments 
    taxon_id, acc_id, output_dir, n_retries, files_to_download, bin_dir = args
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # run ncbi `datasets`
    try:
        ncbi_retries = n_retries
        while ncbi_retries > 0:
            try:

                logger.info(f"Running datasets for assembly accession ID: {
                            acc_id}, taxon ID: {taxon_id}")
                download_process = subprocess.run(
                    f"{bin_dir}/datasets download genome accession {acc_id} --include {files_to_download} --filename {output_dir}/{acc_id}.zip", shell=True)
                logger.info(
                    f"now extracting files for assembly accession ID: {acc_id}, taxon ID: {taxon_id}")
                # if download_process didn't have success exit code, log error and try again
                if download_process.returncode != 0:
                    logger.error(f"datasets failed for assembly accession ID: {acc_id}, taxon ID: {
                                 taxon_id} with error code {download_process.returncode}")
                    if ncbi_retries == 0:
                        logger.error(f"no more retries for assembly accession ID: {acc_id}, taxon ID: {
                                     taxon_id} after {n_retries} retries, skipping to the next assembly accession ID")
                        break
                    else:
                        logger.info(f"Retrying datasets for assembly accession ID: {
                                    acc_id}, taxon ID: {taxon_id}, retries left: {ncbi_retries}")
                    ncbi_retries -= 1
                    continue
                
                # extract the files from the zip archive, except jsonl files
                # basically rename each of the files to {taxon_id}_{acc_id}_{original_file_name}
                with zipfile.ZipFile(f"{output_dir}/{acc_id}.zip", 'r') as zip_ref:
                    if len(zip_ref.namelist()) == 0:
                        logger.error(f"no files found in zip archive for assembly accession ID: {acc_id}, taxon ID: {taxon_id}")
                        break
                    for file_i in zip_ref.namelist():
                        if file_i.endswith('.jsonl') or file_i.endswith('.json') or file_i.endswith('README.md'):
                            continue
                        logger.info(f"extracting file: {file_i} for assembly accession ID: {acc_id}, taxon ID: {taxon_id}")
                        with zip_ref.open(file_i) as f:
                            file_i_basename = os.path.basename(file_i)
                            # check if file_i_basename starts with acc_id
                            if file_i_basename.startswith(acc_id):
                                with open(f"{output_dir}/{taxon_id}_{file_i_basename}", 'wb') as f_out:
                                    f_out.write(f.read())
                            else:
                                with open(f"{output_dir}/{taxon_id}_{acc_id}_{file_i_basename}", 'wb') as f_out:
                                    f_out.write(f.read())
                    
                # if the command runs successfully, break out of the while loop
                break
            except subprocess.CalledProcessError as e:
                logger.error(f"datasets failed for assembly accession ID: {
                             acc_id}, taxon ID: {taxon_id} with error\n {e}")
                if ncbi_retries == 0:
                    logger.error(
                        f"No more retries left for assembly accession ID: {acc_id}, taxon ID: {taxon_id} after {n_retries} retries, skipping to the next assembly accession ID")
                    break
                else:
                    logger.info(
                        f"Retrying datasets for assembly accession ID: {acc_id}, taxon ID: {taxon_id}, retries left: {ncbi_retries}")
                ncbi_retries -= 1
                continue
    except subprocess.CalledProcessError as e:
        logger.error(f"datasets failed with error\n {e}")
        return None
    # 
    return taxon_id


def retrieve_accession_info(args):
    """
    Fn for `multiprocessing.Pool` to retrieve assembly accession ID for a given taxon ID.
    """
    taxon_id, n_retries, bin_dir = args
    try:
        datasets_retries = n_retries
        while datasets_retries > 0:
            try:
                logger.info(f"Running datasets for taxon ID: {taxon_id}")
                #
                p1 = subprocess.Popen(
                    f"{bin_dir}/datasets summary genome taxon {taxon_id} --as-json-lines", shell=True, stdout=subprocess.PIPE)
                logger.info(
                    f"now running dataformat for taxon ID: {taxon_id}")
                tsv_records_process = subprocess.run(
                    f"{bin_dir}/dataformat tsv genome --fields accession,assminfo-name,source_database,assminfo-refseq-category", shell=True, stdin=p1.stdout, stdout=subprocess.PIPE)
                # if the command runs successfully, break out of the while loop
                if tsv_records_process.returncode == 0:
                    if len(tsv_records_process.stdout) == 0:
                        logger.info(
                            f"No records found for taxon ID: {taxon_id}")
                        return None
                    else:
                        tsv_records = tsv_records_process.stdout.decode(
                            'utf-8').split('\n')
                        break
                else:
                    logger.error(
                        f"dataformat failed for taxon ID: {taxon_id} with returncode {tsv_records_process.returncode} and error:\n {tsv_records_process.stderr}")
                    datasets_retries -= 1
                    if datasets_retries == 0:
                        logger.error(
                            f"datasets failed for taxon ID: {taxon_id} after {n_retries} retries for taxon ID: {taxon_id}.")
                        return None
            except subprocess.CalledProcessError as e:
                logger.error(f"datasets failed for taxon ID: {
                            taxon_id} with error\n {get_exception_traceback_str(e)}")
                datasets_retries -= 1
                if datasets_retries == 0:
                    logger.error(
                        f"datasets failed for taxon ID: {taxon_id} after {n_retries} retries for taxon ID: {taxon_id}.")
                    return None
                else:
                    logger.info(
                        f"Retrying datasets for taxon ID: {taxon_id}, retries left: {datasets_retries}")
                continue

        # remove header
        if tsv_records[0].startswith('Assembly Accession'):
            tsv_records = tsv_records[1:]
        # if there are no records, log an error and continue to the next taxon ID
        if len(tsv_records) == 0:
            logger.info(f"No records found for taxon ID: {taxon_id}")
            # this typically means that the ID itself doesn't have a genome associated with it
            # but the ID is valid and could be a strain ID.

        # if there are records, log the number of records found
        logger.info(f"Found {len(tsv_records)
                            } records for taxon ID: {taxon_id}")
        try:
            if tsv_records[-1] == '':
                tsv_records = tsv_records[:-1]
            # for each record, extract the assembly accession ID and source database
            # for record in tsv_records:
            # print('\n'.join(tsv_records))
            # if there's only one record, that is the acc_id
            if len(tsv_records) == 1:
                assembly_name = tsv_records[0].split('\t')[1]
            # elif there are 2 or more records,
            # check if one of them is RefSeq or if one of the RefSeq records has category of 'representative genome' or 'reference genome'
            elif len(tsv_records) >= 2:
                refseq_record = None
                for record in tsv_records:
                    if record.split('\t')[2].endswith('REFSEQ'):
                        refseq_record = record
                        if record.split('\t')[3] in ['representative genome', 'reference genome']:
                            assembly_name = tsv_records[0].split('\t')[1]
                            break
                    refseq_record = None
                if refseq_record is None:
                    assembly_name = tsv_records[0].split('\t')[1]
            # now use the assembly_name to get the acc_id of the record
            # i.e. record in tsv_records with the same assembly_name but from GENBANK
            # if there's no record from GENBANK, use the first record
            taxon_record = [record for record in tsv_records if record.split(
                '\t')[1] == assembly_name and record.split('\t')[2] == 'GENBANK']
            if len(taxon_record) == 0:
                taxon_record = tsv_records[0]
            else:
                taxon_record = taxon_record[0]
            return {'taxon_id': taxon_id, 'acc_id': taxon_record.split('\t')[0],
                'assembly_name': taxon_record.split('\t')[1], 'source_db': taxon_record.split('\t')[2]}
        except Exception as e:
            logger.info(
                f"TSV records list that was not processed: {tsv_records}")
            logger.error(f"Failed to extract acc_id for taxon ID: {
                        taxon_id}. Traceback\n {get_exception_traceback_str(e)}")
            return None

    except Exception as e:
        logger.error(f"datasets failed for taxon ID: {
                    taxon_id}. Traceback\n {get_exception_traceback_str(e)}")
        return None



def get_exception_traceback_str(exc: Exception) -> str:
    # Ref: https://stackoverflow.com/a/76584117/
    iofile = io.StringIO()
    traceback.print_exception(exc, file=iofile)
    return iofile.getvalue().rstrip()


def process_taxa_ids(input_file, taxon_ids_list, output_dir, threads, n_retries, files_to_download, bin_dir, acc_id_tsv_filepath):
    """
    This function is used to download genome sequences from NCBI using ncbi-genome-download.
    Args:
        input_file (str): path to the input file containing taxon IDs
        taxon_ids_list (list): list of taxon IDs
        output_dir (str): path to the output directory
        threads (int): number of threads to use
        n_retries (int): number of retries to perform if failures are encountered
        files_to_download (str): files to download from NCBI
        bin_dir (str): path to the directory where the `datasets` and `dataformat` CLI programs are located
        acc_id_tsv_filepath (str): path to the file containing assembly accession IDs
        debug (bool): run in debug mode
    Returns:
        None: if the function runs successfully
    """
    
    to_download_info = False

    # if the acc_id_tsv_filepath is provided, read in the assembly accession IDs from the file and use this df to download genome/gff/etc.
    if acc_id_tsv_filepath is not None:
        logger.info(f"Assembly accession IDs file provided, reading in assembly accession IDs from file: {acc_id_tsv_filepath}")
        taxon_id_acc_id_df = pd.read_csv(acc_id_tsv_filepath, sep='\t')
        taxon_id_acc_id_df['taxon_id'] = taxon_id_acc_id_df['taxon_id'].astype(str)
        logger.info(f"Read in assembly accession IDs. Number of records: {len(taxon_id_acc_id_df)}")
        acc_id_tsv_taxon_ids_list = taxon_id_acc_id_df['taxon_id'].tolist()
        taxa_without_acc_list = set(taxon_ids_list) - set(acc_id_tsv_taxon_ids_list)
        logger.info(f"taxon IDs list looks like this: {taxon_ids_list[:5]}, and in acc_id_tsv file: {acc_id_tsv_taxon_ids_list[:5]}. Difference: {len(taxa_without_acc_list)}")
        # check if the taxon IDs in the file are the same as the taxon IDs in the input file. If not, then to_download_info = True
        if len(taxa_without_acc_list) > 0:
            to_download_info = True
            logger.info(f"{acc_id_tsv_filepath} doesn't have Acc IDs for {len(taxa_without_acc_list)} taxon IDs that are in the input file")
        else:
            to_download_info = False
    else:
        taxon_id_acc_id_df = pd.DataFrame(columns=['taxon_id', 'acc_id', 'assembly_name', 'source_db'])
        to_download_info = True
        taxa_without_acc_list = taxon_ids_list
    
    if to_download_info: # if the assembly accession IDs file contains taxon IDs that are not in the input file, run datasets to get assembly accession IDs for these taxon IDs 
        logger.info(f"Running datasets to get assembly accession IDs for each taxon ID")
        
        # run ncbi 'datasets' to query for assembly accession IDs for each taxon ID, and write it into a pandas dataframe
        taxon_id_acc_id_df2 = pd.DataFrame(columns=['taxon_id', 'acc_id', 'assembly_name', 'source_db'])
        taxon_id_acc_id_list = []
        # use multiprocessing to run the datasets function for each taxon ID, and write the results to a dataframe
        with Pool(processes=threads) as pool:
            for pool_out in pool.imap(retrieve_accession_info, [(taxon_id, n_retries, bin_dir) for taxon_id in taxa_without_acc_list]):
                if pool_out is not None:
                    taxon_id_acc_id_list.append(pool_out)

        taxon_id_acc_id_df2 = pd.DataFrame(taxon_id_acc_id_list)
        # remove rows where duplicates of the acc_id column exist
        taxon_id_acc_id_df2.drop_duplicates(subset='acc_id', inplace=True)
        # add these new records to the existing taxon_id_acc_id_df
        taxon_id_acc_id_df = pd.concat([taxon_id_acc_id_df, taxon_id_acc_id_df2], ignore_index=True)
        # write the new taxon_id_acc_id_df to a file
        taxon_id_acc_id_df.to_csv(
            f"{output_dir}/{input_file.split('/')[-1].split('.')[0]}_accession_ids.tsv", sep='\t', index=False, quoting=1)
        logger.info(f"Assembly accession IDs written to file: {output_dir}/{input_file.split('/')[-1].split('.')[0]}_accession_ids.tsv")

        # if the taxon_id_acc_id_df is empty and there was no original acc_id_tsv_filepath provided, log an error and exit
        if taxon_id_acc_id_df.empty and acc_id_tsv_filepath is None:
            logger.error(f"No assembly accession IDs found for any taxon ID, exiting")
            return None
        elif taxon_id_acc_id_df.empty and acc_id_tsv_filepath is not None:
            logger.error(f"No assembly accession IDs found for any additional taxon ID. Continuing with the original assembly accession IDs file")

    # now we have the assembly accession IDs, we can use ncbi-genome-download to download the genome sequences, in parallel
    # create args list for parallel processing
    tax_acc_tuples_list = list(
        zip(taxon_id_acc_id_df['taxon_id'], taxon_id_acc_id_df['acc_id']))
    logger.info(f"Number of assembly accession IDs in dataframe: {len(taxon_id_acc_id_df)}")
    
    # which `files_to_download` files are already downloaded in output_dir?
    files_to_download_list = files_to_download.split(',')
    extensions_dict = {'genome': '.fna', 'gff3': '.gff', 
                  'gbff': '.gbff', 'protein': '.faa', 
                  'cds': '.ffn', 'rna': '.fna', 
                  'gtf': '.gtf'}
    for file_type in files_to_download_list:
        if not all([os.path.exists(f"{output_dir}/{taxon_id}_{acc_id}{extensions_dict[file_type]}") for taxon_id, acc_id in tax_acc_tuples_list]):
            logger.info(f"Downloading {file_type} files")
            tax_acc_tuples_list = [(taxon_id, acc_id) for taxon_id, acc_id in tax_acc_tuples_list if not os.path.exists(f"{output_dir}/{taxon_id}_{acc_id}{extensions_dict[file_type]}")]
            logger.info(f"Number of assembly accession IDs to download: {len(tax_acc_tuples_list)}")
            if len(tax_acc_tuples_list) == 0:
                logger.info(f"All {file_type} files already downloaded, skipping")
                continue

    args_list = [(taxon_id, acc_id, output_dir, n_retries, files_to_download, bin_dir)
                 for taxon_id, acc_id in tax_acc_tuples_list]
    logger.info(f"Downloading {files_to_download} from NCBI using {
        threads} threads, input file: {input_file}, output directory: {output_dir}")
    with Pool(processes=threads) as pool:
        # iterate through what has been created via pool.imap()
        for pool_out in pool.imap(download_genome_sequences, args_list):
            if pool_out is not None:
                logger.info(
                    f"Downloading {files_to_download} for taxon ID: {pool_out} finished successfully")

    logger.info(f"Downloading genome sequences finished successfully")
    return None


if __name__ == '__main__':
    # log time
    logger.info(f"Running {__file__} at {time.asctime()}")

    start_time = time.time()

    # import argparse and read in the arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Download genome sequences from NCBI')
    parser.add_argument(
        '--input_file', '-i', help='Path to the input file containing taxon IDs', type=str, required=True)
    parser.add_argument('--output_dir', '-o', help='Path to the output directory',
                        type=str, default='../data/genome_data/')
    parser.add_argument('--threads', '-t',
                        help='Number of threads to use', type=int, default=100)
    parser.add_argument('--to-download', type=str, default='genome,gff3',
                        help='Files to dload from NCBI. Default is genome,gff3. \
                            See NCBI datasets CLI documentation for more possibilities.')
    parser.add_argument(
        '--retries', '-r', help='Number of retries to perform if failures are encountered, for each taxon ID input.', type=int, default=5)
    parser.add_argument(
        '--debug', '-d', help='Run in debug mode (runs first <--threads> entries only)', action='store_true')
    parser.add_argument(
        '--bin-dir', '-b', type=str, default='/root/mambaforge/envs/hgt_analyses/bin/',
        help='Path to dir where `datasets` and `dataformat` CLI programs are.')
    parser.add_argument('--acc_ids_tsv', '-a',
                        type=str, default=None,
                        help='Path to the file containing assembly accession IDs')
    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # read in all the taxon IDs from the input file
    with open(args.input_file, 'r') as f:
        taxon_ids_list = [line.strip()
                     for line in f.readlines() if line.strip() != '']

    # debugging: run only for first 'threads' lines using a tmp_ file
    if args.debug:
        logger.debug(f"Running in debug mode, using only first {
                     args.threads} taxon IDs")
        taxon_ids_list = taxon_ids_list[:args.threads]

    # run the main function
    process_taxa_ids(args.input_file, taxon_ids_list, args.output_dir, args.threads, args.retries, args.to_download, args.bin_dir, args.acc_ids_tsv)

    # log the time
    end_time = time.time()
    duration = end_time - start_time
    human_readable_duration = time.strftime("%H:%M:%S", time.gmtime(duration))
    logger.info(f"Finished running {__file__} in {human_readable_duration}")
