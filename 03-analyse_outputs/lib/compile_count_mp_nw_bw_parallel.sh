#!/bin/bash

# Arguments
program_runs_dir=$1
taxonomic_id=$2
input_tree_filepath=$3
res_dir=$4

# compiling nogwise.branchwise file is somewhat tricky, since we don't directly get info of nogwise transfers per branch
# but in the nogwise transfers output file (_families.tsv), we have the number of genes in each NOG in each branch, including internal ones
# we can use this info to calculate the transfers per branch for each NOG
# basically, we traverse the input tree, and for each NOG, for each branch, the total number of gains is the genes in the branch of this NOG, minus the genes in the ancestor.
# we do this by calling `output_compilation_functions.py` using bash, in parallel, across all the `_family.tsv` files
# with the output filename derived from the input filename with the gain penalty ratio included
# the output files are of the form `compiled_transfers.nogwise.branchwise.count.mp.{gain_penalty_ratio}.tsv`

# input_dir is the directory where all the _families.tsv files are present
input_dir="$program_runs_dir/Count/Count_MP/"
echo "Input dir is $input_dir"
# output_dir is the directory where all the output files will be written
output_dir="$res_dir/count_mp/"
# make the output dir if it doesn't exist
mkdir -p $output_dir
tree_filepath="$input_tree_filepath"
echo "Tree filepath is $tree_filepath and output dir is $output_dir"
# the python script to run
script_path="lib/compile_count_mp_nw_bw_parallel.py"

# the function to call
process_file() {
    input_file=$1
    output_dir=$3
    # extract gain penalty ratio from the filename
    gain_penalty_ratio=$(basename $input_file .tsv | rev | cut -d'_' -f2 | rev)
    output_file="$output_dir/compiled_transfers.nogwise.branchwise.count.mp.$gain_penalty_ratio.tsv"
    tree_filepath=$2

    echo "Running: python3 $script_path -c $input_file -t $tree_filepath -o $output_file"
    python3 $script_path -c $input_file -t $tree_filepath -o $output_file
}

# export the function so that it's available to parallel
export -f process_file
export input_dir
export output_dir
export script_path

# get the list of all the _families.tsv files that start with the taxonomic_id
# these files are of the form `1236_Count_output_gain_{0.x}_families.tsv` where {0.x} is the gain penalty ratio used
input_files=($(ls $input_dir | grep $taxonomic_id | grep "_families.tsv"))
# prefix the filename with the input_dir path to get the full path
for i in "${!input_files[@]}"; do
    input_files[$i]="$input_dir/${input_files[$i]}"
done
echo "Files are ${input_files[@]}"
echo "Number of files is ${#input_files[@]}"
num_files=${#input_files[@]}

# run the function in parallel
# the output filepath is of the form `compiled_transfers.nogwise.branchwise.count.mp.{gain_penalty_ratio}.tsv` where gain_penalty_ratio is the number in the filename (index -2)
parallel -j $num_files process_file ::: "${input_files[@]}" ::: $tree_filepath ::: $output_dir