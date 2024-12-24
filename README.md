# Comparative Study of HGT Inference Methods

Code to reproduce the results of the paper, "Horizontal Gene Transfer Inference: Gene presence-absence outperforms gene trees" where we perform a comparative study of HGT inference methods.

Each folder for steps 01, 02, and 03, contain Jupyter notebooks that were run in the order of their numbering. Scripts and other helper functions that they depend on or are mentioned in the notebooks can be found in the `src/` or `lib/` directories respectively.

Please note that using 'Run all' or equivalent in the jupyter notebooks will generally not be useful. Some of the intervening steps in the notebooks are markdown cells instructing how to run programs via shell, separately. These programs are time consuming ones, often utilising multiprocessing in an HPC. Please run the notebooks, cell by cell, keeping this in mind.

This study downloads and processes an number of large files, which were stored in the `data` directory, which is empty in this repo. If you follow/run the notebooks you will progressively fill the `data` directory to reproduce all the results as well a the figures in the paper. Alternatively, you can download this repo containing the results and figures in the `data` directory, from the Zenodo link in the paper.

## Required python packages

A Mamba/Conda environment called `hgt_analyses` was used for all the analyses. This environment with all required packages can easily be created again using the `mamba_packages.yml` file, using the following command:
```
mamba env create -f mamba_packages.yml
```
I use Mamba since it's faster than using regular Conda but if you use Conda just replace `mamba` with `conda` in the above command.

