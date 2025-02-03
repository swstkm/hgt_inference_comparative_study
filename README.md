# Comparative Study of HGT Inference Methods

Code to reproduce the results of the paper, "Horizontal Gene Transfer Inference: Gene presence-absence outperforms gene trees" where we perform a comparative study of HGT inference methods. See: [preprint in bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.27.630302).

The project is organized into three main steps (`01`, `02`, `03`), each containing sequentially numbered Jupyter notebooks. Supporting scripts and helper functions referenced in these notebooks are located in the `src/` and `lib/` directories.

## Notebook Usage
- Execute notebooks **cell by cell** instead of using 'Run All'.
- Many cells contain **markdown instructions** for running external shell programs. This is because several processes are computationally intensive and intended for HPC environments with multiprocessing capabilities.

## Data Management
- The `data` directory is initially empty in this repository.
- Running the notebooks sequentially will populate the `data` directory with:
  - Downloaded files
  - Processed results
  - Generated figures
- Alternatively, a complete dataset including results and figures can be downloaded from the [Zenodo link](https://zenodo.org/records/14555036) in the paper.

## Required python packages

A Mamba/Conda environment called `hgt_analyses` was used for all the analyses. This environment with all required packages can easily be created again using the `mamba_packages.yml` file, using the following command:
```
mamba env create -f mamba_packages.yml
```
If you use Conda instead of Mamba just replace `mamba` with `conda` in the above command.

