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

Use the Mamba/Conda environment in `hgt_analyses.yml` for all the analyses, except when you run AnGST (i.e. `02-run_programs/src/run_angst.py`) in which case you need to use the python2 environment `hgt_analyses_py2.yml`.
