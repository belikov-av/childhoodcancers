# Childhood cancers

## Description
Current repository contains code for reproduce article: (**past link here**).

## Requirements and installation
Analysis is done with Python 3.7.

To install all the requirements, please run the command:

```pip install -r requirements.txt```

## Scripts description
In order to read information about script arguments,
please run the command:
```python <script_path> -h```, 
where `script_path` is path to the script.

Detailed scripts' description is presented below.

### 1. Distribution parameters calculation

Script `surf_not_ln.py` calculates R<sup>2</sup> score for every pair of parameters
for current distribution and current grid.
To change the grid, please change `configs/fit_configs/<distribution_name>.yml` file.

Script starts calculation for all cohorts in parallel.

**!Warning!** Running this script with default grid parameters takes about 7 hours on CPU.
