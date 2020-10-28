# Childhood cancers

## Description
Current repository contains code to reproduce the article: (**past link here**).

## Requirements and installation
Analysis is done with Python 3.7.

To install all the requirements, please run the command:

`pip install -r requirements.txt`

## Scripts description
In order to read information about script arguments,
please run the command:

`python <script_path> -h`, 

where `script_path` is path to the script.

Detailed scripts' description is presented below.

### 1. Distribution parameters calculation

Script `grid_search.py` calculates R<sup>2</sup> score for every pair of parameters
for current distribution and current grid.
To change the grid, please change `configs/fit_configs/<distribution_name>.yml` file.

Script starts calculation for all cohorts in parallel.

**!Warning!** Running this script with default grid parameters takes about 7 hours on CPU.

### 2. Optimal parameters and intervals extraction

Script ``

### 3. Plotting metric surface

Script `plot_surfaces.py` takes as input path output folder from script `grid_search.py` 
and visualize the results. Basically, it plots contour plot for two parameters and 
R<sup>2</sup> score. Note that in `configs/plot_configs/<distribution_name>_plot.yml`
file (`distribution_name` is the name of current distribution) you may change plotting parameters.

Also note that it is possible to change a lot of plotting parameters, as in config file
`configs/plot_configs/erlang_plot_small.yml`.
