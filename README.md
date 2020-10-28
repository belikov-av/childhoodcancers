# Childhood cancers

## Description
Current repository contains code to reproduce the article "The Poisson process is the universal law of cancer development: driver mutations accumulate randomly, silently, at constant rate and for many decades, likely in stem cells" by Aleksey V Belikov , Alexey D Vyatkin and Sergey V Leonov: (**paste link here**).

## Requirements and installation
Analysis is done with Python 3.7.

To install all the requirements, please run the command:

`pip install -r requirements.txt`

## Scripts description
In order to read information about script arguments,
please run the command:

`python <script_path> -h`, 

where `script_path` is the path to the script.

Detailed scripts' description is presented below.

### 1. Distribution parameters calculation

The `grid_search.py` script calculates the R<sup>2</sup> score for every pair of parameters
for the current distribution and the current grid.
To change the grid, please change the `configs/fit_configs/<distribution_name>.yml` file.

:floppy_disk: Input data for the script are contained in the `data/Childhood Incidence Data.xlsx`
file. Detailed explanation of the file content and the way it was retrieved could be found in the article.

The script starts calculation for all cohorts in parallel.

:heavy_exclamation_mark:**Warning**:heavy_exclamation_mark:
Running this script with default grid parameters takes about 7 hours on CPU.

### 2. Optimal parameters and intervals extraction

The `find_all_opts.py` script takes as an input the path to the folder,
which should contain `<distribution_name>_child_r2` folders 
(outputs from the `grid_search.py` script for every distribution).
Then the script finds the optimal parameters and the R<sup>2</sup> score for each
distribution and each cohort, and gathers it in one table.

The `find_intervals.py` script takes as an input the path to the folder, 
which is the output of the `grid_search,py` script, thus contains tables with 
parameters and scores for current distribution and all cohorts.
Then it finds the confidence interval for each parameter.

### 3. Plotting metric surface

The `plot_surfaces.py` script  takes as an input the path to the output folder from the `grid_search.py` script 
and visualizes the results. Basically, it plots the contour plot for two parameters and the
R<sup>2</sup> score. Note that in `configs/plot_configs/<distribution_name>_plot.yml`
file (`distribution_name` is the name of the current distribution) you may change plotting parameters.

Also note that it is possible to change many plotting parameters, as in the config file
`configs/plot_configs/erlang_plot_small.yml`.
