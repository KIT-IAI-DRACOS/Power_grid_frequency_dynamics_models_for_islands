
Python code to reproduced the figures in the scientific publication "Non-linear, bivariate stochastic modelling of
power-grid frequency applied to islands",
[//]: # submitted to [IEEE Access](https://doi.org/10.1109/ACCESS.2022.3150338)

## Python packages
In the following follow various python scripts to obtain the figures in the scientific publication. They requires a set of python packages, which are listed below, and can easily be installed using `pip`.

Using a conventional `python >v3.4` installation, e.g., with `anaconda`, most of the standard packages should be included. These are

```code
 - numpy
 - scipy
 - matplotlib
```

two additional packages are needed. Firstly, one for estimating the Kramer–Moyal coefficients from stochastic data. The package `kramersmoyal` ([kramersmoyal](https://github.com/LRydin/kramersmoyal)) can be installed via


```code
pip install kramersmoyal
```

Secondly, `cartopy` is needed to produce the map in figure 1 (`https://github.com/KIT-IAI-DRACOS/Stochastic-modelling-of-power-grid-frequency-applied-to-islands/blob/main/Create_figures/figure_1.py`). This package can prove difficult to install, does we do not recommend the user to install it as long as reproducing the map is strictly needed.

## Data

The original Data can be found in the folder "Data".

With the plots come a few additional data sources that have generated the figures and avoid long computational times.

## Plots

