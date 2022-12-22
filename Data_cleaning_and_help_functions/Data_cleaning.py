import pandas as pd
import numpy as np
import scipy as sc
import seaborn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d
#from scipy.ndimage import uniform_filter1d
from scipy.stats import beta
import sdeint
#!pip install kramersmoyal
from kramersmoyal import km
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy

from .Data import data 

data_cleaning(data)
    limit = 0.05

    data_clean = data.copy()
    high_val = np.argwhere((data).values>51)
    low_val  = np.argwhere((data).values<49)
    #np.shape(freq_old.values)
    data_clean[high_val[:,0]]=np.nan
    data_clean[low_val[:,0]]=np.nan

    inc_too_high = np.argwhere((np.abs(np.diff(data)) > limit))[:, 0]
    data_clean[inc_too_high]=np.nan

    bool_arr = (np.abs(np.diff(data)) < 1e-9)

    mask = np.concatenate([[True], ~bool_arr, [True]])
    interval_bounds = np.flatnonzero(mask[1:] != mask[:-1]).reshape(-1, 2)
    interval_sizes = interval_bounds[:, 1] - interval_bounds[:, 0]
    np.shape(interval_sizes)

    wind_bounds, wind_sizes = interval_bounds, interval_sizes
    long_windows = [[]]
    long_window_bounds = wind_bounds[wind_sizes > 60]

    if long_window_bounds.size != 0:
        long_windows = np.hstack([np.r_[i:j] for i, j in long_window_bounds])

    long_window_indices = long_windows

    if long_window_indices != [[]]:
        data_clean[long_window_indices]=np.nan

    # for i in np.size(freq):
    #     if all(np.abs(np.diff(freq_old[i:i+10]))<1e-9):
    #         freq_m[i]=np.nan


    data_clean = pd.Series(data_clean)
    N_f=9
    data_clean = data_clean.fillna(method='ffill', limit=N_f)
    if data_clean.size > 0:
        data_clean = data_clean.fillna(method='ffill', limit=N_f)
        if data_clean.size > 0:
            data_clean = data_clean.fillna(method='ffill', limit=N_f)
        
    return data_clean
