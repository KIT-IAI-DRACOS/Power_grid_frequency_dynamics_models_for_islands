import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

#!pip install kramersmoyal
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy

from Data import data 

if __name__ == '__main__':

    # Add path to processed data!
    # path_to_data = '/BalearicIslands/'

    # %%  Set parameters for identifying corrupted data

    # Nan-points to fill
    N_f = 9
    # Maximum length of allowed constant windows
    T_c = 15
    # Minimum height of isolated peaks
    df_c = 0.05

    # %%  Mark and clean corrupted data points

    # Paths to the output files of `convert_data_format.py`
    input_dir = '/Users/thorbjornlundonsaker/workspace/Master/format_data/Ireland/format_Ireland.csv'
    # Paths to the cleansed output
    output_dir = '/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/Ireland/clean_data_I.zip'
    precision = '%.4f'

    print('Marking and cleansing data from {}'.format(input_dir))

    # Read the frequency data and calculate the increments #

    print('Load data ...')
    # file = glob.glob(input_dir + '*.zip')
    # files = np.sort(file)
    data = pd.Series(dtype='float64')
    frame = pd.read_csv(input_dir, index_col=0, usecols=[0, 1], names=['time', 'frequency'],
                        header=None, squeeze=True, parse_dates=[0])
    data = data.append(frame)
    data = data.multiply(0.001).add(50)
    data = data.round(4)
    df = data.diff()

    # Find positions and numbers of corrupted data #

    print('Find corrupted data ...')
    # Indices where f(t) is too high/low
    f_too_low, f_too_high = extreme_points(data, (49, 51))
    # Location of isolated peaks
    peak_loc = isolated_peaks(df, df_c)
    # Right/Left bounds, sizes and indices of windows (>1 point) and long windows (> T_c points)
    window_bounds, window_sizes, long_windows_indices, long_window_bounds = const_windows(df, T_c)
    # Right/Left indices and sizes of missing data
    missing_data_bounds, missing_data_sizes = nan_windows(data)

    # Mark corrupted data as NaN #

    print('Mark corrupted data ...')
    data_m = data.copy()
    # Mark extreme values >51Hz and <49Hz
    data_m.iloc[f_too_low] = np.nan
    data_m.iloc[f_too_high] = np.nan
    # Mark isolated peaks with abs. increments > df_c
    data_m.iloc[peak_loc] = np.nan
    # Mark windows with const. freq. longer than T_c
    data_m.iloc[long_windows_indices] = np.nan

    # Cleansing data by filling intervals of missing/ corrupted data #

    # You can add your own cleansing procedure!
    # Here, we fill up to N_f values by propagating the last valid entry
    print('Clean corrupted data ...')
    data_cl = data_m.fillna(method='ffill', limit=N_f)

    # data_cl = data_cl.subtract(50.000).multiply(1000.000)

    # Save cleansed data (including the remaining NaN-values) #

    print('Saving the results ...')
    data_cl.to_csv(output_dir, float_format=precision, na_rep='NaN',
                   compression={'method': 'zip', 'archive_name': 'Ireland_cleaned.csv'}, header=False)






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
