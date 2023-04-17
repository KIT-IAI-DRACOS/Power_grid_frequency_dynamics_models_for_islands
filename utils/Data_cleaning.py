import numpy as np
from utils.Help_functions import extreme_points, isolated_peaks, const_windows, nan_windows, extreme_inc

def data_cleaning(data ):
    # %%  Set parameters for identifying corrupted data
    # Nan-points to fill
    N_f = 20 #6 #20 for Iceland, 6 are suitable for the other grids
    # Maximum length of allowed constant windows
    T_c = 60 #15
    # Minimum height of isolated peaks
    df_c = 0.05 #0.05

    # %%  Mark and clean corrupted data points

    df = data.diff()

    # Find positions and numbers of corrupted data #

    # Indices where f(t) is too high/low
    f_too_low, f_too_high = extreme_points(data, (49, 51))

    # Location of high increments
    peak_loc = extreme_inc(df, df_c) 

    # Right/Left bounds, sizes and indices of windows (>1 point) and long windows (> T_c points)
    window_bounds, window_sizes, long_windows_indices, long_window_bounds = const_windows(df, T_c)
    # Right/Left indices and sizes of missing data
    missing_data_bounds, missing_data_sizes = nan_windows(data)
    
    # Mark corrupted data as NaN #
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

    return data_cl


