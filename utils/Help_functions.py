import numpy as np

def true_intervals(bool_arr):
    """ Get intervals where bool_arr is true"""

    mask = np.concatenate([[True], ~bool_arr, [True]])
    interval_bounds = np.flatnonzero(mask[1:] != mask[:-1]).reshape(-1, 2)
    interval_sizes = interval_bounds[:, 1] - interval_bounds[:, 0]

    return interval_bounds, interval_sizes


def extreme_points(data, limit=(49, 51)):
    f_too_low = np.argwhere((data < limit[0]).values)[:, 0]
    f_too_high = np.argwhere((data > limit[1]).values)[:, 0]

    print('Number of too high frequency values: ', f_too_high.size,
          'Number of too low frequency values: ', f_too_low.size)

    return f_too_low, f_too_high


def extreme_inc(increments, limit=0.05):
    inc_too_high = np.argwhere((increments.abs() > limit).values)[:, 0]

    print('Number of too large increments: ', inc_too_high.size)

    return inc_too_high


def const_windows(increments, limit=60):
    wind_bounds, wind_sizes = true_intervals(increments.abs() < 1e-9)

    long_windows = [[]]
    long_window_bounds = wind_bounds[wind_sizes > limit]

    if long_window_bounds.size != 0:
        long_windows = np.hstack([np.r_[i:j] for i, j in long_window_bounds])

    print('Number of windows with constant frequency for longer than {}s: '.format(limit),
          long_window_bounds.shape[0])

    return wind_bounds, wind_sizes, long_windows, long_window_bounds


def nan_windows(data):
    wind_bounds, wind_sizes = true_intervals(data.isnull())

    print('Number of Nan-intervals: ', wind_sizes.shape[0])

    return wind_bounds, wind_sizes


def isolated_peaks(increments, limit=0.05):
    high_incs = increments.where(increments.abs() > limit)
    peak_locations = np.argwhere((high_incs * high_incs.shift(-1) < 0).values)[:, 0]

    print('Number of isolated peaks: ', peak_locations.size)

    return peak_locations
