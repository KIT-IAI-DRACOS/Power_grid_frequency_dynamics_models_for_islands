limit = 0.05

freq_m = freq_old.copy()
high_val = np.argwhere((freq_old).values>51)
low_val  = np.argwhere((freq_old).values<49)
#np.shape(freq_old.values)
freq_m[high_val[:,0]]=np.nan
freq_m[low_val[:,0]]=np.nan

inc_too_high = np.argwhere((np.abs(np.diff(freq_old)) > limit))[:, 0]
freq_m[inc_too_high]=np.nan

bool_arr = (np.abs(np.diff(freq_old)) < 1e-9)

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
wind_bounds, wind_sizes, long_window_indices, long_window_bounds

if long_window_indices != [[]]:
    freq_m[long_window_indices]=np.nan

# for i in np.size(freq):
#     if all(np.abs(np.diff(freq_old[i:i+10]))<1e-9):
#         freq_m[i]=np.nan


freq_m = pd.Series(freq_m)
N_f=9
freq = freq_m.fillna(method='ffill', limit=N_f)
freq
