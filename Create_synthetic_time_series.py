#!pip install kramersmoyal
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d
#from scipy.ndimage import uniform_filter1d
import sdeint

from kramersmoyal import km
from scipy.stats import entropy

from Data_cleaning import data_cleaning
from Functions import data_filter, integrate_omega, KM_Coeff_1, KM_Coeff_2, daily_profile, power_mismatch, exp_decay, Euler_Maruyama, Increments, autocor 

grids = ['Iceland','Irish','Balearic']
models = ['model 1','model 2','model 3','model 4']

'''The bandwidth is chosen such that we receive a scmooth distribution'''


'''Model 1...'''
synth_data_model_1 = {}
increments_model_1 = {}
autocor_model_1 = {}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
  data = ...
  data = data_cleaning(data)
  bw_drift = 0.1
  bw_diff = 0.1
  dist_drift = 500
  dist_diff = 500
  
  c_1 = ...
  epsilon = ...
  omega_synth_model_1 = Euler_Maruyama(...)
  
  synth_data_model_1(grid) = omega_synth_model_1
  increments_model_1(grid) = Increments(omega_synth_model_1,...)
  autocor_model_1(grid) = autocor(omega_synth_model_1)
  
  '''Model 2...'''
synth_data_model_2 = {}
increments_model_2 = {}
autocor_model_2 = {}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
  data = ...
  data = data_cleaning(data)
  bw_drift = 0.1
  bw_diff = 0.1
  dist_drift = 500
  dist_diff = 500
  if grid = 'Balearic':
    data = data_filter(data)
  elif grid = 'Irish':
    data = data_filter(data)
  #elif grid = 'Iceland':
  c_1 = ...
  c_2 = ...
  if grid != 'Iceland':
    Delta_P = ...
  epsilon = ...
  omega_synth_model_2 = Euler_Maruyama(...)
  
  synth_data_model_2(grid) = omega_synth_model_2
  increments_model_2(grid) = Increments(omega_synth_model_2,...)
  autocor_model_2(grid) = autocor(omega_synth_model_2)

   '''Model 3...'''
synth_data_model_3 = {}
increments_model_3 = {}
autocor_model_3 = {}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
  data = ...
  data = data_cleaning(data)
  bw_drift = 0.1
  bw_diff = 0.1
  dist_drift = 1200
  if grid = 'Balearic':
    data = data_filter(data)
    dist_diff = 350
  elif grid = 'Irish':
    data = data_filter(data)
    dist = 500
  elif grid = 'Iceland':
    dist_diff = 300
  c_1 = ...
  c_2 = ...
  if grid != 'Iceland':
    Delta_P = ...
  epsilon = ...
  omega_synth_model_3 = Euler_Maruyama(...)
  
  synth_data_model_3(grid) = omega_synth_model_3
  increments_model_3(grid) = Increments(omega_synth_model_3,...)
  autocor_model_3(grid) = autocor(omega_synth_model_3)
  
   '''Model 4...'''
synth_data_model_4 = {}
increments_model_4 = {}
autocor_model_4 = {}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
  data = ...
  data = data_cleaning(data)
  bw_drift, bw_diff = 0.05, 0.05
  if grid = 'Balearic':
    dist_drift, dist_diff = 20,20
  elif grid = 'Irish':
    dist_drift, dist_diff = 15,15
  elif grid = 'Iceland':
    dist_drift, dist_diff = 20,20
  c_1 = ...
  c_2 = ...
  Delta_P = ...
  epsilon = ...
  omega_synth_model_4 = Euler_Maruyama(...)
  
  synth_data_model_4(grid) = omega_synth_model_4
  increments_model_4(grid) = Increments(omega_synth_model_4,...)
  autocor_model_4(grid) = autocor(omega_synth_model_4)

  


for grid in grids:
  np.savez_compressed("%s_data"%(grad),freq_orig = data(grid)/(2*np.pi+50), freq_model_1 = synth_data_model_4(grid), 
                      freq_model_2 = synth_data_model_2(grid), freq_model_3 = synth_data_model_3(grid), freq_model_4 = synth_data_model_4(grid), 
                      incr_orig = Incr_freq, incr_model_1 = increments_model_1(grid), incr_model_2 = increments_model_2(grid), 
                      incr_model_3 = increments_model_3(grid), incr_model_4 = increments_model_4(grid),  autocor_orig_90min = AUTO_freq(???), 
                      autocor_model_1_90min = autocor_model_1(grid), autocor_model_2_90min = autocor_model_2(grid), 
                      autocor_model_3_90min = autocor_model_3(grid), autocor_model_4_90min = autocor_model_4(grid))
  
  
  
  
