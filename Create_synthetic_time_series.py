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

freq_orig = data/(2*np.pi+50)
increments_orig = Increments(data/(2*np.pi+50))

'''For calculations: use angular velocity omega = 2*pi*frequency '''
'''The bandwidth is chosen such that we receive a scmooth distribution'''

'''Choose the grid '''
raw=pd.read_csv('Data/Frequency_data_Balearic.csv', sep=',')
freq = raw[['Frequency']]/1000 +50
print(freq)
data = (freq-50)/(2*np.pi)
print(data)


'''Model 1...'''
synth_data_model_1 = {}
increments_model_1 = {}
autocor_model_1 = {}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
  data = ...
  data = data_cleaning(data,grid)
  bw_drift = 0.1
  bw_diff = 0.1
  dist_drift = 500
  dist_diff = 500
  
  c_1 = KM_Coeff_1(data,dim= 1,time_res = 1,bandwidth = bw_drift,dist = dist_drift, order = 1)
  epsilon = KM_Coeff_2(data,dim = 1,time_res = 1,bandwidth = bw_diff,dist = dist_diff,multiplicative_noise = False)
  delta_t = 0.1
  omega_synth_model_1 = Euler_Maruyama(data,delta_t=0.1,t_final=5,model=1,c_1,c_2=0,Delta_P=0,epsilon,factor_daily_profile=0)
  
  synth_data_model_1(grid) = omega_synth_model_1
  increments_model_1(grid) = Increments(omega_synth_model_1,time_res = delta_t,step_seconds = 1)
  autocor_model_1(grid) = autocor(omega_synth_model_1,time_res = delta_t)
  
  '''Model 2...'''
synth_data_model_2 = {}
increments_model_2 = {}
autocor_model_2 = {}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
  trend = 1 #trend is boolean
  data = ...
  data = data_cleaning(data,grid)
  bw_drift = 0.1
  bw_diff = 0.1
  dist_drift = 500
  dist_diff = 500
  if grid == 'Balearic':
    Delta_P = power_mismatch(data,avg_for_each_hour = False,dispatch=2,start_minute=0,end_minute=1/6,length_seconds_of_interval=5):
  elif grid == 'Irish':
    Delta_P = power_mismatch(data_filter(data,sigma = 6),avg_for_each_hour = False,dispatch=2,start_minute=0,end_minute=1/6,length_seconds_of_interval=5):
    #we use a filter for the power mismatch of the Iroish data because of regular outliers (every 60 seconds)
  elif grid == 'Irish':
    Delta_P = 0
    trend = 0 # Represents a no-existing trend as there is no power dispatch schedule
  c_1 = KM_Coeff_1(data - trend*data_filter(data),dim= 1,time_res = 1,bandwidth = bw_drift,dist = dist_drift, order = 1)
  c_2 = trend*exp_decay(data,time_res=1,size = 899)
  epsilon =   epsilon = KM_Coeff_2(data - trend*data_filter(data),dim = 1,time_res = 1,bandwidth = bw_diff,dist = dist_diff,multiplicative_noise = False)
  delta_t = 0.1
  omega_synth_model_2 = Euler_Maruyama(data,delta_t=delta_t,t_final=5,model=2,c_1,c_2,Delta_P,epsilon,factor_daily_profile=0)
  
  synth_data_model_2(grid) = omega_synth_model_2
  increments_model_2(grid) = Increments(omega_synth_model2,time_res = delta_t,step_seconds = 1)
  autocor_model_2(grid) = autocor(omega_synth_model_2,time_res = delta_t)

   '''Model 3...'''
synth_data_model_3 = {}
increments_model_3 = {}
autocor_model_3 = {}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
  trend = 1
  data = ...
  data = data_cleaning(data,grid)
  bw_drift = 0.1
  bw_diff = 0.1
  dist_drift = 1200
  if grid = 'Balearic':
    dist_diff = 350
    Delta_P = power_mismatch(data,avg_for_each_hour = True,dispatch=1,start_minute=-2,end_minute=0,length_seconds_of_interval=5)
  elif grid = 'Irish':
    Delta_P = power_mismatch(data,avg_for_each_hour = True,dispatch=1,start_minute=-2,end_minute=0,length_seconds_of_interval=5)
    dist = 500
  elif grid = 'Iceland':
    dist_diff = 300
    Delta_P = 0
    trend = 0 # Represents a no-existing trend as there is no power dispatch schedule
  c_1 = KM_Coeff_1(data - trend * data_filter(data),dim= 1,time_res = 1,bandwidth = bw_drift,dist = dist_drift, order = 3)
  c_2 = trend * exp_decay(data,time_res=1,size = 899)
  epsilon =   epsilon = KM_Coeff_2(data - trend*data_filter(data), dim = 1, time_res = 1, bandwidth = bw_diff, dist = dist_diff, multiplicative_noise = True)
  delta_t = 0.1
  omega_synth_model_3 = Euler_Maruyama(data,delta_t=delta_t,t_final=5,model=3,c_1,c_2,Delta_P,epsilon,factor_daily_profile=0)
  
  synth_data_model_3(grid) = omega_synth_model_3
  increments_model_3(grid) = Increments(omega_synth_model_3,time_res = delta_t,step_seconds = 1)
  autocor_model_3(grid) = autocor(omega_synth_model_3,time_res = delta_t)
  
   '''Model 4...'''
synth_data_model_4 = {}
increments_model_4 = {}
autocor_model_4 = {}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
  trend = 1
  data = ...
  data = data_cleaning(data,grid)
  bw_drift, bw_diff = 0.05, 0.05
  if grid = 'Balearic':
    dist_drift, dist_diff = 20,20
    factor_daily_profile = 2.5 
  elif grid = 'Irish':
    dist_drift, dist_diff = 15,15
    factor_daily_profile = 3.5 
  elif grid = 'Iceland':
    dist_drift, dist_diff = 20,20
    factor_daily_profile = 0
    trend = 0
  c_1 = KM_Coeff_1(data - trend*data_filter(data),dim= 2,time_res = 1,bandwidth = bw_drift,dist = dist_drift, order = 1)[0]
  c_2 = KM_Coeff_1(data - trend*data_filter(data),dim= 2,time_res = 1,bandwidth = bw_drift,dist = dist_drift, order = 1)[1]
  Delta_P = 0 # Use multple ofdaily profile for describing the trend
  epsilon =   epsilon =  KM_Coeff_2(data - trend*data_filter(data), dim = 2, time_res = 1, bandwidth = bw_diff, dist = dist_diff, multiplicative_noise = True)
  delta_t = 0.1
  omega_synth_model_4 = Euler_Maruyama(data,delta_t=delta_t,t_final=5,model=4,c_1,c_2,Delta_P,epsilon,factor_daily_profile)
  
  synth_data_model_4(grid) = omega_synth_model_4
  increments_model_1(grid) = Increments(omega_synth_model_1,time_res = delta_t,step_seconds = 1)
  autocor_model_1(grid) = autocor(omega_synth_model_1,time_res = delta_t)
  


for grid in grids:
  np.savez_compressed("%s_data"%(grad),freq_orig = data(grid)/(2*np.pi+50), freq_model_1 = synth_data_model_4(grid), 
                      freq_model_2 = synth_data_model_2(grid), freq_model_3 = synth_data_model_3(grid), freq_model_4 = synth_data_model_4(grid), 
                      incr_orig = Incr_freq, incr_model_1 = increments_model_1(grid), incr_model_2 = increments_model_2(grid), 
                      incr_model_3 = increments_model_3(grid), incr_model_4 = increments_model_4(grid),  autocor_orig_90min = AUTO_freq(???), 
                      autocor_model_1_90min = autocor_model_1(grid), autocor_model_2_90min = autocor_model_2(grid), 
                      autocor_model_3_90min = autocor_model_3(grid), autocor_model_4_90min = autocor_model_4(grid))
  
  
  
  
