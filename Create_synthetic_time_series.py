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

#freq_orig = data/(2*np.pi+50)
#increments_orig = Increments(data/(2*np.pi+50))

'''For calculations: use angular velocity omega = 2*pi*frequency '''
'''The bandwidth is chosen such that we receive a scmooth distribution'''

'''Choose the grid '''

'''Data analysis of the original time series'''
data_orig          = {i:[]for i in grids}
increments_orig = {i:[]for i in grids}
autocor_orig    = {i:[]for i in grids}

for grid in grids:
  '''Choose the grid '''
  raw=pd.read_csv('Data/Frequency_data_%s.csv'%(grid), sep=',')
  freq = (raw[['Frequency']]/1000 +50).squeeze()
  freq = data_cleaning(freq)
  data = (freq-50)*(2*np.pi)   #Use the angular velocity for the calcualltions

'''Model 1...'''
time_res = 1
synth_data_model_1 = {i:[]for i in grids}
increments_model_1 = {i:[]for i in grids}
autocor_model_1 = {i:[]for i in grids}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
  '''Choose the grid '''
  raw=pd.read_csv('Data/Frequency_data_%s.csv'%(grid), sep=',')
  freq = (raw[['Frequency']]/1000 +50).squeeze()
  freq = data_cleaning(freq)

  data_orig[grid].append(freq)
  increments_orig[grid].append(Increments(freq,time_res = time_res,step = 1))
  autocor_orig[grid].append(autocor(freq_synth_model_1,steps = 10, time_res = time_res))
  
  
  bw_drift = 0.1
  bw_diff = 0.1
  dist_drift = 500
  dist_diff = 500
  
  c_1 = KM_Coeff_1(data,dim= 1,time_res = 1,bandwidth = bw_drift,dist = dist_drift, order = 1)
  epsilon = KM_Coeff_2(data,dim = 1,time_res = 1,bandwidth = bw_diff,dist = dist_diff,multiplicative_noise = False)
  delta_t = 1
  omega_synth_model_1 = Euler_Maruyama(data,c_1=c_1,c_2_decay=0,Delta_P = 0,epsilon=epsilon,time_res = 1,dispatch = 0,delta_t=delta_t,t_final=5,model=1)
  
  freq_synth_model_1 = omega_synth_model_1/(2*np.pi) + 50
  #(data,c_1,c_2,Delta_P,epsilon,c_1_weight = 3,time_res = 1,dispatch = 1,delta_t=0.1,t_final=5,model=3,factor_daily_profile=0)
  synth_data_model_1[grid].append(freq_synth_model_1)
  increments_model_1[grid].append(Increments(freq_synth_model_1,time_res = delta_t,step = 1))
  autocor_model_1[grid].append(autocor(freq_synth_model_1,steps = 10, time_res = delta_t))
  
 
'''Model 2...'''
synth_data_model_2 = {i:[]for i in grids}
increments_model_2 = {i:[]for i in grids}
autocor_model_2 = {i:[]for i in grids}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
    
  '''Choose the grid '''
  raw=pd.read_csv('Data/Frequency_data_%s.csv'%(grid), sep=',')
  freq = (raw[['Frequency']]/1000 +50).squeeze()
  freq = data_cleaning(freq)
  data = (freq-50)*(2*np.pi)   #Use the angular velocity for the calcualltions
    
  trend = 1 #trend is boolean
  bw_drift = 0.1
  bw_diff = 0.1
  dist_drift = 500
  dist_diff = 500
  if grid == 'Balearic':
    Delta_P = power_mismatch(data,avg_for_each_hour = False,dispatch=2,start_minute=0,end_minute=1/6,length_seconds_of_interval=5)
    dispatch = 1
  elif grid == 'Irish':
    Delta_P = power_mismatch(data_filter(data,sigma = 6),avg_for_each_hour = False,dispatch=1,start_minute=0,end_minute=1/6,length_seconds_of_interval=5)
    dispatch = 2
    #we use a filter for the power mismatch of the Iroish data because of regular outliers (every 60 seconds)
  elif grid == 'Iceland':
    Delta_P = 0
    dispatch = 0
    trend = 0 # Represents a no-existing trend as there is no power dispatch schedule
  c_1 = KM_Coeff_1(data - trend*data_filter(data),dim= 1,time_res = 1,bandwidth = bw_drift,dist = dist_drift, order = 1)
  c_2_decay = trend*exp_decay(data,time_res=1,size = 899)
  epsilon =   epsilon = KM_Coeff_2(data - trend*data_filter(data),dim = 1,time_res = 1,bandwidth = bw_diff,dist = dist_diff,multiplicative_noise = False)
  delta_t = 1
  omega_synth_model_2 = Euler_Maruyama(data,c_1=c_1,c_2_decay=c_2_decay,Delta_P = Delta_P,epsilon=epsilon,time_res = 1,dispatch = dispatch,delta_t=delta_t,t_final=5,model=2,factor_daily_profile=0)
  
  #(data,c_1,c_2_decay,Delta_P,epsilon,time_res = 1,dispatch = 1,delta_t=0.1,t_final=5,model=3,factor_daily_profile=0,prim_control_lim = 0.15*2*np.pi,prim_weight = 1)  
  freq_synth_model_2 = omega_synth_model_2/(2*np.pi) + 50

  synth_data_model_2[grid].append(freq_synth_model_2)
  increments_model_2[grid].append(Increments(freq_synth_model_2,time_res = delta_t,step = 1))
  autocor_model_2[grid].append(autocor(freq_synth_model_2,time_res = delta_t))
# 

'''Model 3...'''
synth_data_model_3 = {i:[]for i in grids}
increments_model_3 = {i:[]for i in grids}
autocor_model_3 = {i:[]for i in grids}
'''adapt the parameter estimation to the particulat grids'''
for grid in ['Irish']:#grids:

  '''Choose the grid '''
  raw=pd.read_csv('Data/Frequency_data_%s.csv'%(grid), sep=',')
  freq = (raw[['Frequency']]/1000 +50).squeeze()
  freq = data_cleaning(freq)
  data = (freq-50)*(2*np.pi)   #Use the angular velocity for the calcualltions
  
  trend = 1
  bw_drift = 0.1
  bw_diff = 0.1
  dist_drift = 1200
  if grid == 'Balearic':
    dist_diff = 350
    dispatch = 1
    prim_control_lim, prim_weight = 0.15*2*np.pi, 3
    Delta_P = power_mismatch(data,avg_for_each_hour = True,dispatch=1,start_minute=-2,end_minute=0,length_seconds_of_interval=5)
  elif grid == 'Irish':
    Delta_P = power_mismatch(data_filter(data,sigma = 6),avg_for_each_hour = True,dispatch=2,start_minute=0,end_minute=5,length_seconds_of_interval=5)
    dispatch = 2
    prim_control_lim, prim_weight = 0.13*2*np.pi, 3
    dist = 500
  elif grid == 'Iceland':
    dist_diff = 300
    dispatch = 0
    prim_control_lim, prim_weight = 0, 1 #no additional control via HVDC transmission in the Iceland power grid
    Delta_P = 0
    trend = 0 # Represents a no-existing trend as there is no power dispatch schedule
  c_1 = KM_Coeff_1(data - trend * data_filter(data),dim= 1,time_res = 1,bandwidth = bw_drift,dist = dist_drift, order = 3)
  c_2_decay = trend * exp_decay(data,time_res=1,size = 899)
  epsilon =   epsilon = KM_Coeff_2(data - trend*data_filter(data), dim = 1, time_res = 1, bandwidth = bw_diff, dist = dist_diff, multiplicative_noise = True)
  delta_t = 1
  omega_synth_model_3 = Euler_Maruyama(data,c_1=c_1,c_2_decay=c_2_decay,Delta_P = Delta_P,epsilon=epsilon,time_res = 1,dispatch = dispatch,delta_t=delta_t,t_final=5,model=3,factor_daily_profile=0,prim_control_lim = prim_control_lim, prim_weight = prim_weight)
  freq_synth_model_3 = omega_synth_model_3/(2*np.pi) + 50

  synth_data_model_3[grid].append(freq_synth_model_3)
  increments_model_3[grid].append(Increments(freq_synth_model_3,time_res = delta_t,step = 1))
  autocor_model_3[grid].append(autocor(freq_synth_model_3,time_res = delta_t))
  
   
'''Model 4...'''
'''Model 4...'''
synth_data_model_4 = {i:[]for i in grids}
increments_model_4 = {i:[]for i in grids}
autocor_model_4 = {i:[]for i in grids}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
  
  '''Choose the grid '''
  raw=pd.read_csv('Data/Frequency_data_%s.csv'%(grid), sep=',')
  freq = (raw[['Frequency']]/1000 +50).squeeze()
  freq = data_cleaning(freq)
  data = (freq-50)*(2*np.pi)   #Use the angular velocity for the calcualltions
  
  trend = 1
  bw_drift, bw_diff = 0.05, 0.05
  if grid == 'Balearic':
    dist_theta, dist_omega = 20,20
    prim_control_lim, prim_weight = 0.13*2*np.pi, 3
    factor_daily_profile = 2.5 
  elif grid == 'Irish':
    dist_theta, dist_omega = 15,15
    prim_control_lim, prim_weight = 0.13*2*np.pi, 3
    factor_daily_profile = 3.5 
  elif grid == 'Iceland':
    dist_theta, dist_omega = 30,70    # choose larger intervals as the deviations in the grid are larger
    prim_control_lim, prim_weight = 0, 1
    factor_daily_profile = 0
    trend = 0
  c_1 = KM_Coeff_1(data - trend*data_filter(data),dim= 2,time_res = 1,bandwidth = bw_drift,dist = [dist_theta, dist_omega], order = 1)[0]
  c_2 = KM_Coeff_1(data - trend*data_filter(data),dim= 2,time_res = 1,bandwidth = bw_drift,dist = [dist_theta, dist_omega], order = 1)[1]
  Delta_P = 0 # Use multple ofdaily profile for describing the trend
  '''Rename dist_drift and dist_diff here!'''
  epsilon =   epsilon =  KM_Coeff_2(data - trend*data_filter(data), dim = 2, time_res = 1, bandwidth = bw_diff, dist = [dist_theta, dist_omega], multiplicative_noise = True)
  delta_t = 1
  omega_synth_model_4 = Euler_Maruyama(data,c_1=c_1,c_2_decay=c_2,Delta_P = Delta_P,epsilon=epsilon,time_res = 1,dispatch = 0,delta_t=delta_t,t_final=5,model=4,factor_daily_profile=factor_daily_profile,prim_control_lim = prim_control_lim, prim_weight = prim_weight)

  freq_synth_model_4 = omega_synth_model_4/(2*np.pi) + 50

  synth_data_model_4[grid].append(freq_synth_model_4)
  increments_model_4[grid].append(Increments(freq_synth_model_4,time_res = delta_t,step = 1))
  autocor_model_4[grid].append(autocor(freq_synth_model_4,time_res = delta_t))
  


for grid in grids:
  np.savez_compressed("%s_data"%(grid),freq_orig = data_orig[grid]/(2*np.pi+50), freq_model_1 = synth_data_model_4[grid], 
                      freq_model_2 = synth_data_model_2[grid], freq_model_3 = synth_data_model_3[grid], freq_model_4 = synth_data_model_4[grid], 
                      incr_orig = increments_orig[grid], incr_model_1 = increments_model_1[grid], incr_model_2 = increments_model_2[grid], 
                      incr_model_3 = increments_model_3[grid], incr_model_4 = increments_model_4[grid],  autocor_orig_90min = autocor_orig[grid], 
                      autocor_model_1_90min = autocor_model_1[grid], autocor_model_2_90min = autocor_model_2[grid], 
                      autocor_model_3_90min = autocor_model_3[grid], autocor_model_4_90min = autocor_model_4[grid])


