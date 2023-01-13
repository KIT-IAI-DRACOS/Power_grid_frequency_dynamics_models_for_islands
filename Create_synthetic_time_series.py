#!pip install kramersmoyal
import pandas as pd
import numpy as np
from kramersmoyal import km

import os
   

from Data_cleaning import data_cleaning
from Functions import data_filter, integrate_omega, KM_Coeff_1, KM_Coeff_2, daily_profile, power_mismatch, exp_decay, Euler_Maruyama, Increments, autocor 

#grids = ['Iceland','Irish','Balearic']    
grids = ['Balearic']  
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


edges_1d     = {i:[]for i in grids}
drift_1d     = {i:[]for i in grids}
diffusion_1d = {i:[]for i in grids}
edges_2d     = {i:[]for i in grids}
kmc_2d       = {i:[]for i in grids}

for grid in grids:
  time_res = 1
  '''Choose the grid '''
  raw=pd.read_csv('./Data/Frequency_data_%s.csv'%(grid), sep=',')
  freq = (raw[['Frequency']]/1000 +50).squeeze()
  freq = data_cleaning(freq)

  
  data_orig[grid].append(freq)
  increments_orig[grid].append(Increments(freq,time_res = time_res,step = 1))
  autocor_orig[grid].append(autocor(freq,steps = 10, time_res = time_res))
  
  
'''Model 1...'''
synth_data_model_1 = {i:[]for i in grids}
increments_model_1 = {i:[]for i in grids}
autocor_model_1 = {i:[]for i in grids}
c_1_model1 = {i:[]for i in grids}
epsilon_model1 = {i:[]for i in grids}
'''adapt the parameter estimation to the particulat grids'''

for grid in grids:
  raw=pd.read_csv('./Data/Frequency_data_%s.csv'%(grid), sep=',')
  freq = (raw[['Frequency']]/1000 +50).squeeze()
  freq = data_cleaning(freq)
  data = (freq-50)*(2*np.pi)   #Use the angular velocity for the calcualltions

  
  bw_drift = 0.1
  bw_diff = 0.1
  dist_drift = 800 #for small amount of data choose a larger value for dist_drift
  dist_diff = 500
  
  c_1 = KM_Coeff_1(data,dim= 1,time_res = 1,bandwidth = bw_drift,dist = dist_drift, order = 1)
  

  epsilon = KM_Coeff_2(data,dim = 1,time_res = 1,bandwidth = bw_diff,dist = dist_diff,multiplicative_noise = False)
  
  c_1_model1[grid] = c_1
  epsilon_model1[grid] = epsilon
  
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
c_1_model2 = {i:[]for i in grids}
epsilon_model2 = {i:[]for i in grids}

'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
    
  raw=pd.read_csv('./Data/Frequency_data_%s.csv'%(grid), sep=',')
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
  
  kmc,edges = km(data - trend * data_filter(data),powers = [0,1,2],bins = np.array([6000]),bw=bw_drift)
  edges_1d[grid] = edges[0]
  drift_1d[grid] = kmc[1]
  diffusion_1d[grid] = kmc[2] 
  c_1_model2[grid] = c_1
  epsilon_model2[grid] = epsilon

  delta_t = 1
  omega_synth_model_2 = Euler_Maruyama(data,c_1=c_1,c_2_decay=c_2_decay,Delta_P = Delta_P,epsilon=epsilon,time_res = 1,dispatch = dispatch,delta_t=delta_t,t_final=5,model=2,factor_daily_profile=0)
  
  #(data,c_1,c_2_decay,Delta_P,epsilon,time_res = 1,dispatch = 1,delta_t=0.1,t_final=5,model=3,factor_daily_profile=0,prim_control_lim = 0.15*2*np.pi,prim_weight = 1)  
  freq_synth_model_2 = omega_synth_model_2/(2*np.pi) + 50

  synth_data_model_2[grid].append(freq_synth_model_2)
  increments_model_2[grid].append(Increments(freq_synth_model_2,time_res = delta_t,step = 1))
  autocor_model_2[grid].append(autocor(freq_synth_model_2,time_res = delta_t))

'''Model 3...'''
synth_data_model_3 = {i:[]for i in grids}
increments_model_3 = {i:[]for i in grids}
autocor_model_3 = {i:[]for i in grids}
p_3_model3, p_1_model3 = {i:[]for i in grids},{i:[]for i in grids}
d_2_model3, d_0_model3 = {i:[]for i in grids},{i:[]for i in grids}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:

  raw=pd.read_csv('./Data/Frequency_data_%s.csv'%(grid), sep=',')
  freq = (raw[['Frequency']]/1000 +50).squeeze()
  freq = data_cleaning(freq)
  data = (freq-50)*(2*np.pi)   #Use the angular velocity for the calcualltions
  
  trend = 1
  bw_drift = 0.1
  bw_diff = 0.1
  dist_drift = 1200
  if grid == 'Balearic':
    dist_drift = 1600
    dist_diff = 350
    dispatch = 1
    prim_control_lim, prim_weight = 0.15*2*np.pi, 10 #3
    Delta_P = power_mismatch(data,avg_for_each_hour = True,dispatch=1,start_minute=-2,end_minute=0,length_seconds_of_interval=5)
  elif grid == 'Irish':
    Delta_P = power_mismatch(data_filter(data,sigma = 6),avg_for_each_hour = True,dispatch=2,start_minute=0,end_minute=5,length_seconds_of_interval=5)
    dispatch = 2
    prim_control_lim, prim_weight = 0.13*2*np.pi, 3
    dist_diff = 700 #500
  elif grid == 'Iceland':
    dist_diff = 300
    dispatch = 0
    prim_control_lim, prim_weight = 0, 1 #no additional control via HVDC transmission in the Iceland power grid
    Delta_P = 0
    trend = 0 # Represents a no-existing trend as there is no power dispatch schedule
  c_1 = KM_Coeff_1(data - trend * data_filter(data),dim= 1,time_res = 1,bandwidth = bw_drift,dist = dist_drift, order = 3)
  c_2_decay = trend * exp_decay(data,time_res=1,size = 899)
  epsilon = KM_Coeff_2(data - trend*data_filter(data), dim = 1, time_res = 1, bandwidth = bw_diff, dist = dist_diff, multiplicative_noise = True)
  
  (p_3_model3[grid], p_1_model3[grid]) = c_1 
  (d_2_model3[grid], d_0_model3[grid]) = epsilon

  
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
  raw=pd.read_csv('./Data/Frequency_data_%s.csv'%(grid), sep=',')
  freq = (raw[['Frequency']]/1000 +50).squeeze()
  freq = data_cleaning(freq)
  data = (freq-50)*(2*np.pi)   #Use the angular velocity for the calcualltions
  
  trend = 1
  bw_drift, bw_diff = 0.05, 0.05
  if grid == 'Balearic':
    dist_theta, dist_omega = 20,20
    prim_control_lim, prim_weight = 0.13*2*np.pi, 3
    factor_daily_profile = 1#2.5 
  elif grid == 'Irish':
    dist_theta, dist_omega = 15,15
    prim_control_lim, prim_weight = 0.13*2*np.pi, 3
    factor_daily_profile = 3.2
  elif grid == 'Iceland':
    dist_theta, dist_omega = 30,70    # choose larger intervals as the deviations in the grid are larger
    prim_control_lim, prim_weight = 0, 1
    factor_daily_profile = 0
    trend = 1 #as we calculate witha 1-second resolution, we use also for Iceland a Gaussian filter (time window 60 seconds) as the 1-seconds resolution is too rough for the integration of the angular velocity 
  c_1 = KM_Coeff_1((data - trend*data_filter(data)),dim= 2,time_res = 1,bandwidth = bw_drift,dist = [dist_theta, dist_omega], order = 1)[0]
  c_2 = KM_Coeff_1((data - trend*data_filter(data)),dim= 2,time_res = 1,bandwidth = bw_drift,dist = [dist_theta, dist_omega], order = 1)[1]
  Delta_P = 0 
 
  epsilon =   KM_Coeff_2(data - trend*data_filter(data), dim = 2, time_res = 1, bandwidth = bw_diff, dist = [dist_theta, dist_omega], multiplicative_noise = True)
  delta_t = 1
  omega_synth_model_4 = Euler_Maruyama(data,c_1=c_1,c_2_decay=c_2,Delta_P = Delta_P,epsilon=epsilon,time_res = 1,dispatch = 0,delta_t=delta_t,t_final=5,model=4,factor_daily_profile=factor_daily_profile,prim_control_lim = prim_control_lim, prim_weight = prim_weight)

  freq_synth_model_4 = omega_synth_model_4/(2*np.pi) + 50

  synth_data_model_4[grid].append(freq_synth_model_4)
  increments_model_4[grid].append(Increments(freq_synth_model_4,time_res = delta_t,step = 1))
  autocor_model_4[grid].append(autocor(freq_synth_model_4,time_res = delta_t))
  
  powers = np.array([[0,0],[1,0],[0,1],[1,1],[2,0],[0,2],[2,2]])
  bins = np.array([300,300])
  data_2d = np.array([integrate_omega(data - trend * data_filter(data),time_res=time_res,start_value = 0),data - trend * data_filter(data)]) #use theta as integrated omega
  kmc_2d[grid], edges_2d[grid] = km(data_2d.transpose(),powers = powers,bins = bins,bw=0.05)

for grid in grids:

    file_kmc  = str(os.getcwd().replace(os.sep, '/')) + '/Create_figures/%s_kmc'%(grid)   #data for figure 2
    file_data = str(os.getcwd().replace(os.sep, '/')) + '/Create_figures/%s_data'%(grid)  #data for figures 3/4
    
    

    
    np.savez_compressed(file_data,freq_origin = np.asarray(data_orig[grid]), freq_model1 = np.asarray(synth_data_model_1[grid]), 
                      freq_model2 = np.asarray(synth_data_model_2[grid]), freq_model3 = np.asarray(synth_data_model_3[grid]), freq_model4 = np.asarray(synth_data_model_4[grid]), 
                      incr_origin = np.asarray(increments_orig[grid]), incr_model1 = np.asarray(increments_model_1[grid]), incr_model2 = np.asarray(increments_model_2[grid]), 
                      incr_model3 = np.asarray(increments_model_3[grid]), incr_model4 = np.asarray(increments_model_4[grid]),  
                      auto_origin = np.asarray(autocor_orig[grid]), auto_model1 = np.asarray(autocor_model_1[grid]), auto_model2 = np.asarray(autocor_model_2[grid]), 
                      auto_model3 = np.asarray(autocor_model_3[grid]), auto_model4 = np.asarray(autocor_model_4[grid]))


    np.savez_compressed(file_kmc,
        edges_1d = edges_1d[grid],
        drift_1d = drift_1d[grid],
        diffusion_1d = diffusion_1d[grid],
        c_1_model1 = c_1_model1[grid],
        c_1_model2 = c_1_model2[grid],
        p_1_model3 = p_1_model3[grid],
        p_3_model3 = p_3_model3[grid],
        epsilon_model1 = epsilon_model1[grid],
        epsilon_model2 = epsilon_model2[grid],
        d_0_model3 = d_0_model3[grid],
        d_2_model3 = d_2_model3[grid],
        edges_2d = edges_2d[grid],
        kmc_2d = kmc_2d[grid])
