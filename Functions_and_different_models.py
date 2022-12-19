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

'''For calculations: use angular velocity omega = 2*pi*frequency '''



'''Necessary functions: De-trending, Estimation of KM-Coefficients, Calculation of Power mismatch, Calculation of c_2'''
'''Different models: 1: linear c_1 and epsilon, no c_2 and Delta_P
                     2: linear c_1,c_2 and epsilon,
                     3: cubic c_1(omega), state-dependent c_2 and epsilon
                     4: Calculation of c_2 from bivariate Fokker-Planck equation, linear c_1 and c_2, state-dependent noise   '''


data = (freq+50)/(2*np.pi)

'''Detrending of the time series:'''

def filter(data,sigma = 60):
  datafilter = gaussian_filter1d(data,sigma=sigma)
  return datafilter

datafilter = filter(data,sigma = 60)
data_detrended = data - datafilter

'''Integration of the angular velocity for calcualtion of theta (voltage angle)'''
'''Integrate the omegas by using a sum'''

def integrate_omega(data,time_res=1,start_value = 0):
  #def theta_int_unscaled(t):
  #s = 0.1 * np.sum(data_100ms[start_int:t])
  #return s
  theta = np.zeros(data.size)
  theta[start_value] = data[start_value]
  for i in range(start_value + 1,data.size - start_value):
      theta[i] = theta[i-1] + 1*data[i]
  '''scale values of theta by substracting the average'''
  theta = theta-np.mean(theta)
  return theta


'''1.Calculation the noise amplitude'''

'''here: Use angular velocity (omega) instead of frequency(f)'''
def KM_Coeff_1 (data,dimension = 1,bandwidth,interval)
  if dimension = 1:
    powers = [0,1,2]
    bins = np.array([6000])
  '''dimension signifies the usage of the univariate or the bivariate Fokker-Planck equation''' 
  elif dimension = 2:
    
  kmc, edges = km(data,powers = powers,bins = bins,bw=bandwidth)
  '''Attention: use here as data the detrended data (doriginal data minus filter'''
  zero_frequency = np.argmin(space[0]**2)
  #dist = 500
  '''old model: constant epsilon'''
  epsilon = np.sqrt(np.mean(kmc[2,zero_frequency-dist:zero_frequency+dist])*2)
  #epsilon = np.sqrt((diffusion[1,zero_frequency])*2)

  peak = zero_frequency-500+np.argmin(kmc[2,zero_frequency-500:zero_frequency+500])
  np.argmin(kmc[2,zero_frequency-500:zero_frequency+500]**2),zero_frequency,zero_frequency-500+np.argmin(kmc[2,zero_frequency-500:zero_frequency+500])
  peak

  dist = 350
  '''Try density-dependent epsilon:'''
  '''new model (resp. one new model): density(frequency)-dependent noise'''
  #d_2 = curve_fit(lambda t , a  : a*(t-0)**2 + kmc[2,zero_frequency],space[ 0 ][zero_frequency-dist:zero_frequency+dist],kmc[2,zero_frequency-dist:zero_frequency+dist])[0]    
  d_2 = curve_fit(lambda t , a  : a*(t-0)**2 + kmc[2,zero_frequency],space[ 0 ][zero_frequency-dist:zero_frequency+dist],kmc[2,peak-dist:peak+dist])[0]    

  # #e_4,e_3,e_2,e_1,e_0
  #diff_zero=kmc[2,zero_frequency]
  diff_zero=kmc[2,peak]

  d_0=diff_zero
  # def e(x):
  #     eps = np.sqrt(2*(d*x**34 + diff_0))
  #     return eps



  '''Simple (stupid) model: Ornstein-Uhlenbeck process:'''
  kmc_simple, edges_2_simple = km(data,powers = [0,1,2],bins = np.array([6000]),bw=bw_diff)
  space_simple = edges_2_simple
  #np.shape(diffusion_simple)
  np.shape(space_simple)
  zero_frequency_simple = np.argmin(space_simple[0]**2)
  #zero_frequency_simple = np.argmin(diffusion_simple[1])#[1500:-1500])+1500
  peak_simple = space_simple[0][zero_frequency_simple]

  '''old model: constant epsilon'''
  dist_simple = 500
  epsilon_simple = np.sqrt(np.mean(kmc_simple[2,zero_frequency_simple-dist_simple:zero_frequency_simple+dist_simple])*2)
  #epsilon_simple = np.sqrt((diffusion_simple[1,zero_frequency_simple])*2)

  epsilon,d_2,d_0,epsilon_simple


  
  
  
  

'''2. Estimation of the drift (primary control)'''
def KM_Coeff_1(data,):
  def KM_Coeff_1 (data,sigma=60,,dimension = 1,bandwidth,interval)
  if dimension = 1:
    powers = [0,1,2]
    bins = np.array([6000])
  '''dimension signifies the usage of the univariate or the bivariate Fokker-Planck equation''' 
  elif dimension = 2:
    
    
  kmc,edges = km(data,powers = powers,bins = bins,bw=bandwidth)
  '''Attention: use here as data the detrended data (doriginal data minus filter'''
  mid_point = np.argmin(edges[0]**2)
  #D = 1300
  D = 300
  D_lin=500
  '''old model: constant c_1'''
  c_1 = curve_fit(lambda t,a,b: a - b*t , xdata = space[ 0 ][mid_point-D_lin:mid_point+D_lin] ,ydata = kmc[1][ mid_point - D_lin: mid_point + D_lin] , p0 = ( 0.0002 ,0.0005 ) ,maxfev=10000)[ 0 ][ 1 ]

'''new model (resp. one new model): polynomial drift'''
p_3,p_2,p_1,p_0=np.polyfit(space[ 0 ][mid_point-D:mid_point+D] , kmc[1][ mid_point - D: mid_point + D] ,3)

#np.shape(drift)
#c_3,c_2,c_1,c_0
#c_15,c_14,c_13,c_12,c_11,c_10
#c_13,c_12,c_11,c_10
#c_1=np.abs(p_1)
#print('c_1: %f'%c_1)

'''Simple (stupid) model: Ornstein-Uhlenbeck process:'''
kmc_simple, edges_1_simple = km(data,powers = [0,1,2],bins = np.array([6000]),bw=bw_drift)#0.05)##bw_drift)
#drift, edges_1 = km(data_diff,powers = [0,1],bins = np.array([6000]),bw=bw_drift)

space_simple = edges_1_simple
mid_point_simple = np.argmin(space_simple[0]**2)

D_lin=500
'''old model: constant c_1'''
c_1_simple = curve_fit(lambda t,a,b: a - b*t , xdata = space_simple[ 0 ][mid_point_simple-D_lin:mid_point_simple+D_lin] ,ydata = kmc_simple[1][ mid_point_simple - D_lin: mid_point_simple + D_lin] , p0 = ( 0.0002 ,0.0005 ) ,maxfev=10000)[ 0 ][ 1 ]



print(p_3,p_2,p_1,p_0)
print(c_1)
print(c_1_simple)
#q_7,q_6,q_5,q_4,q_3,q_2,q_1,q_0
print(q_5,q_4,q_3,q_2,q_1,q_0)




'''Calculate daily profiles:'''
def daily_profile(data,time_res = 1):
  '''time_res represents the time resolution of the data'''
  daily_profile=np.zeros(24*3600*time_res)
  day_number = data.size//(24*3600time_res)
  for i in range(daily_prof.size):
    daily_profile[i] = np.mean([data[i+3600*24*j]for j in range(day_number)])
  return daily_profile

'''3. Calculation of the power mismatch'''
'''Calculate the power mismatch as the derivative of the trajectories around times of power dispatches:'''

'''Delta_P: Find ROCOF in (5-minutes-interval around full hours (resp. power injections)!)'''

power_mismatch(data,avg_for_each_hour = 'True',dispatch=2,start_minute=0,end_minute=7,length_seconds_of_interval=5):
  '''Attention: Use the data that you want to use for the power mismatch (for ex. for Ireland we take 5-second filtered data because of hourly and 60-seconds-junks)'''
  data_range = data.size//(3600*24)
  s,e,l = start_minute,end_minute,length_seconds_of_interval
  end,steps = 2*length_seconds_of_interval, end + 1
  #m = np.zeros((24*change,data_range))
  argm = np.zeros((24*dispatch,data_range))
  Delta_P_slopes = np.zeros((24*dispatch,data_range))
  for i in range(24*dispatch):
      for j in range(0,data_range-1):
          argm[i,j] = np.argmax(np.abs([curve_fit( lambda t , a , b : a + b*t , np.linspace( 0 , end , steps ) , data[ i*int(3600/dispatch)+ 3600 * 24 * j  -s*60 + k*l -l : i*int(3600/dispatch)+ 3600 * 24 * j -s*60 + k*l + l] , p0 = ( 0.0 , 0.0 ) ,maxfev=10000)[ 0 ][ 1 ] for k in range(1,int((s+e)*60/l))]))
          Delta_P_slopes[i,j] =  (curve_fit( lambda t , a , b : a + b*t , np.linspace( 0 , end , steps ) , data[ i*int(3600/dispatch)+ 3600 * 24 * j  -s*60 + int(argm[i,j]+1)*l -l : i*int(3600/dispatch)+ 3600 * 24 * j -s*60 +  int(argm[i,j]+1)*l + l] , p0 = ( 0.0 , 0.0 ) ,maxfev=10000)[ 0 ][ 1 ] )

   sign = np.zeros(change*24)
   daily_profile = daily_profile(data)
   daily_prof_25 = np.zeros(25*3600*time_res)   #add 1st hour of daily profile to the daily prile to calculate the average sign of the slope at each power dispatch
   daily_prof_25[0:24*3600*time_res] = daily_profile
   daily_prof_25[24*3600*time_res:] = daily_profile[0:1*24*3600*time_res]
   for i in range(sign.size):
       if np.mean(np.diff(daily_prof_25[(i+1)*(int(4/change))*900 -int(s*60) : (i+1)*(int(4/change))*900 + int(e*60)])) > 0:
        sign[(i+1)%24]=1
       else:
        sign[(i+1)%24]=-1
  P_arr = np.zeros(24*change)
  for i in range(24*change):
      P_arr[i] = np.mean(np.abs(Delta_P_slopes[i,:]))
  if avg_for_each_hour = 'True':
    Delta_P = sign*P_arr
  else:
    Delta_P = np.mean(np.abs(Delta_P_slopes[i,:]))
  
  
  
  '''Calculation of c_2 from the exponential decay'''
'''4. STATE-DEPENDENT Secondary control c_2 /( EXPERIMENTATION)!!!'''



def exp_decay(data,time_res,size = 899):
  #gap   =0
  size  = 899
  steps = size+1

  window = 3600
  data_range = data.size // window

  c_2_decays = np.zeros(data_range)

  for j in range(1,data_range) :
      # if the frequency trajectory increases positively
      if np.sum(( np.diff( data[ 3600 *( j ) : 3600 * ( j ) +10]) ) ) > 0 :
          c_2_decays[j] = curve_fit(lambda t , a , b :#,  c : 
          a*np.exp(-b*t ) ,#* (1-np.exp(-c * t+2*b* t ) ) ,
          np.linspace( 0 , size ,steps ) , data[ 3600 * ( j ) : 3600 * ( j ) + steps] ,
          p0 = ( 0.08 , 0.00455 
               ),#, 0.035  ) , 
          maxfev=10000)[ 0 ][ 1 ]
      elif np.sum(( np.diff( data[ 3600 *( j ) : 3600 * ( j ) +10]) ) ) <= 0 :
          c_2_decays[j] = curve_fit(lambda t , a , b :#,,  c : 
          (-a)*np.exp(-b*t ) ,#* (1-np.exp(-c * t+2*b* t ) ) ,
          np.linspace( 0 , size ,steps ) , data[ 3600 * ( j ) -gap: 3600 * ( j )-gap + steps] ,
          p0 = ( 0.08 , 0.00455 
               ),#, 0.035  ) , 
          maxfev=10000)[ 0 ][ 1 ]
    temp_c_2_decays = c_2_decays[np.argsort(c_2_decays)][: - c_2_decays.size // 5]
    return np.mean(temp_c_2_decays )
  # c_2_linear = np.mean(temp_c_2_decays ) * (c_1)
  #omega_arr = np.linspace(-0.5,0.5,101)
  #c_2_arr = np.mean(temp_c_2_decays)*(3*(-p_3)*omega_arr**2 - p_1)




'''Euler-Maruyama'''

def Euler_Maruyama(data,delta_t,t_final):
  t_steps = int(t_final/delta_t)
  
  
  
  
  
  
  

