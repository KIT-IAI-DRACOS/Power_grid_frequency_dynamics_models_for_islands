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

'''1.Calculation the noise amplitude'''

'''Fokker-Planck-Model-Test'''
'''here: Use angular velocity (omega) instead of frequency(f)'''

'''1. Noise epsilon''' 

#def KM_Coeff_1 (data):
datafilter = gaussian_filter1d(data, sigma = 1*60)
kmc, edges_2 = km(data-datafilter,powers = [0,1,2],bins = np.array([6000]),bw=bw_diff)
#kmc, edges_2 = km(data,powers = [0,1,2],bins = np.array([6000]),bw=bw_diff)
edges = edges_2
#diffusion = kmc[1,:]
space = edges_2
#np.shape(diffusion)
#np.shape(space)
zero_frequency = np.argmin(space[0]**2)
dist = 500
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



'''2. Estimation of the drift (primary control)



'''3. Calculation of the power mismatch'''
'''Calculate the power mismatch as the derivative of the trajectories around times of power dispatches:'''

'''Delta_P: Find ROCOF in (5-minutes-interval around full hours (resp. power injections)!)'''

power_mismatch(data,dispatch=2,start=0,end=7,length=5):
  data_range = data.size//(3600*24)
  s,e,l = start,end,length
  #steps , end = 5+5,steps+1
  #l=5

  #m = np.zeros((24*change,data_range))
  argm = np.zeros((24*change,data_range))
  Delta_P_slopes = np.zeros((24*change,data_range))
  test = np.zeros((24*change,data_range))
  for i in range(24*change):
      for j in range(0,data_range-1):
           #m[i,j] = np.argmax(np.diff([data[i*int(3600/change)+j*3600*24-s*60:i*int(3600/change)+j*3600*24+e*60]]))
          #Delta_P_slopes[i, j ] = curve_fit ( lambda t , a , b : a + b*t , np.linspace( 0 , end , steps ) , data[ i*int(3600/change)+ 3600 * 24 * j  -s*60 + int(m[i,j]) -5 : i*int(3600/change)+ 3600 * 24 * j -s*60 + int(m[i,j]) + 5] , p0 = ( 0.0 , 0.0 ) ,maxfev=10000)[ 0 ][ 1 ]
          argm[i,j] = np.argmax(np.abs([curve_fit ( lambda t , a , b : a + b*t , np.linspace( 0 , end , steps ) , filter_5[ i*int(3600/dispatch)+ 3600 * 24 * j  -s*60 + k*l -l : i*int(3600/change)+ 3600 * 24 * j -s*60 + k*l + l] , p0 = ( 0.0 , 0.0 ) ,maxfev=10000)[ 0 ][ 1 ] for k in range(1,int((s+e)*60/l))]))
          Delta_P_slopes[i,j] =  (curve_fit ( lambda t , a , b : a + b*t , np.linspace( 0 , end , steps ) , filter_5[ i*int(3600/change)+ 3600 * 24 * j  -s*60 + int(argm[i,j]+1)*l -l : i*int(3600/change)+ 3600 * 24 * j -s*60 +  int(argm[i,j]+1)*l + l] , p0 = ( 0.0 , 0.0 ) ,maxfev=10000)[ 0 ][ 1 ] )
          #test[i,j] = np.max(np.abs([curve_fit ( lambda t , a , b : a + b*t , np.linspace( 0 , end , steps ) , data[ i*int(3600/change)+ 3600 * 24 * j  -s*60 + k*5 -5 : i*int(3600/change)+ 3600 * 24 * j -s*60 + k*5 + 5] , p0 = ( 0.0 , 0.0 ) ,maxfev=10000)[ 0 ][ 1 ] for k in range(1,int((s+e)*60/5))]))
          '''Tried datafilter instead of data here!'''


 


'''Calculation of c_2 from the exponential decay'''
'''4. STATE-DEPENDENT Secondary control c_2 /( EXPERIMENTATION)!!!'''

gap   =0
size  = 899
steps = size+1

window = 3600
data_range = data.size // window

c_2_decays = np.zeros(data_range)

for j in range(1,data_range) :
    # i f the f r e q u e n c y t r a j e c t o r y moves p o s i t i v e l y
    
    if np.sum(( np.diff( data[ 3600 *( j ) : 3600 * ( j ) +10]) ) ) > 0 :
        c_2_decays[j] = curve_fit(lambda t , a , b :#,  c : 
        a*np.exp(-b*t ) ,#* (1-np.exp(-c * t+2*b* t ) ) ,
        np.linspace( 0 , size ,steps ) , data[ 3600 * ( j ) -gap: 3600 * ( j )-gap + steps] ,
        p0 = ( 0.08 , 0.00455 
             ),#, 0.035  ) , 
        maxfev=10000)[ 0 ][ 1 ]
#g = datafilter
'''Use g (datafilter with sigma =60) for calculating c_2'''      
#     if np.sum(( np.diff( g[ 3600 *( j ) : 3600 * ( j ) +10]) ) ) > 0 :
#         c_2_decays[j] = curve_fit(lambda t , a , b  : 
#         a*np.exp(-b*t )/(q_1-2*b),# * (1-np.exp(-c * t+2*b* t ) ) ,   #use different c_? !!!
#         np.linspace( 0 , size ,steps ) , g[ 3600 * ( j ) -gap: 3600 * ( j )-gap + steps] ,
#         p0 = ( 0.08 , 0.00455 
#              ),#, 0.035  ) , 
#         maxfev=10000)[ 0 ][ 1 ]
print(c_2_decays)

for j in range(1,data_range) :
    
    if np.sum(( np.diff( data[ 3600 *( j ) : 3600 * ( j ) +10]) ) ) <= 0 :
        c_2_decays[j] = curve_fit(lambda t , a , b :#,,  c : 
        (-a)*np.exp(-b*t ) ,#* (1-np.exp(-c * t+2*b* t ) ) ,
        np.linspace( 0 , size ,steps ) , data[ 3600 * ( j ) -gap: 3600 * ( j )-gap + steps] ,
        p0 = ( 0.08 , 0.00455 
             ),#, 0.035  ) , 
        maxfev=10000)[ 0 ][ 1 ]
        
    '''Use g (datafilter with sigma =60) for calculating c_2'''   
#     if np.sum(( np.diff( g[ 3600 *( j ) : 3600 * ( j ) +10]) ) ) <= 0 :
#         c_2_decays[j] = curve_fit(lambda t , a , b :#,,  c : 
#         (-a)*np.exp(-b*t )/(q_1-2*b) ,#* (1-np.exp(-c * t+2*b* t ) ) ,             #use different c_? !!!
#         np.linspace( 0 , size ,steps ) , g[ 3600 * ( j ) -gap: 3600 * ( j )-gap + steps] ,
#         p0 = ( 0.08 , 0.00455 
#              ),#, 0.035  ) , 
#         maxfev=10000)[ 0 ][ 1 ]
print(c_2_decays)

c_2_decays
temp_c_2_decays = c_2_decays[np.argsort(c_2_decays)][: - c_2_decays.size // 5]
# c_2_linear = np.mean(temp_c_2_decays ) * (c_1)
# #c_2 = np.mean(temp_c_2_decays ) * (c_21)
# #c_2 = np.mean(temp_c_2_decays ) * (-q_1)
# c_2_pol = np.mean(temp_c_2_decays ) * (-p_1)
# c_2_far = np.mean(temp_c_2_decays ) * (-p_3)
# '''Attention: sign!'''
# c_2_linear,c_2_pol,c_2_far, c_1

omega_arr = np.linspace(-0.5,0.5,101)
c_2_arr = np.mean(temp_c_2_decays)*(3*(-p_3)*omega_arr**2 - p_1)


