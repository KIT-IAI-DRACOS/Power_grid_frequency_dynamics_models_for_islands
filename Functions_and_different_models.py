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

'''For calculations: use angular velocity omega = 2*pi*frequency



'''Necessary functions: De-trending, Estimation of KM-Coefficients, Calculation of Power mismatch, Calculation of c_2'''
'''Different models: 1: linear c_1 and epsilon, no c_2 and Delta_P
                     2: linear c_1,c_2 and epsilon,
                     3: cubic c_1(omega), state-dependent c_2 and epsilon
                     4: 
'''
