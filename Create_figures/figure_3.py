#%% Created by Leonardo Rydin Gorjão, Ulrich Jakob Oberhofer, and Benjamin
# Schäfer. Most python libraries are standard (e.g. via Anaconda). If TeX is not
# present in your system, comment out lines 9 to 12.

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 18,
    'axes.labelsize': 18,'axes.titlesize': 28, 'figure.titlesize' : 28})
matplotlib.rcParams['text.usetex'] = True
from matplotlib.ticker import LogLocator, NullFormatter

#data_sources = ['Iceland.npz', 'Ireland.npz', 'Balearic.npz']
data_sources = ['Iceland_data.npz', 'Irish_data.npz', 'Balearic_data.npz']

colours = ['k','#D81B60','#1E88E5','#FFC107','#004D40']
labels = ['Empirical','Model 1','Model 2','Model 3','Model 4']
ls = ['--','-','-','-','-']

# %% helping functions
def generate_hist_data(loc, bins, datatype, ranges=(None,None)):
    hist_freq = np.zeros((5, bins))
    edge_freq = np.zeros((5, bins))

    data = np.load(loc)

    ls = ['_origin', '_model1', '_model2', '_model3', '_model4']
    l = [datatype+s for s in ls] # remove [:-1] when 2d model ready

    x_freq = [data[ele] for ele in l]

    for i, ele in enumerate(x_freq):
        hist_freq[i,:], _ = np.histogram(x_freq[i], bins=bins, density=True, range=ranges)
        edge_freq[i,:] = (_[1:] + _[:-1])/2

    return hist_freq, edge_freq

def generate_AC_data(loc, length, datatype='auto'):
    data = np.load(loc)

    ls = ['_origin', '_model1', '_model2', '_model3', '_model4']
    l = [datatype+s for s in ls] # remove [:-1] when 2d model ready

    AC_freq = np.ones((5, data[l[0]].size))

    for i in range(5):
        AC_freq[i,:] = data[l[i]]

    tau = np.linspace(0, length, data[l[0]].size)
    # calculate AC manually
    # l_ = ['freq'+s for s in ['origin', 'model1', 'model2', 'model3', 'model4']]
    # x_freq = [data[ele] for ele in l_]

    # for i in range(4):
    #     for j in range(1,length):
    #         x = x_freq[i] - np.mean(x_freq[i])
    #         AC_freq[i,j] = np.corrcoef(x[j:],x[:-j])[0,1]

    return tau, AC_freq

# %%
fig, ax = plt.subplots(3,3,figsize=(16,8))

for k, loc in enumerate(data_sources):
    hist_freq, edge_freq = generate_hist_data(loc, bins=100, datatype='freq',
        ranges=(49.75,50.25))
    [ax[k,0].semilogy(edge_freq[i,:], hist_freq[i,:], lw=2, ls=ls[i],
        color=colours[i], label=labels[i]) for i in range(5)]
    hist_incr, edge_incr = generate_hist_data(loc, bins=100, datatype='incr',
        ranges=(-.04,.04))
    [ax[k,1].semilogy(edge_incr[i,:], hist_incr[i,:], lw=2, ls=ls[i],
        color=colours[i]) for i in range(5)]

for k, loc in enumerate(data_sources):
    tau, AC_freq = generate_AC_data(loc, 90)
    [ax[k,2].plot(tau, AC_freq[i,:], lw=2, color=colours[i], ls=ls[i]) for i in range(5)]

ax[0,0].set_ylabel(r'\textbf{Iceland}~\\\\ PDF $\rho(f)$')
ax[1,0].set_ylabel(r'\textbf{Ireland}~\\\\ PDF $\rho(f)$')
ax[2,0].set_ylabel(r'\textbf{Balearic}~\\\\ PDF $\rho(f)$')
[ax[i,1].set_ylabel(r'PDF $\rho(\Delta f)$') for i in range(3)]
[ax[i,2].set_ylabel(r'Autocorrelation') for i in range(3)]

ax[2,0].set_xlabel(r'$f$ [Hz]')
ax[2,1].set_xlabel(r'$\Delta f$ [Hz]')
ax[2,2].set_xlabel(r'$\tau$ [min]')

[[ax[i,j].yaxis.set_major_locator(LogLocator(10,
    numticks=100)) for i in range(3)] for j in range(2)]
[[ax[i,j].yaxis.set_minor_locator(LogLocator(10,
    np.arange(2, 10)*.1, numticks=100)) for i in range(3)] for j in range(2)]
[[ax[i,j].yaxis.set_minor_formatter(NullFormatter()) for i in range(3)] for j in range(2)]

[ax[i,2].set_xticks([0,30,60,90]) for i in range(3)]

[ax[i,0].set_xlim([49.75,50.25]) for i in range(3)]
[ax[i,1].set_xlim([-0.035,0.035]) for i in range(3)]
[ax[i,0].set_ylim([2e-3,None]) for i in range(3)]
[ax[i,1].set_ylim([2e-3,None]) for i in range(3)]

[[ax[i,j].set_xticklabels([]) for i in range(2)] for j in range(3)]

ax[0,0].set_title('full data', fontsize=20)
ax[0,1].set_title('increments', fontsize=20)
ax[0,2].set_title('autocorrelation', fontsize=20)

ax[0,0].legend(handlelength=1, handletextpad=.5, ncol=5, columnspacing=1, loc=1, bbox_to_anchor=(2.65,1.45))
fig.subplots_adjust(left=.08, bottom=.08, right=.99, top=.89, wspace=.25, hspace=.04)
#fig.savefig('figs/fig3.pdf', transparent=True)
