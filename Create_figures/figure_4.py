# %% Created by Leonardo Rydin Gorjão, Ulrich Oberhofer, and Benjamin
# Schäfer. Most python libraries are standard (e.g. via Anaconda). If TeX is not
# present in your system, comment out lines 10 to 13.

import numpy as np
import scipy.stats as st

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 18,
    'axes.labelsize': 18,'axes.titlesize': 28, 'figure.titlesize' : 28})
matplotlib.rcParams['text.usetex'] = True

data_sources = ['Iceland.npz', 'Ireland.npz', 'Balearic.npz']
colours = ['#D81B60','#1E88E5','#FFC107','#004D40']
labels = ['Model 1','Model 2','Model 3','Model 4']
# %%
bins = 100

def generate_KL_divergence(loc, bins, datatype, ranges=(None,None)):
    hist_freq = np.zeros((5, bins))
    KL = np.zeros(4)

    data = np.load(loc)
    ls = ['_origin', '_model1', '_model2', '_model3', '_model4']
    l = [datatype+s for s in ls] # remove [:-1] when 2d model ready

    x_freq = [data[ele] for ele in l]
    for i, ele in enumerate(x_freq):
        hist_freq[i,:], _ = np.histogram(x_freq[i], bins=bins, density=True,
            range=ranges)

    for i in range(1,5):
        mask_l = hist_freq[0,:] == 0
        mask_r = hist_freq[i,:] == 0
        KL[i-1] = st.entropy(hist_freq[0,:][~(mask_l + mask_r)],
            hist_freq[i,:][~(mask_l + mask_r)])

    return KL

def generate_KS_test(loc, bins, datatype):
    hist_freq = np.zeros((5, bins))
    KS = np.zeros(4)

    data = np.load(loc)
    ls = ['_origin', '_model1', '_model2', '_model3', '_model4']
    l = [datatype+s for s in ls] # remove [:-1] when 2d model ready

    x_freq = [data[ele] for ele in l]
    for i in range(1,4):
        KS_freq[i-1] = st.kstest(x_freq[0], x_freq[i])[0]

    return KS_freq

# %%
fig, ax = plt.subplots(1, 1, figsize=(7,3.5))

for k, loc in enumerate(data_sources):
    KL = generate_KL_divergence(loc, bins, datatype='freq', ranges=(49.75,50.25))
    [ax.scatter(k -.27 + i*0.04, KL[i], marker='o', s=300, edgecolor='k',
        facecolor=colours[i], lw=1.5) for i in range(4)]

ax_ = ax.twinx()

for k, loc in enumerate(data_sources):
    KL = generate_KL_divergence(loc, bins, datatype='incr', ranges=(-.04,.04))
    [ax.scatter(k +.15 + i*0.04, KL[i], marker='s', s=300, edgecolor='k',
        facecolor=colours[i], lw=1.5) for i in range(4)]

ax.set_yscale('log')
ax_.set_yscale('log')
ax.set_ylabel(r'$D_{\mathrm{KL}}(f_{\mathrm{emp}} | f_{\mathrm{syn}})$')
ax.yaxis.set_label_coords(-0.1, .6)
ax_.set_ylabel(r'$D_{\mathrm{KL}}(\Delta f_{\mathrm{emp}} | \Delta f_{\mathrm{syn}})$')
ax_.yaxis.set_label_coords(1.1, .6)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([r'\textbf{Iceland}', r'\textbf{Ireland}', r'\textbf{Balearic}'])
ax.set_xlim([-.5, 2.5])

ax.set_ylim([0.01, 2])
ax_.set_ylim([0.01, 2])
ax.set_yticks([0.01, 0.04, 0.1, 0.4, 1])
ax.set_yticklabels([0.01, 0.04, 0.1, 0.4, 1])
ax_.set_yticks([0.01, 0.04, 0.1, 0.4, 1])
ax_.set_yticklabels([0.01, 0.04, 0.1, 0.4, 1])

ax.scatter(-0.88, 0.035, marker='o', edgecolor='k', facecolor='white', s=200,
    hatch='////', clip_on=False)
ax.scatter(2.87, 0.022, marker='s', edgecolor='k', facecolor='white', s=200,
    hatch='////', clip_on=False)

# fake legend
[ax.scatter(-2, -2, marker='D', s=200, edgecolor=None,
    facecolor=colours[i], label=labels[i], lw=0) for i in range(4)]

ax.legend(handlelength=1, handletextpad=.5, ncol=4, columnspacing=1, loc=1,
    bbox_to_anchor=(1.05,1.24))
fig.subplots_adjust(left=.12, bottom=.11, right=.88, top=.85, wspace=.4)
fig.savefig('figs/fig4.pdf', transparent=True)

