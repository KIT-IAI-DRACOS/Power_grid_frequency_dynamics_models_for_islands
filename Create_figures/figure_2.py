# %% Created by Leonardo Rydin Gorjão, Ulrich Oberhofer, and Benjamin
# Schäfer. Most python libraries are standard (e.g. via Anaconda). If TeX is not
# present in your system, comment out lines 12 to 15. kramersmoyal can be
# installed using pip install kramersmoyal.

import numpy as np

from kramersmoyal import kmc

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 18,
    'axes.labelsize': 18,'axes.titlesize': 28, 'figure.titlesize' : 28})
matplotlib.rcParams['text.usetex'] = True

colours = ['k','#D81B60','#1E88E5','#FFC107','#004D40']
labels = ['Empirical','Model 1','Model 2','Model 3','Model 4']
ls = ['--','-','-','-','-']

# %%
data = np.load('fig2_Ireland.npz')

c_1_model1 = data['c_1_model1'].flatten()[0]
c_1_model2 = data['c_1_model2'].flatten()[0]

drift = data['drift_1d'][0]
diffusion = data['diffusion_1d'][0]
edges_1d = data['edges_1d'][0]

p_1 = data['p_1_model3'].flatten()[0]
p_3 = data['p_3_model3'].flatten()[0]

d_0 = data['d_0_model3'].flatten()[0]
d_2 = data['d_2_model3'].flatten()[0]

epsilon_model1 = data['epsilon_model1'].flatten()[0]
epsilon_model2 = data['epsilon_model2'].flatten()[0]

edges_2d = data['edges_2d']
kmc_2d = data['kmc_2d']

# %%
fig, ax = plt.subplots(3,1,figsize=(7,8))

l_, r_ = 500, -500

x = edges_1d[l_:r_]
ax[0].plot(x, drift[l_:r_], label=labels[0], color=colours[0], lw=2, ls=ls[0])
ax[0].plot(x, -x*c_1_model1, label=labels[1], color=colours[1], lw=2, ls=ls[1])
ax[0].plot(x, -x*c_1_model2, label=labels[2], color=colours[2], lw=2, ls=ls[2])
ax[0].plot(x, p_3*x**3+p_1*x, label=labels[3], color=colours[3], lw=2, ls=ls[3])

ax[0].set_ylim([-0.03,0.03])
ax[0].set_xlim([-0.5,0.5])

ax[0].set_ylabel(r'Drift $D_1(\omega)$', labelpad=0)
ax[0].set_xticklabels([])

ax[1].plot(x, diffusion[l_:r_], label=labels[0], color=colours[0], lw=2, ls=ls[0])
ax[1].plot(x, (np.ones(x.size)*epsilon_model1**2)/2, label=labels[1], color=colours[1], lw=2, ls=ls[1])
ax[1].plot(x, d_2*(x-0)**2 + d_0, label=labels[3], color=colours[3], lw=2, ls=ls[3])
ax[1].plot(x, (np.ones(x.size)*epsilon_model2**2)/2, label=labels[2], color=colours[2], lw=2, ls=ls[2])

ax[1].set_xlim([-0.7,0.7])
ax[1].set_ylim([1e-5,9e-4])

ax[1].set_yticks([2.5e-4,5e-4,7.5e-4])
ax[1].set_yticklabels([2.5,5,7.5])
fig.text(.05,.635,r'$\times10^{-4}$')

ax[1].set_ylabel(r'Diffusion $D_2(\omega)$', labelpad=20)
ax[1].set_xlabel(r'$\omega$ [rads$^{-1}$]')

ax[2].axis('off')
##

ax1 = fig.add_axes([0.07,0.01,0.38,0.38], projection='3d', proj_type='ortho')

d = 20
l_, r_ = np.argmin(edges_2d[0]**2), np.argmin(edges_2d[1]**2)

X_1, X_2 = np.meshgrid(edges_2d[0][l_-d:l_+d], edges_2d[1][r_-d:r_+d])
ax1.contour3D(X_1, X_2, ((kmc_2d[2,l_-d:l_+d,r_-d:r_+d]).T), 500,
    cmap='Greens_r')

ax1.set_xlabel(r'$\theta$', fontsize=18)
ax1.set_ylabel(r'$\omega$', fontsize=18)

ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax2 = fig.add_axes([0.57,0.01,0.38,0.38], projection='3d', proj_type='ortho')


d = 20
l_, r_ = np.argmin(edges_2d[0]**2), np.argmin(edges_2d[1]**2)

X_1, X_2 = np.meshgrid(edges_2d[0][l_-d:l_+d], edges_2d[1][r_-d:r_+d])
ax2.contour3D(X_1, X_2, ((kmc_2d[5,l_-d:l_+d,r_-d:r_+d]).T), 500,
    cmap='Greens_r')

ax2.set_xlabel(r'$\theta$', fontsize=18)
ax2.set_ylabel(r'$\omega$', fontsize=18)

ax1.set_yticks([-.05,0,.05])
ax1.set_yticklabels([r'$-5$',r'$0$',r'$5$'], ha='right', va='center')

ax2.set_yticks([-.05,0,.05])
ax2.set_yticklabels([r'$-5$',r'$0$',r'$5$'], ha='right', va='center')

ax1.set_xticks([-2,0,2])
ax1.set_xticklabels([r'$-2$',r'$0$',r'$2$'], ha='right', va='center')

ax1.set_zticks([-0.005,.0,.005])
ax1.set_zticklabels([r'$-5$',r'$0$',r'$5$'], ha='right', va='center')

ax2.set_xticks([-2,0,2])
ax2.set_xticklabels([r'$-2$',r'$0$',r'$2$'], ha='right', va='center')

ax2.set_zticks([0,.0001,.0002])
ax2.set_zticklabels([r'$0$',r'$1$',r'$2$'], ha='right', va='center')

ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

fig.text(.41,.06,r'$\times10^{-2}$', fontsize=16, rotation=30)
fig.text(.91,.06,r'$\times10^{-2}$', fontsize=16, rotation=30)
fig.text(.91,.28,r'$\times10^{-4}$', fontsize=16)
fig.text(.41,.28,r'$\times10^{-2}$', fontsize=16)

fig.text(0.02,0.1, r'Drift $D_{0,1}(\theta,\omega)$',
    fontsize=18, rotation=90)
fig.text(0.52,0.06, r'Diffusion $D_{0,2}(\theta,\omega)$',
    fontsize=18, rotation=90)

ax1.view_init(30, -45)
ax2.view_init(30, -45)

ax[0].legend(handlelength=1, handletextpad=.5, ncol=4, columnspacing=1, loc=1,
    bbox_to_anchor=(1.0,1.29))
fig.subplots_adjust(left=.15, bottom=.12, right=.99, top=.93, hspace=.03)
fig.savefig('figs/fig2.pdf', dpi=200, transparent=True)
