# %% Created by Leonardo Rydin Gorjão, Ulrich Oberhofer, and Benjamin
# Schäfer. Most python libraries are standard (e.g. via Anaconda). If TeX is not
# present in your system, comment out lines 16 to 19. Note that cartopy is
# sometimes difficult to install on Windows/iOS systems.

import numpy as np
from scipy.ndimage import gaussian_filter1d

import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.mpl.ticker as cticker

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 20,
    'axes.labelsize': 20,'axes.titlesize': 28, 'figure.titlesize' : 28})
matplotlib.rcParams['text.usetex'] = True
from matplotlib.lines import Line2D

data_sources = ['Iceland_data.npz', 'Irish_data.npz', 'Balearic_data.npz']
colours = ['#D81B60','#1E88E5','#FFC107','#004D40']
labels = [r'\textbf{Iceland}', r'\textbf{Ireland}', r'\textbf{Balearic}']

# %% Map-related data -- this downloads data so an internet connection is needed
shpfilename = shpreader.natural_earth(resolution='10m',
                                      category='cultural',
                                      name='admin_0_countries')
shpfilename_2 = shpreader.natural_earth(resolution='10m',
                                      category='cultural',
                                      name='admin_0_map_subunits')

countries = shpreader.Reader(shpfilename).records()
states = shpreader.Reader(shpfilename_2).records()

grids_country = ['IRL', 'ISL']
grids_states = {'ESP':'Balearic Islands','GBR': 'Northern Ireland'}

# %%
fig, ax = plt.subplots(1, 1,
    subplot_kw={'projection': ccrs.Orthographic(-10,45)}, figsize=(10,8))

ax.coastlines(resolution='10m',lw=.5)
ax.add_feature(cartopy.feature.OCEAN, facecolor='#65C2F5')
ax.gridlines()
ax.set_extent((-18, 4, 35.5, 68), ccrs.PlateCarree())

for country in countries:
    if country.attributes['ADM0_A3'] in grids_country:
        ax.add_geometries(country.geometry, ccrs.PlateCarree(),
            facecolor='#008631')

for state in states:
    if state.attributes['ADM0_A3'] in grids_states.keys():
        if state.attributes['SUBUNIT'] in grids_states.values():
            ax.add_geometries(state.geometry, ccrs.PlateCarree(),
                facecolor='#008631')

ax.add_feature(cartopy.feature.BORDERS, linestyle='-', lw=.5, alpha=1)

ax = [fig.add_axes([0.5,0.7,.4,.21]),
      fig.add_axes([0.5,0.40,.4,.21]),
      fig.add_axes([0.5,0.1,.4,.21])]

fig.text(0.47,0.76, labels[0], rotation=90)
fig.text(0.47,0.45, labels[1], rotation=90)
fig.text(0.47,0.14, labels[2], rotation=90)

[ax[i].set_xlabel(r'time $t$ [min]') for i in [0,1,2]]
[ax[i].set_ylabel(r'$f$ [Hz]') for i in [0,1,2]]
[ax[i].yaxis.tick_right() for i in [0,1,2]]
[ax[i].yaxis.set_label_position("right") for i in [0,1,2]]

for i, loc in enumerate(data_sources):
    x = np.load(loc)['freq_origin'][:3600*5]
    g = gaussian_filter1d(x, 60)

    t = np.linspace(0,60,3600)
    ax[i].plot(t,x[3600*4:], lw=2, color='#008631', label='data')
    ax[i].plot(t,g[3600*4:], lw=2, color='k', label='trend')

    ax[i].set_xticks([0,15,30,45,60])

ax[0].legend(handlelength=1, handletextpad=.5, ncol=2, columnspacing=1, loc=1, bbox_to_anchor=(0.85,1.45))
fig.subplots_adjust(left=-.52, bottom=.01, right=.97, top=.99,
                    hspace=.1, wspace=.2)
plt.savefig('figs/fig1.pdf', transparent=True)

