import glob
import matplotlib
from collections import OrderedDict
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['mathtext.rm'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
matplotlib.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
matplotlib.rcParams.update({'font.size': 9})

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

import pandas as pd

import dedalus.public as de


mColor = 'darkorange'#olivedrab'
fColor = 'indigo'
colors = [fColor, mColor]

# Get data from files
base_dir = './data/rbc/3D/'
TT_dir   = 'TT/ra1e8'
FT_dir   = 'Nu_based_FT/ra2.61e9'

data = OrderedDict()
for label, dr in zip(['TT', 'FT'], [base_dir+TT_dir, base_dir+FT_dir]):
    ra = float(dr.split('/')[-1].split('ra')[-1])

    #Scalar Data
    data[label] = OrderedDict()
    with h5py.File('{:s}/traces/full_traces.h5'.format(dr), 'r') as f:
        for k in f.keys():
            data[label][k] = f[k][()]

    Nu = np.mean(data[label]['Nu'][-3500:])
    data[label]['Nu_mean'] = Nu
    if label == 'TT':
        data[label]['t_ff'] = 1
    else:
        data[label]['t_ff'] = np.sqrt(Nu)
    data[label]['ra_temp'] = ra*data[label]['delta_T']
    data[label]['ra_flux'] = ra**(3./2)*data[label]['left_flux']

    deltaT = np.mean(data[label]['delta_T'][-3500:])
    data[label]['delta_T_mean'] = deltaT

    #PDF data
    with h5py.File('{:s}/pdfs/pdf_data.h5'.format(dr), 'r') as f:
        for k in f.keys():
            for sk in f[k].keys():
                data[label]['pdf_{:s}_{:s}'.format(k, sk)] = f[k][sk][()]

    #slice data
    with h5py.File('{:s}/temp_slice.h5'.format(dr), 'r') as f:
        for k in f.keys():
            data[label]['slice_{}'.format(k)] = f[k][()]


for k in data['TT']: print(k)


# Set up figure subplots
fig = plt.figure(figsize=(7.5, 3))
gs = gridspec.GridSpec(1000, 1000)

subplots = [( (0  , 0  ),       450,    250),
            ( (550, 0  ),       450,    250),
            ( (50 , 280),       450,    360),
            ( (50 , 640),       450,    180),
            ( (50 , 820),       450,    180),
            ( (500, 280),       450,    360),
            ( (500, 640),       450,    180),
            ( (500, 820),       450,    180),
            ]
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]
cax = plt.subplot(gs.new_subplotspec((00, 440), 50, 400))

#PDF plots
for i, bc in enumerate(data.keys()):
    axs[0].plot(data[bc]['pdf_T_xs'] /data[bc]['delta_T_mean'],        data[bc]['pdf_T_pdf']*data[bc]['delta_T_mean'],         c=colors[i], label=bc)
    axs[1].plot(data[bc]['pdf_wT_xs']/data[bc]['delta_T_mean']**(3/2), data[bc]['pdf_wT_pdf']*data[bc]['delta_T_mean']**(3/2), c=colors[i])

    axs[0].fill_between(data[bc]['pdf_T_xs'] /data[bc]['delta_T_mean'],        1e-16, data[bc]['pdf_T_pdf']*data[bc]['delta_T_mean'],         color=colors[i], alpha=0.5)
    axs[1].fill_between(data[bc]['pdf_wT_xs']/data[bc]['delta_T_mean']**(3/2), 1e-16, data[bc]['pdf_wT_pdf']*data[bc]['delta_T_mean']**(3/2), color=colors[i], alpha=0.5)

axs[0].legend(loc='best', fontsize=8)
axs[0].set_xlabel(r'$T$')
axs[0].set_ylabel(r'$P(T)$')
axs[0].xaxis.tick_top()
axs[0].xaxis.set_label_position('top')


axs[1].set_xlabel(r'$wT$')
axs[1].set_ylabel(r'$P(wT)$')

for i in range(2):
    axs[i].set_yscale('log')
    axs[i].set_ylim(1e-3, 30)
axs[0].set_xlim(0, 1.5)
axs[1].set_xlim(-0.28, 0.28)



# Dynamics Plots

# Dedalus domain for interpolation
x_basis = de.Fourier('x', len(data['TT']['slice_x']), interval=[-1, 1], dealias=1)
y_basis = de.Fourier('y', len(data['TT']['slice_y']), interval=[-1, 1], dealias=1)
z_basis = de.Chebyshev('z', len(data['TT']['slice_z']), interval=[-0.5, 0.5], dealias=1)
horiz_domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
vert_domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
hires = 1024

horiz_scales = int(np.round(hires/len(data['TT']['slice_x'])))
vert_scales = int(np.round(hires/len(data['TT']['slice_z'])))

horiz_field = horiz_domain.new_field()
vert_field  = vert_domain.new_field()

yy_tb, xx_tb = np.meshgrid(horiz_domain.grid(-1, scales=horiz_scales), horiz_domain.grid(0, scales=horiz_scales))
zz_mid, xx_mid = np.meshgrid(vert_domain.grid(-1, scales=vert_scales), vert_domain.grid(0, scales=vert_scales))

for i, bc in enumerate(data.keys()):
    top_map = data[bc]['slice_Ttop'] - np.mean(data[bc]['slice_Ttop'])
    bot_map = data[bc]['slice_Tbot'] - np.mean(data[bc]['slice_Tbot'])
    vert_map = data[bc]['slice_T'] - np.mean(data[bc]['slice_T'], axis=0)
    maxval = data[bc]['delta_T_mean']/3

    vert_field.set_scales(1)
    vert_field['g'] = vert_map
    vert_field.set_scales(vert_scales, keep_data=True)
    c = axs[2+3*i].pcolormesh(xx_mid, zz_mid, vert_field['g'], vmin=-maxval, vmax=maxval, cmap='RdBu_r')

    horiz_field.set_scales(1)
    horiz_field['g'] = bot_map
    horiz_field.set_scales(horiz_scales, keep_data=True)
    c = axs[3+3*i].pcolormesh(xx_tb,  yy_tb,  horiz_field['g'],  vmin=-maxval, vmax=maxval, cmap='RdBu_r')

    horiz_field.set_scales(1)
    horiz_field['g'] = top_map
    horiz_field.set_scales(horiz_scales, keep_data=True)
    c = axs[4+3*i].pcolormesh(xx_tb,  yy_tb,  horiz_field['g'],  vmin=-maxval, vmax=maxval, cmap='RdBu_r')

bar = plt.colorbar(c, cax=cax, orientation='horizontal')
cax.set_xticklabels(())
bar.set_ticks(())
cax.text(-0.2, 0.25, r'$-\Delta T / 3$', transform=cax.transAxes)
cax.text(1.02, 0.25, r'$+\Delta T / 3$', transform=cax.transAxes)
cax.text(0.5, 1.4,   r'$T - \overline{T}$',     transform=cax.transAxes, ha='center')

axs[5].text(0.5, -0.12, '$T(y = 0)$',   transform=axs[5].transAxes, ha='center')
axs[6].text(0.5, -0.12, '$T(z=-0.49)$', transform=axs[6].transAxes, ha='center')
axs[7].text(0.5, -0.12, '$T(z=0.49)$',  transform=axs[7].transAxes, ha='center')


bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.75)
for i, bc in enumerate(['TT', 'FT']):
    axs[2+3*i].text(0.08, 0.87, bc, ha="center", va="center", size=8, bbox=bbox_props, transform=axs[2+3*i].transAxes)
    axs[3+3*i].text(0.16, 0.87, bc, ha="center", va="center", size=8, bbox=bbox_props, transform=axs[3+3*i].transAxes)
    axs[4+3*i].text(0.16, 0.87, bc, ha="center", va="center", size=8, bbox=bbox_props, transform=axs[4+3*i].transAxes)


for i in [2, 3, 4, 5, 6, 7]:
    axs[i].set_xticks(())
    axs[i].set_yticks(())

for i in [3, 4, 6, 7]:
    axs[i].axhline(0, c='black', lw=0.25)





fig.savefig('rbc_3D_panels.png', dpi=300, bbox_inches='tight')
#fig.savefig('rbc_3D_panels.pdf', dpi=300, bbox_inches='tight')
