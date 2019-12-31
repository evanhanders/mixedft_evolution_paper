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


def read_data(ra_list, dir_list, keys=['Nu', 'delta_T', 'sim_time', 'Pe', 'KE', 'left_flux', 'right_flux']):
    full_data = OrderedDict()
    for ra, dr in zip(ra_list, dir_list):
        data = OrderedDict()
        sub_runs = glob.glob('{:s}/run*/'.format(dr))
        if len(sub_runs) > 0:
            partial_data = OrderedDict()
            for k in keys:
                partial_data[k] = []

            N_tot = 0
            for i, dr in enumerate(sub_runs):
                with h5py.File('{:s}/traces/full_traces.h5'.format(dr), 'r') as f:
                    N_tot += len(f['Nu'])
                    for k in keys:
                        partial_data[k].append(f[k][()])

            data = OrderedDict()
            for k in keys:
                data[k] = np.zeros(N_tot)
                start_ind=0
                for arr in partial_data[k]:
                    data[k][start_ind:start_ind+len(arr)] = arr
                    start_ind += len(arr)
        else:
            with h5py.File('{:s}/traces/full_traces.h5'.format(dr), 'r') as f:
                for k in keys:
                    data[k] = f[k][()]

        data['ra_temp'] = ra*data['delta_T']
        data['ra_flux'] = ra**(3./2)*data['left_flux']
        full_data['{:.4e}'.format(ra)] = data
    return full_data

def read_pdfs(ra_list, dir_list):
    full_data = OrderedDict()
    for ra, dr in zip(ra_list, dir_list):
        pdf_files = sub_runs = glob.glob('{:s}/pdfs/pdf_data.h5'.format(dr))
        if len(pdf_files) > 0:
            data = OrderedDict()
            with h5py.File('{:s}'.format(pdf_files[0]), 'r') as f:
                for k in f.keys():
                    data[k] = OrderedDict()
                    for sk in f[k].keys():
                        data[k][sk] = f[k][sk][()]
            full_data['{:.4e}'.format(ra)] = data
    return full_data

mixed_dirs = glob.glob("./data/rbc/mixedFT_2d/ra*/")
fixed_dirs = glob.glob("./data/rbc/fixedT_2d/ra*/")
numbered_dirs  = [(f, float(f.split("./data/rbc/mixedFT_2d/ra")[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("./data/rbc/fixedT_2d/ra")[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_data = read_pdfs(mixed_ras, mixed_dirs)
fixed_data = read_pdfs(fixed_ras, fixed_dirs)
mixed_scalars = read_data(mixed_ras, mixed_dirs)
fixed_scalars = read_data(fixed_ras, fixed_dirs)

mk = '{:.4e}'.format(2.61e9)
fk = '{:.4e}'.format(1.00e8)

# Set up figure subplots
fig = plt.figure(figsize=(7.5, 2))
gs = gridspec.GridSpec(1000, 1000)

subplots = [( (50 , 0),         900,    300),
            ( (50 , 400),       900,    400),
            ( (350, 800),       300,    150),
            ( (650, 800),       300,    150)
            ]
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]

cax = plt.subplot(gs.new_subplotspec((100, 800), 125, 150))

# Panel 1, PDF comparison
plt.sca(axs[0])
ax = axs[0]
mx, mp, mdx = [mixed_data[mk]['T'][k] for k in ['xs', 'pdf', 'dx']]
fx, fp, fdx = [fixed_data[fk]['T'][k] for k in ['xs', 'pdf', 'dx']]

delta_T_mixed = mixed_scalars[mk]['delta_T']
dT = np.mean(delta_T_mixed[-5000:])
plt.plot(mx/dT, mp*dT, label='mixed-FT', c='olivedrab')
plt.plot(fx, fp, label='fixed-T', c='darkorange')
plt.yscale('log')
ax.set_ylabel(r'$P(T/\Delta T)$')
ax.set_xlabel(r'$T/\Delta T$')
ax.legend(loc='upper right', frameon=False, fontsize=7, markerfirst=False, borderpad=0.1)

ax.fill_between(mx/dT, 1e-16, mp*dT, color='olivedrab', alpha=0.5)
ax.fill_between(fx, 1e-16, fp, color='darkorange', alpha=0.5)

minv = np.min((np.min(mp[mp > 0]*dT), np.min(fp[fp > 0])))
maxv = np.max((np.max(mp[mp > 0]*dT), np.max(fp[fp > 0])))
ax.set_ylim(minv, maxv)
ax.set_xlim(0, np.max(mx/dT))


# Panel 2, Dynamics
plt.sca(axs[1])
ax = axs[1]

with h5py.File('./data/rbc/mixedFT_2d/ra2.61e9/slice_file.h5', 'r') as f:
    x = f['scales/x/1.0'][()].flatten()
    z = f['scales/z/1.0'][()].flatten()
    T = f['tasks/T'][()][10,:]

with h5py.File('./data/rbc/mixedFT_2d/ra2.61e9/avg_profs/averaged_avg_profs.h5', 'r') as f:
    T_prof = f['T'][()][0,:]


x_basis = de.Fourier(    'x', len(x), interval=[-1, 1], dealias=1)
z_basis = de.Chebyshev(  'z', len(z), interval=[-1/2, 1/2], dealias=1)
bases = [x_basis, z_basis]
domain = de.Domain(bases, grid_dtype=np.float64)
base_scale=1
highres_n = 4096
big_scale=int(highres_n/len(x))
T_field = domain.new_field()
T_field['g'] = T - T_prof

x_big = domain.grid(0, scales=big_scale)
z_big = domain.grid(1, scales=big_scale)
zz_b, xx_b = np.meshgrid(z_big.flatten(), x_big.flatten())

T_field.set_scales(big_scale, keep_data=True)
c = ax.pcolormesh(xx_b, zz_b, T_field['g'], cmap='RdBu_r', rasterized=True, vmin=-dT/2, vmax=dT/2)

top_plume_bounds = [(-0.75, -0.5, -0.5, -0.75, -0.75),  (0.5, 0.5, 0.4, 0.4, 0.5)]
bot_plume_bounds = [(0.65, 0.9, 0.9, 0.65, 0.65), (-0.4, -0.4, -0.5, -0.5, -0.4)]
for i in range(4):
    plt.plot(top_plume_bounds[0][i:i+2], top_plume_bounds[1][i:i+2], c='k')
    plt.plot(bot_plume_bounds[0][i:i+2], bot_plume_bounds[1][i:i+2], c='k')

ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 0.5)
ax.set_ylabel('z')
ax.set_xlabel('x')

ax.text(0.02, 0.02, 'Fixed Flux (bottom)', transform = ax.transAxes)
ax.text(0.51, 0.92, 'Fixed Temp (top)', transform = ax.transAxes)

 
# Panel 3, Dynamics--Upper plume
plt.sca(axs[2])
ax = axs[2]
ax.pcolormesh(xx_b, zz_b, T_field['g'], cmap='RdBu_r', rasterized=True, vmin=-dT/2, vmax=dT/2)
ax.set_xlim(np.min(top_plume_bounds[0]), np.max(top_plume_bounds[0]))
ax.set_ylim(np.min(top_plume_bounds[1]), np.max(top_plume_bounds[1]))


# Panel 4, Dynamics--Lower plume
plt.sca(axs[3])
ax = axs[3]
ax.pcolormesh(xx_b, zz_b, T_field['g'], cmap='RdBu_r', rasterized=True, vmin=-dT/2, vmax=dT/2)
ax.set_xlim(np.min(bot_plume_bounds[0]), np.max(bot_plume_bounds[0]))
ax.set_ylim(np.min(bot_plume_bounds[1]), np.max(bot_plume_bounds[1]))

for i in [2, 3]:
    axs[i].set_xticks(())
    axs[i].set_yticks(())

#Colorbar
bar = plt.colorbar(c, cax=cax, orientation='horizontal')
cax.set_xticklabels(())
bar.set_ticks(())
cax.text(0.28, 2.1, r'$\pm\Delta T / 2$', transform=ax.transAxes)
cax.text(0.32, 2.9, r'$T - \bar{T}$', transform=ax.transAxes)
#cax.annotate(r'$-|S| \times 10^{-5}$', fontsize=8,  xy=(-0.37, 0.5), va='center', annotation_clip=False)
#cax.annotate(r'$|S| \times 10^{-5}$', fontsize=8,  xy=(1.02, 0.5),  va='center',  annotation_clip=False)





fig.savefig('rbc_dynamics_asymmetries.png', dpi=300, bbox_inches='tight')
fig.savefig('rbc_dynamics_asymmetries.pdf', dpi=300, bbox_inches='tight')
