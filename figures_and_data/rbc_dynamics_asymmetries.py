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

mColor = 'darkorange'
fColor = 'indigo'

def read_data(ra_list, dir_list, keys=['Nu', 'delta_T', 'sim_time', 'Pe', 'KE', 'left_flux', 'right_flux']):
    full_data = OrderedDict()
    for ra, dr in zip(ra_list, dir_list):
        data = OrderedDict()
        sub_runs = glob.glob('{:s}/run*/'.format(dr))
        if len(sub_runs) > 0:
            numbered_dirs  = [(f, float(f.split("run")[-1].split("/")[0])) for f in sub_runs]
            sub_runs, run_num = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
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

mixed_dirs = glob.glob("./data/rbc/classic_FT_2D/ra*/")
fixed_dirs = glob.glob("./data/rbc/TT_2D/ra*/")
restarted_dirs = glob.glob("./data/rbc/TT-to-FT_2D/ra*/")
numbered_dirs  = [(f, float(f.split("./data/rbc/classic_FT_2D/ra")[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("./data/rbc/TT_2D/ra")[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("./data/rbc/TT-to-FT_2D/ra")[-1].split("/")[0])) for f in restarted_dirs]
restarted_dirs, restarted_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_data = read_pdfs(mixed_ras, mixed_dirs)
fixed_data = read_pdfs(fixed_ras, fixed_dirs)
restarted_data = read_pdfs(restarted_ras, restarted_dirs)
mixed_scalars = read_data(mixed_ras, mixed_dirs)
fixed_scalars = read_data(fixed_ras, fixed_dirs)
restarted_scalars = read_data(restarted_ras, restarted_dirs)

#mk = '{:.4e}'.format(2.61e9)
#fk = '{:.4e}'.format(1.00e8)

mk = '{:.4e}'.format(9.51e11)
fk = '{:.4e}'.format(1.00e10)

# Set up figure subplots
fig = plt.figure(figsize=(7.5, 2))
gs = gridspec.GridSpec(1000, 1000)

subplots = [( (0 , 0),          900,    300),
            ( (0 , 400),        900,    400),
            ( (0, 800),         425,    150),
            ( (425, 800),       425,    150)
            ]
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]

cax = plt.subplot(gs.new_subplotspec((950, 825), 50, 100))

# Panel 1, PDF comparison
plt.sca(axs[0])
ax = axs[0]
mx, mp, mdx = [restarted_data[mk]['T'][k] for k in ['xs', 'pdf', 'dx']]
fx, fp, fdx = [fixed_data[fk]['T'][k] for k in ['xs', 'pdf', 'dx']]

delta_T_mixed = restarted_scalars[mk]['delta_T']
dT = np.mean(delta_T_mixed[-5000:])
plt.plot(mx/dT, mp*dT, label='FT', c=mColor)
plt.plot(fx, fp, label='TT', c=fColor)
plt.yscale('log')
ax.set_ylabel(r'$P(T/\Delta T)$', labelpad=0)
ax.set_xlabel(r'$T/\Delta T$', labelpad=0)
ax.legend(loc='upper right', frameon=False, fontsize=7, markerfirst=False, borderpad=0.1)

ax.fill_between(mx/dT, 1e-16, mp*dT, color=mColor, alpha=0.5)
ax.fill_between(fx, 1e-16, fp, color=fColor, alpha=0.5)

minv = np.min((np.min(mp[mp > 0]*dT), np.min(fp[fp > 0])))
maxv = np.max((np.max(mp[mp > 0]*dT), np.max(fp[fp > 0])))
ax.set_ylim(3e-5, maxv)
ax.set_xlim(0, 1.55)#np.max(mx/dT))


# Panel 2, Dynamics
plt.sca(axs[1])
ax = axs[1]

with h5py.File('./data/rbc/TT-to-FT_2D/ra9.51e11/temp_slice.h5', 'r') as f:
    x = f['x'][()]
    z = f['z'][()]
    T = f['T'][()]

delta_T_mixed = restarted_scalars['{:.4e}'.format(9.51e11)]['delta_T']
dT = np.mean(delta_T_mixed[-5000:])

x_basis = de.Fourier(    'x', len(x), interval=[-1, 1], dealias=1)
z_basis = de.Chebyshev(  'z', len(z), interval=[-1/2, 1/2], dealias=1)
bases = [x_basis, z_basis]
domain = de.Domain(bases, grid_dtype=np.float64)
base_scale=1
highres_n = 4096
big_scale=int(highres_n/len(x))
T_field = domain.new_field()
T_field['g'] = T - T.mean(axis=0)

x_big = domain.grid(0, scales=big_scale)
z_big = domain.grid(1, scales=big_scale)
zz_b, xx_b = np.meshgrid(z_big.flatten(), x_big.flatten())

T_field.set_scales(big_scale, keep_data=True)
c = ax.pcolormesh(xx_b, zz_b, T_field['g'], cmap='RdBu_r', rasterized=True, vmin=-dT/3, vmax=dT/3)

top_plume_bounds = [(-0.72, -0.65, -0.65, -0.72, -0.72),  (0.5, 0.5, 0.45, 0.45, 0.5)]
bot_plume_bounds = [(0.53, 0.6, 0.6, 0.53, 0.53), (-0.45, -0.45, -0.5, -0.5, -0.45)]
for i in range(4):
    plt.plot(top_plume_bounds[0][i:i+2], top_plume_bounds[1][i:i+2], c='k', lw=1)
    plt.plot(bot_plume_bounds[0][i:i+2], bot_plume_bounds[1][i:i+2], c='k', lw=1)

ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 0.5)
ax.set_ylabel('z', labelpad=-1)
ax.set_xlabel('x', labelpad=-1)

bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.75)
ax.text(0.03, 0.07, 'Fixed Flux (bottom)', transform = ax.transAxes, bbox=bbox_props)
ax.text(0.50, 0.88, 'Fixed Temp (top)', transform = ax.transAxes, bbox=bbox_props)

 
# Panel 3, Dynamics--Upper plume
plt.sca(axs[2])
ax = axs[2]
ax.pcolormesh(xx_b, zz_b, T_field['g'], cmap='RdBu_r', rasterized=True, vmin=-dT/3, vmax=dT/3)
ax.set_xlim(np.min(top_plume_bounds[0]), np.max(top_plume_bounds[0]))
ax.set_ylim(np.min(top_plume_bounds[1]), np.max(top_plume_bounds[1]))


# Panel 4, Dynamics--Lower plume
plt.sca(axs[3])
ax = axs[3]
ax.pcolormesh(xx_b, zz_b, T_field['g'], cmap='RdBu_r', rasterized=True, vmin=-dT/3, vmax=dT/3)
ax.set_xlim(np.min(bot_plume_bounds[0]), np.max(bot_plume_bounds[0]))
ax.set_ylim(np.min(bot_plume_bounds[1]), np.max(bot_plume_bounds[1]))

for i in [2, 3]:
    axs[i].set_xticks(())
    axs[i].set_yticks(())

#Colorbar
bar = plt.colorbar(c, cax=cax, orientation='horizontal')#, rasterized=True)
cax.set_xticklabels(())
bar.set_ticks(())
cax.text(0.5, -0.55, r'$\pm\Delta T / 3$', transform=ax.transAxes, ha='center')
cax.text(0.5, -0.21, r'$T - \overline{T}$', transform=ax.transAxes, ha='center')
#cax.annotate(r'$-|S| \times 10^{-5}$', fontsize=8,  xy=(-0.37, 0.5), va='center', annotation_clip=False)
#cax.annotate(r'$|S| \times 10^{-5}$', fontsize=8,  xy=(1.02, 0.5),  va='center',  annotation_clip=False)





fig.savefig('rbc_dynamics_asymmetries.png', dpi=300, bbox_inches='tight')
fig.savefig('rbc_dynamics_asymmetries.pdf', dpi=300, bbox_inches='tight')
