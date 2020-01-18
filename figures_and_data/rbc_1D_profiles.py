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

def read_profs(ra_list, dir_list):
    full_data = OrderedDict()
    for ra, dr in zip(ra_list, dir_list):
        prof_files = sub_runs = glob.glob('{:s}/avg_profs/averaged_avg_profs.h5'.format(dr))
        if len(prof_files) > 0:
            data = OrderedDict()
            with h5py.File('{:s}'.format(prof_files[0]), 'r') as f:
                for k in f.keys():
                    data[k] = f[k][()]
            full_data['{:.4e}'.format(ra)] = data
    return full_data

def read_asymms(ra_list, dir_list):
    full_data = OrderedDict()
    for ra, dr in zip(ra_list, dir_list):
        prof_files = sub_runs = glob.glob('{:s}/asymmetries/asymmetry_data.h5'.format(dr))
        if len(prof_files) > 0:
            data = OrderedDict()
            with h5py.File('{:s}'.format(prof_files[0]), 'r') as f:
                for k in f.keys():
                    if k == 'z':
                        data[k] = f[k][()]
                    else:
                        data['{:s}_pos'.format(k)] = f[k]['pos_mask'][()]
                        data['{:s}_neg'.format(k)] = f[k]['neg_mask'][()]
            full_data['{:.4e}'.format(ra)] = data
    return full_data


mixed_dirs = glob.glob("./data/rbc/mixedFT_2d/ra*/")
fixed_dirs = glob.glob("./data/rbc/fixedT_2d/ra*/")
numbered_dirs  = [(f, float(f.split("./data/rbc/mixedFT_2d/ra")[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("./data/rbc/fixedT_2d/ra")[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_profs = read_profs(mixed_ras, mixed_dirs)
fixed_profs = read_profs(fixed_ras, fixed_dirs)
mixed_scalars = read_data(mixed_ras, mixed_dirs)
fixed_scalars = read_data(fixed_ras, fixed_dirs)
mixed_asymms = read_asymms(mixed_ras, mixed_dirs)
fixed_asymms = read_asymms(fixed_ras, fixed_dirs)


mk = '{:.4e}'.format(2.61e9)
fk = '{:.4e}'.format(1.00e8)

mz = mixed_profs[mk]['z']
fz = fixed_profs[fk]['z']
max_nz = np.max((len(mz), len(fz)))


delta_T_mixed = mixed_scalars[mk]['delta_T']
dT = np.mean(delta_T_mixed[-5000:])


z_basis = de.Chebyshev(  'z', max_nz, interval=[-1/2, 1/2], dealias=1)
bases = [z_basis]
domain = de.Domain(bases, grid_dtype=np.float64)

p_mixed = domain.new_field()
p_fixed = domain.new_field()

z = domain.grid(0)

fig = plt.figure()

# Set up figure subplots
fig = plt.figure(figsize=(7.5, 2))
gs = gridspec.GridSpec(1000, 1000)

subplots = [( (50 , 0),         600,    315),
            ( (650, 0),         300,    315),
            ( (50,  315),       600,    315),
            ( (650, 315),       300,    315),
            ( (50,  750),       600,    250),
            ( (650, 750),       300,    250)
            ]
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]



#Panel 1 - Temp
k = 'T'
ax1, ax2 = axs[0], axs[1]
ax1.plot(fz, fixed_profs[fk][k][0,:],    c=fColor, label='fixed-T', lw=2)
ax1.plot(mz, mixed_profs[mk][k][0,:]/dT, c=mColor, label='mixed-FT')
ax1.set_ylim(0, 1)
ax1.legend(loc='lower left', frameon=False, fontsize=7)
ax1.set_ylabel(r'$\bar{T}/\Delta T$')

#Panel 2 - Temp diff
p_mixed.set_scales(len(mixed_profs[mk][k][0,:])/max_nz)
p_fixed.set_scales(len(fixed_profs[fk][k][0,:])/max_nz)
p_mixed['g'] = mixed_profs[mk][k][0,:]/dT
p_fixed['g'] = fixed_profs[fk][k][0,:]
for p in [p_mixed, p_fixed]: p.set_scales(1, keep_data=True)

ax2.plot(z, 100*np.abs((p_fixed['g'] - p_mixed['g'])/p_fixed['g']), c='k')
ax2.set_yscale('log')
ax2.set_ylim(1e-1, 3e0)
ax2.set_ylabel('% diff')
ax2.set_xticks((-0.5, 0, 0.5))

#Panel 3 - Asymmetries
ax1, ax2 = axs[2], axs[3]
k1 = 'T-w_pos'
k2 = 'T-w_neg'

ax1.plot(mz, mixed_asymms[mk][k1]/dT, c=mColor, lw=2)
ax1.plot(mz, mixed_asymms[mk][k2]/dT, c=mColor, ls='--', lw=2)
ax1.plot(fz, fixed_asymms[fk][k1], c=fColor, lw=1, label='Upflows')
ax1.plot(fz, fixed_asymms[fk][k2], c=fColor, ls='--', lw=1, label='Downflows')
ax1.legend(loc='lower left', frameon=False, fontsize=7)


#Panel 4 - Asymmetry diffs
for k, ls in [(k1, '-'), (k2, '--')]:
    p_mixed.set_scales(len(mixed_asymms[mk][k])/max_nz)
    p_fixed.set_scales(len(fixed_asymms[fk][k])/max_nz)
    p_mixed['g'] = mixed_asymms[mk][k]/dT
    p_fixed['g'] = fixed_asymms[fk][k]
    for p in [p_mixed, p_fixed]: p.set_scales(1, keep_data=True)

    good = np.abs(p_fixed['g'])/p_fixed['g'].max() > 0.1
    ax2.plot(z[good], np.abs((p_fixed['g'] - p_mixed['g'])/p_fixed['g'])[good], c='k', ls=ls)
ax2.set_yscale('log')
ax2.set_ylim(1e-3, 3e-2)


#Panel 5 - enth Flux
k = 'enth_flux'
ax1, ax2 = axs[4], axs[5]
ax1.axhline(0, c='k', lw=0.5)
ax1.plot(fz, fixed_profs[fk][k][0,:], c=fColor, lw=2)
ax1.plot(mz, mixed_profs[mk][k][0,:]/dT**(3/2), c=mColor, label=r'$F_{\mathrm{enth}}$')

#Panel 6 - enth flux diff
ax2.axhline(1, c='k', lw=0.5)
p_mixed.set_scales(len(mixed_profs[mk][k][0,:])/max_nz)
p_fixed.set_scales(len(fixed_profs[fk][k][0,:])/max_nz)
p_mixed['g'] = mixed_profs[mk][k][0,:]/dT**(3/2)
p_fixed['g'] = fixed_profs[fk][k][0,:]
for p in [p_mixed, p_fixed]: p.set_scales(1, keep_data=True)

good = np.abs(p_fixed['g'])/p_fixed['g'].max() > 0.02
ax2.plot(z[good], 100*np.abs((p_fixed['g'] - p_mixed['g'])/p_fixed['g'])[good], c='k')

#Panel 5 - kappa Flux
k = 'kappa_flux'
ax1.plot(fz, fixed_profs[fk][k][0,:], c=fColor, ls='-.', lw=2)
ax1.plot(mz, mixed_profs[mk][k][0,:]/dT**(3/2), c=mColor, ls='-.', label=r'$F_{\mathrm{cond}}$')
ax1.set_ylim(-3e-4, 0.0027)
ax1.set_yticks((0, 0.001, 0.002))
ax1.legend(loc='center', frameon=False, fontsize=7)
ax1.set_ylabel(r'Flux/($\Delta T$)$^{3/2}$')

#Panel 6 - kappa flux diff
p_mixed.set_scales(len(mixed_profs[mk][k][0,:])/max_nz)
p_fixed.set_scales(len(fixed_profs[fk][k][0,:])/max_nz)
p_mixed['g'] = mixed_profs[mk][k][0,:]/dT**(3/2)
p_fixed['g'] = fixed_profs[fk][k][0,:]
for p in [p_mixed, p_fixed]: p.set_scales(1, keep_data=True)

good1 = np.abs(p_fixed['g'][z<0])/p_fixed['g'][z<0].max() > 0.02
good2 = np.abs(p_fixed['g'][z>0])/p_fixed['g'][z>0].max() > 0.02
ax2.plot(z[z<0][good1], 100*np.abs((p_fixed['g'] - p_mixed['g'])/p_fixed['g'])[z<0][good1], c='k', ls='-.')
ax2.plot(z[z>0][good2], 100*np.abs((p_fixed['g'] - p_mixed['g'])/p_fixed['g'])[z>0][good2], c='k', ls='-.')
ax2.set_ylabel('% diff')
ax2.set_yscale('log')
ax2.set_ylim(1e-1, 30)
ax2.set_yticks((1e-1, 1, 10))





#mixed_profs[mk][k][0,:]/dT/fixed_profs[fk][k][0,:])
for i in [1, 3, 5]:
    axs[i].set_xlabel('z')

for i in [0, 2, 4]:
    axs[i].set_xticks(())

for i in [2, 3]:
    axs[i].set_yticks(())
    axs[i].set_xticks(())

for i in range(6):
    axs[i].set_xlim(-0.5, 0.5)

fig.savefig('rbc_1D_profiles.png', dpi=300, bbox_inches='tight')
fig.savefig('rbc_1D_profiles.pdf', dpi=300, bbox_inches='tight')
