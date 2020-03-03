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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
restarted_dirs = glob.glob("./data/rbc/restarted_mixed_T2m/ra*/")
numbered_dirs  = [(f, float(f.split("./data/rbc/mixedFT_2d/ra")[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("./data/rbc/fixedT_2d/ra")[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("./data/rbc/restarted_mixed_T2m/ra")[-1].split("/")[0])) for f in restarted_dirs]
restarted_dirs, restarted_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_profs = read_profs(mixed_ras, mixed_dirs)
fixed_profs = read_profs(fixed_ras, fixed_dirs)
restarted_profs = read_profs(restarted_ras, restarted_dirs)
mixed_scalars = read_data(mixed_ras, mixed_dirs)
fixed_scalars = read_data(fixed_ras, fixed_dirs)
restarted_scalars = read_data(restarted_ras, restarted_dirs)
mixed_asymms = read_asymms(mixed_ras, mixed_dirs)
fixed_asymms = read_asymms(fixed_ras, fixed_dirs)
restarted_asymms = read_asymms(restarted_ras, restarted_dirs)

#ra_dt = 1.00e8
#mk = '{:.4e}'.format(2.61e9)
#fk = '{:.4e}'.format(1.00e8)

ra_dt = 1.00e9
mk = '{:.4e}'.format(4.83e10)
fk = '{:.4e}'.format(1.00e9)

mz = mixed_profs[mk]['z']
fz = fixed_profs[fk]['z']
max_nz = np.max((len(mz), len(fz)))


delta_T_mixed = mixed_scalars[mk]['delta_T']
dT = np.mean(delta_T_mixed[-1000:])
Nu_fixedT = fixed_scalars[fk]['Nu']
Nu = np.mean(Nu_fixedT[-1000:])
flux_scale = (np.sqrt(ra_dt)/Nu)**(-1)

print(dT, mixed_profs[mk]['T'][0,int(len(mz)/2)])


z_basis = de.Chebyshev(  'z', max_nz, interval=[-1/2, 1/2], dealias=1)
bases = [z_basis]
domain = de.Domain(bases, grid_dtype=np.float64)

p_mixed = domain.new_field()
p_fixed = domain.new_field()

z = domain.grid(0)

fig = plt.figure()

# Set up figure subplots
fig = plt.figure(figsize=(7.5, 4))
gs = gridspec.GridSpec(1000, 1000)

subplots = [( (0 , 0),         285,    400),
            ( (330, 0),        285,    400),
            ( (660, 0),        285,    400), 
            ( (0 ,  500),      285,    250),
            ( (330, 500),      285,    250),
            ( (660, 500),      285,    250), 
            ( (0 ,  750),      285,    250),
            ( (330, 750),      285,    250),
            ( (660, 750),      285,    250), 
            ]
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]
axins_l = []
axins_r = []


bounds_out = 0.5
bounds_in  = 0.45

#Row 1 - Temp
k = 'T'
axf, axl, axr= axs[0], axs[3], axs[6]
for ax in [axf, axl, axr]:
    ax.plot(fz, fixed_profs[fk][k][0,:],    c=fColor, label='TT', lw=2)
    ax.plot(mz, mixed_profs[mk][k][0,:]/dT, c=mColor, label='FT')
#ax1.legend(loc='lower left', frameon=False, fontsize=7)
axf.set_ylabel(r'$\bar{T}/\Delta T$')
axf.legend(loc='lower center', ncol=2, frameon=False, fontsize=8)

axf.set_ylim(0, 1)
axl.set_ylim(0.5, 1)
axr.set_ylim(0, 0.5)


# Inset panels - temp diff
p_mixed.set_scales(len(mixed_profs[mk][k][0,:])/max_nz)
p_fixed.set_scales(len(fixed_profs[fk][k][0,:])/max_nz)
p_mixed['g'] = mixed_profs[mk][k][0,:]/dT
p_fixed['g'] = fixed_profs[fk][k][0,:]
for p in [p_mixed, p_fixed]: p.set_scales(1, keep_data=True)

for ax, loc in [(axl, 'upper right'), (axr, 'lower left')]:
    axins = inset_axes(ax, width='50%', height='30%', loc=loc)
    err = 100*np.abs((p_fixed['g'] - p_mixed['g'])/p_fixed['g'])
    axins.plot(z, err, c='k')
    xlim = ax.get_xbound()
    this_err = err[(z < xlim[-1])*(z > xlim[0])]
    bounds = this_err.min()/2, this_err.max()*2
    if ax is axl: 
        axins.set_ylabel('% diff', fontsize=8, labelpad=0)
        axins_l.append(axins)
        axins.set_ylim(0, 3)
        axins.set_yticks((0, 1.5, 3))
#        axins.set_ylim(0, 0.5)
#        axins.set_yticks((0, 0.25, 0.5))
    elif ax is axr: 
        axins_r.append(axins)
        axins.set_ylim(0, 2)
        axins.set_yticks((0, 1, 2))

    axins.xaxis.set_ticklabels([])

        
    if 'left' in loc:
        axins.yaxis.tick_right()
        

#Middle Row - Asymmetries
k1 = 'T-w_pos'
k2 = 'T-w_neg'

axf, axl, axr= axs[1], axs[4], axs[7]
for ax in [axf, axl, axr]:
    ax.plot(mz, mixed_asymms[mk][k1]/dT, c=mColor, lw=2)
    ax.plot(mz, mixed_asymms[mk][k2]/dT, c=mColor, ls='--', lw=2)
    ax.plot(fz, fixed_asymms[fk][k1], c=fColor, lw=1, label='Upflows')
    ax.plot(fz, fixed_asymms[fk][k2], c=fColor, ls='--', lw=1, label='Downflows')
axf.set_ylabel(r'$\bar{T}/\Delta T$')
axf.legend(loc='lower center', frameon=False, fontsize=8, ncol=2)

axf.set_ylim(0, 1)
axl.set_ylim(0.5, 1)
axr.set_ylim(0, 0.5)



#Panel 4 - Asymmetry diffs
for ax, loc in [(axl, 'upper right'), (axr, 'lower left')]:
    axins = inset_axes(ax, width='50%', height='30%', loc=loc)
    if ax is axl: 
        axins_l.append(axins)
    elif ax is axr: 
        axins_r.append(axins)

    for k, ls in [(k1, '-'), (k2, '--')]:
        p_mixed.set_scales(len(mixed_asymms[mk][k])/max_nz)
        p_fixed.set_scales(len(fixed_asymms[fk][k])/max_nz)
        p_mixed['g'] = mixed_asymms[mk][k]/dT
        p_fixed['g'] = fixed_asymms[fk][k]
        for p in [p_mixed, p_fixed]: p.set_scales(1, keep_data=True)

        good = np.abs(p_fixed['g'])/p_fixed['g'].max() > 0.1
        this_z = z[good]
        err = 100*np.abs((p_fixed['g'] - p_mixed['g'])/p_fixed['g'])[good]

        axins.plot(this_z, err, c='k', ls=ls)
        xlim = ax.get_xbound()
        this_err = err[(this_z < xlim[-1])*(this_z > xlim[0])]
        bounds = this_err.min()/2, this_err.max()*2
        if ax is axl:
            axins.set_ylim(0, 3)#this_err.min()/2, this_err.max()*2)
            axins.set_yticks((0, 1.5, 3))
        else:
            axins.set_ylim(0, 2)#this_err.min()/2, this_err.max()*2)
            axins.set_yticks((0, 1, 2))

        axins.xaxis.set_ticklabels([])

            
        if 'left' in loc:
            axins.yaxis.tick_right()



#
#
##Row 3 - fluxes
k1 = 'enth_flux'
k2 = 'kappa_flux'

axf, axl, axr= axs[2], axs[5], axs[8]
for ax in [axf, axl, axr]:
    ax.plot(mz, mixed_profs[mk][k1][0,:]/dT**(3/2)/flux_scale, c=mColor, lw=2)
    ax.plot(mz, mixed_profs[mk][k2][0,:]/dT**(3/2)/flux_scale, c=mColor, ls='--', lw=2)
    ax.plot(fz, fixed_profs[fk][k1][0,:]/flux_scale, c=fColor, lw=1, label=r'$F_{\mathrm{enth}}$')
    ax.plot(fz, fixed_profs[fk][k2][0,:]/flux_scale, c=fColor, ls='--', lw=1, label=r'$F_{\mathrm{cond}}$')
axf.set_ylabel(r'Flux$\,\cdot\,\frac{\mathrm{Nu}}{\sqrt{\mathrm{Ra}_{\Delta T}}}$')
axf.legend(loc='center', frameon=False, fontsize=8, ncol=2)


#Flux diffs
for ax, loc in [(axl, 'center right'), (axr, 'center left')]:
    axins = inset_axes(ax, width='50%', height='30%', loc=loc)
    if ax is axl: 
        axins_l.append(axins)
    elif ax is axr: 
        axins_r.append(axins)

    for k, ls in [(k1, '-')]:
        p_mixed.set_scales(len(mixed_profs[mk][k][0,:])/max_nz)
        p_fixed.set_scales(len(fixed_profs[fk][k][0,:])/max_nz)
        p_mixed['g'] = mixed_profs[mk][k][0,:]/dT**(3/2)/flux_scale
        p_fixed['g'] = fixed_profs[fk][k][0,:]/flux_scale
        for p in [p_mixed, p_fixed]: p.set_scales(1, keep_data=True)
        xlim = ax.get_xbound()

        this_z = z
        err = 100*np.abs((p_fixed['g'] - p_mixed['g'])/p_fixed['g'])
#        print(np.max(np.abs(p_fixed['g'][(this_z <= 0.45)*(this_z >= -0.45)])))
        z_bounds = (this_z <= xlim[-1])*(this_z >= xlim[0])
        good = (p_fixed['g'][z_bounds] >  0.1)

        axins.plot(this_z[z_bounds][good], err[z_bounds][good], c='k', ls=ls)
        this_err = err[z_bounds]
        bounds = this_err.min()/2, this_err.max()*2
        if ax is axl:
            axins.set_ylim(0, 2)#this_err.min()/2, this_err.max()*2)
            axins.set_yticks((0, 1, 2))
        else:
            axins.set_ylim(0, 1)#this_err.min()/2, this_err.max()*2)
            axins.set_yticks((0, 0.5, 1))
#        axins.set_yscale('log')
#        axins.set_yticks((1e-1, 1e0, 10))

        axins.xaxis.set_ticklabels([])

            
        if 'left' in loc:
            axins.yaxis.tick_right()



for i in [2, 5, 8]:
    axs[i].set_xlabel('z')

for i in [0, 1, 3, 4, 6, 7]:
    axs[i].set_xticks(())


for ax in axins_l:
    ax.set_xlim(-bounds_out, -bounds_in)
    ax.set_xticks((-bounds_out, -bounds_out + (bounds_out-bounds_in)/3, -bounds_out + 2*(bounds_out-bounds_in)/3))
for ax in axins_r:
    ax.set_xlim(-bounds_out, -bounds_in)
    ax.set_xticks((bounds_out - 2*(bounds_out-bounds_in)/3, bounds_out - (bounds_out-bounds_in)/3, bounds_out))

for i in [0, 1, 2]:
    axs[i].set_xlim(-bounds_out, bounds_out)
    axs[i].axvline(-bounds_in, c='k', lw=0.5)
    axs[i].axvline(bounds_in, c='k', lw=0.5)

for i in [3, 4, 5]:
    axs[i].set_xlim(-bounds_out, -bounds_in)
    axs[i].set_xticks((-bounds_out, -bounds_out + (bounds_out-bounds_in)/3, -bounds_out + 2*(bounds_out-bounds_in)/3))
    axs[i].get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))#matplotlib.ticker.ScalarFormatter())

    if i != 5: axs[i].set_xticklabels(())

for i in [6, 7, 8]:
    axs[i].set_xlim(bounds_in, bounds_out)
    axs[i].yaxis.tick_right()
    axs[i].set_xticks((bounds_out - 2*(bounds_out-bounds_in)/3, bounds_out - (bounds_out-bounds_in)/3, bounds_out))
    axs[i].get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))#matplotlib.ticker.ScalarFormatter())
    if i != 8: axs[i].set_xticklabels(())

for i in [0, 1]:
    axs[i].set_ylim(0, 1)
    axs[i].set_yticks((0, 0.5, 1))

for i in [2, 5, 8]:
    axs[i].set_ylim(-0.05, 1.05)



fig.savefig('rbc_1D_profiles.png', dpi=300, bbox_inches='tight')
fig.savefig('rbc_1D_profiles.pdf', dpi=300, bbox_inches='tight')
