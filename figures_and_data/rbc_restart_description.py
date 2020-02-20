import glob
import matplotlib
from collections import OrderedDict
#matplotlib.use('Agg')
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

def read_data(ra_list, dir_list, keys=['Nu', 'delta_T', 'sim_time', 'Pe', 'KE', 'left_flux', 'right_flux', 'left_T', 'right_T']):
    """
    Reads scalar data in from a folder containing a series of subfolders of different Ra values.
    """
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



# Get data from files
base_dir = './data/rbc/'
mixed_dirs = glob.glob("{:s}mixedFT_2d/ra*/".format(base_dir))
fixed_dirs = glob.glob("{:s}fixedT_2d/ra*/".format(base_dir))
restarted_dirs = glob.glob("{:s}restarted_mixed_T2m/ra*/".format(base_dir))
numbered_dirs  = [(f, float(f.split("{:s}mixedFT_2d/ra".format(base_dir))[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}fixedT_2d/ra".format(base_dir))[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}restarted_mixed_T2m/ra".format(base_dir))[-1].split("/")[0])) for f in restarted_dirs]
restarted_dirs, restarted_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_data = read_data(mixed_ras, mixed_dirs)
fixed_data = read_data(fixed_ras, fixed_dirs)
restarted_data = read_data(restarted_ras, restarted_dirs)
mixed_pdfs = read_pdfs(mixed_ras, mixed_dirs)
fixed_pdfs = read_pdfs(fixed_ras, fixed_dirs)
restarted_pdfs = read_pdfs(restarted_ras, restarted_dirs)

#fk = '{:.4e}'.format(1.00e8)
#mk = '{:.4e}'.format(2.61e9)
#rk = '{:.4e}'.format(2.61e9)

fk = '{:.4e}'.format(1.00e9)
mk = '{:.4e}'.format(4.83e10)
rk = '{:.4e}'.format(4.83e10)

Nu_approx=48.3

# Get rolling averages of all data; Output is every 0.1 time units
avg_window = 25 #time units
mixed_trace = mixed_data[mk]
fixed_trace = fixed_data[fk]
partial_restarted_trace = restarted_data[rk]

restarted_trace = OrderedDict()

good_fixed_times = fixed_trace['sim_time'] <= partial_restarted_trace['sim_time'][0]

for k in fixed_trace.keys():
    fixed = fixed_trace[k][good_fixed_times]
    if k == 'left_flux' or k == 'right_flux':
        restarted_trace[k] = np.concatenate((fixed/Nu_approx**(3./2), partial_restarted_trace[k][1:])) 
    elif 'T' in k:
        restarted_trace[k] = np.concatenate((fixed, partial_restarted_trace[k][1:]*Nu_approx)) 
    else:
        restarted_trace[k] = np.concatenate((fixed, partial_restarted_trace[k][1:])) 

dff = pd.DataFrame(data=fixed_trace)
rolledf = dff.rolling(window=avg_window*10, min_periods=avg_window*10).mean()
dfm = pd.DataFrame(data=mixed_trace)
rolledm = dfm.rolling(window=avg_window*10, min_periods=avg_window*10).mean()
dfr = pd.DataFrame(data=restarted_trace)
rolledr = dfr.rolling(window=avg_window*10, min_periods=avg_window*10).mean()



# Set up figure subplots
fig = plt.figure(figsize=(7.5, 4))
gs = gridspec.GridSpec(1000, 1000)

subplots = [( (50 , 0  ),       300,    330),
            ( (350, 0  ),       300,    330),
            ( (650, 0  ),       300,    330),
            ( (50 , 400),       450,    300),
            ( (50 , 700),       450,    300),
            ( (500 , 400),       450,    300),
            ( (500 , 700),       450,    300),
            ]
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]
for i in [0, 1, 2]:
    axs[i].fill_between([restarted_trace['sim_time'][-1]-500, restarted_trace['sim_time'][-1]], 1e-2, 10, alpha=0.2, color='grey')
    axs[i].axvline(partial_restarted_trace['sim_time'][0], c='k', lw=0.5)


#Panel 1, Ra evolution
plt.sca(axs[0])
ax = axs[0]
#plt.grid(which='both')
plt.plot(rolledr['sim_time'], rolledr['ra_flux']/(1e9*Nu_approx), color='black', lw=1, ls='-.', label=r'Ra$_{\partial_z T}/48.3\,\,$')#\langle\mathrm{Nu}\rangle$')
plt.plot(rolledr['sim_time'], rolledr['ra_temp']/(1e9), color='black', lw=1, label=r'Ra$_{\Delta T}$')
ax.legend(loc='lower center', borderpad=0.25, fontsize=7, ncol=2)
ax.set_ylabel(r'Ra/$10^9$')
plt.ylim(0.9, 1.1)
ax.set_yticks((0.95, 1, 1.05))
plt.yscale('log')


#Panel 2, bot flux 
plt.sca(axs[1])
ax = axs[1]

Nu_final_temp = np.mean(fixed_trace['Nu'][-5000:])
#plt.plot(rolledm['sim_time']-mixed_trace['sim_time'][-1],     rolledm['left_flux']*(np.sqrt(2.61e9)), color='olivedrab', lw=1)
plt.plot(rolledr['sim_time'], rolledr['left_flux']*np.sqrt(1e9*Nu_approx), color='black', lw=1)
print(rolledr['left_flux'])
#plt.axhline(1, c='black', lw=1)
ax.set_ylabel(r'bot $\frac{\mathrm{Pe}_{\mathrm{ff}}^{-1}|\partial_z T|}{\mathrm{Flux}}$')
ax.set_ylim(0.925, 1.075)
ax.set_yticks((0.95, 1, 1.05))

#Panel 3, right T
plt.sca(axs[2])
ax = axs[2]


plt.plot(rolledr['sim_time'], rolledr['left_T'], color='black', lw=1)
#plt.axhline(1, c='black', lw=1)
ax.set_ylabel(r'bot $T/\Delta T$')
#plt.yscale('log')
ax.set_xlabel(r'$t$')
ax.set_ylim(0.925, 1.075)
ax.set_yticks((0.95, 1, 1.05))
print(rolledr['left_T'])

keys = ['T', 'enstrophy', 'enth_flux', 'w']
labels = [r'$T/\Delta T$', r'$\omega^2 / 10^3$', r'$w T\cdot 10^3$', r'$w$']
for i, field in enumerate(keys):
    ax = axs[3+i] 
    mx, mp, mdx = [mixed_pdfs[mk][field][k] for k in ['xs', 'pdf', 'dx']]
    rx, rp, rdx = [restarted_pdfs[rk][field][k] for k in ['xs', 'pdf', 'dx']]

    ax.fill_between(mx, 1e-16, mp, color='olivedrab', alpha=0.5)
    ax.fill_between(rx, 1e-16, rp, color='black', alpha=0.5)
    ax.plot(mx, mp, label='FT', color='olivedrab')
    ax.plot(rx, rp, label='TT-to-FT', color='black')
    ax.set_yscale('log')
    ax.set_xlabel(labels[i])
    ax.set_ylim(np.min((mp[mp>0].min(), rp[rp>0].min())), np.max((mp.max(), rp.max())))
    ax.set_xlim(np.min((mx.min(), rx.min())), np.max((mx.max(), rx.max())))
    if field == 'enstrophy':
        ax.legend(loc='best', frameon=False)

axs[3].set_xticks((0, 0.5/Nu_approx, 1/Nu_approx))
axs[3].set_xticklabels((0, 0.5, 1))
axs[4].set_xticks((0, 1e3, 2e3, 3e3, 4e3, 5e3))
axs[4].set_xticklabels((0, 1, 2, 3, 4, 5))
axs[5].set_xticks((-2e-3, -1e-3, 0, 1e-3, 2e-3))
axs[5].set_xticklabels((-2, -1, 0, 1, 2))
axs[6].set_xticks((-0.2, -0.1, 0, 0.1, 0.2, ))

for i in [3, 4]:
    axs[i].xaxis.set_ticks_position('top')
    axs[i].xaxis.set_label_position('top')
for i in [4, 6]:
    axs[i].yaxis.set_ticks_position('right')
    axs[i].yaxis.set_label_position('right')

#Get rid of bad tick labels, etc.
for i in [0, 1]:
    axs[i].tick_params(labelbottom=False)
    axs[i].set_yticks((0.95, 1, 1.05))
    axs[i].minorticks_off()
    axs[i].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs[i].get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

for i in [0, 1, 2]:
    axs[i].set_xlim(0, restarted_trace['sim_time'][-1])

for i, field in enumerate(keys):
    mx, mp, mdx = [mixed_pdfs[mk][field][k] for k in ['xs', 'pdf', 'dx']]
    rx, rp, rdx = [restarted_pdfs[rk][field][k] for k in ['xs', 'pdf', 'dx']]

    print(field, " FT, then TT-to-FT")
    for x, p, dx in ((mx, mp, mdx), (rx, rp, rdx)):
        mean = np.sum(x*p*dx)
        sigma= np.sqrt(np.sum((x-mean)**2*p*dx))
        skew = (1/sigma**3)*np.sum((x-mean)**3*p*dx)
        kurt = (1/sigma**4)*np.sum((x-mean)**4*p*dx)
        print('mean: {:.4e}, sigma: {:.4e}, skew: {:.4e}, kurt: {:.4e}'.format(mean, sigma, skew, kurt))





fig.savefig('rbc_restart_description.png', dpi=300, bbox_inches='tight')
fig.savefig('rbc_restart_description.pdf', dpi=300, bbox_inches='tight')
