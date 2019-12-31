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

def read_data(ra_list, dir_list, keys=['Nu', 'delta_T', 'sim_time', 'Pe', 'KE', 'left_flux', 'right_flux']):
    """
    Reads scalar data in from a folder containing a series of subfolders of different Ra values.
    """
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

# Get data from files
base_dir = './data/rbc/'
mixed_dirs = glob.glob("{:s}mixedFT_2d/ra*/".format(base_dir))
fixed_dirs = glob.glob("{:s}fixedT_2d/ra*/".format(base_dir))
numbered_dirs  = [(f, float(f.split("{:s}mixedFT_2d/ra".format(base_dir))[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}fixedT_2d/ra".format(base_dir))[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_data = read_data(mixed_ras, mixed_dirs)
fixed_data = read_data(fixed_ras, fixed_dirs)

# Get rolling averages of all data; Output is every 0.1 time units
avg_window = 100 #time units
fixed_trace = fixed_data['{:.4e}'.format(1.00e8)]
mixed_trace = mixed_data['{:.4e}'.format(2.61e9)]
dff = pd.DataFrame(data=fixed_trace)
rolledf = dff.rolling(window=avg_window*10, min_periods=avg_window*10).mean()
dfm = pd.DataFrame(data=mixed_trace)
rolledm = dfm.rolling(window=avg_window*10, min_periods=avg_window*10).mean()



# Set up figure subplots
fig = plt.figure(figsize=(7.5, 4))
gs = gridspec.GridSpec(1000, 1000)

subplots = [( (50 , 0  ),       300,    430),
            ( (350, 0  ),       300,    430),
            ( (650, 0  ),       300,    430),
            ( (50 , 570),       450,    430),
            ( (500, 570),       450,    430)
            ]
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]


#Panel 1, Ra evolution
plt.sca(axs[0])
ax = axs[0]
#plt.grid(which='both')
plt.axhline(np.mean(mixed_trace['ra_flux'])/2.61e9, color='olivedrab', lw=1)
plt.axhline(np.mean(fixed_trace['ra_temp'])/2.61e9, color='darkorange', lw=1, ls='-.')
plt.plot(rolledm['sim_time']-mixed_trace['sim_time'][-1], rolledm['ra_temp']/2.61e9, color='olivedrab', lw=2, ls='-.')
plt.plot(rolledf['sim_time']-fixed_trace['sim_time'][-1], rolledf['ra_flux']/2.61e9, color='darkorange', lw=2)
ax.set_ylabel(r'Ra/$2.61\times 10^9$')
plt.xlim(-mixed_trace['sim_time'][-1], 0)
plt.yscale('log')


##Approx function for evolution of Ra
#Nu_final_temp = np.mean(fixed_trace['Nu'][-5000:])
#t_S = np.sqrt(2.61e9 / 1298)
#t_best = t_S / 2
#t_kappa = 5e4 / (Nu_final_temp/2)**2
##plt.plot(mixed_trace['sim_time']-mixed_trace['sim_time'][-1], (1/26.1)+(1/2-1/26.1)*np.exp(-mixed_trace['sim_time']/t_kappa), c='k', lw=0.5) #1418
##plt.plot(mixed_trace['sim_time']-mixed_trace['sim_time'][-1], (1/26.1)+(1/2-1/26.1)*np.exp(-mixed_trace['sim_time']/t_S)    , c='k', lw=0.5) #1418
#plt.plot(mixed_trace['sim_time']-mixed_trace['sim_time'][-1], (1/26.1)+(1/2-1/26.1)*np.exp(-mixed_trace['sim_time']/t_best) , c='k', lw=0.5)

plt.text(-1500, 0.05, r'Ra$_{\Delta T}$')
plt.text(-2500, 0.58, r'Ra$_{\partial_z T}$')


#Panel 2, Nu evolution
plt.sca(axs[1])
ax = axs[1]

Nu_final_temp = np.mean(fixed_trace['Nu'][-5000:])

plt.plot(rolledm['sim_time']-mixed_trace['sim_time'][-1], rolledm['Nu']/Nu_final_temp, color='olivedrab', lw=2)
plt.plot(rolledf['sim_time']-fixed_trace['sim_time'][-1], rolledf['Nu']/Nu_final_temp, color='darkorange', lw=2)
plt.axhline(1, c='darkorange', lw=1)
plt.yscale('log')
plt.xlim(-mixed_trace['sim_time'][-1], 0)
ax.set_ylabel(r'Nu/Nu$_{\Delta T}$')

#Panel 3, Pe evolution
plt.sca(axs[2])
ax = axs[2]


Pe_final_temp = np.mean(fixed_trace['Pe'][-5000:])
plt.plot(mixed_trace['sim_time']-mixed_trace['sim_time'][-1], mixed_trace['Pe']/Pe_final_temp, color='olivedrab', lw=2)
plt.plot(fixed_trace['sim_time']-fixed_trace['sim_time'][-1], fixed_trace['Pe']/Pe_final_temp, color='darkorange', lw=2)
plt.axhline(1, c='darkorange', lw=1)
plt.xlim(-mixed_trace['sim_time'][-1], 0)
ax.set_ylabel(r'Pe/Pe$_{\Delta T}$')
#plt.yscale('log')
ax.set_yticks((1, 2, 3, 4))
ax.set_xlabel(r'$t - t_{\mathrm{final}}$')
plt.ylim(0.5, 5)


#Panel 4, Nu v. Ra
plt.sca(axs[3])
ax = axs[3]

zhu_ra = [1e8,  2.15e8, 4.64e8, 1e9,  2.15e9, 4.64e9, 1e10, 2.15e10, 4.64e10]
zhu_nu = [26.1, 31.2,   38.9,   48.3, 61.1,   76.3,   95.1, 120.1,   152.2]

ra_trace = np.logspace(8, 11, 100)
nu_func = lambda ra: 0.138*np.array(ra)**(0.285) #Johnston & Doering 2009
nu_guess = nu_func(ra_trace) 
plt.axhline(1, c='k', lw=0.5)#plt.plot(ra_trace, nu_guess/nu_guess, color='k')#, label='J&D09')
for k, data in mixed_data.items():
    df = pd.DataFrame(data=data)
    rolled = df.rolling(window=1000, min_periods=1000).mean()
    label='Dedalus-mixedFT'
    plt.arrow(0.835, 0.85, -0.003, -0.16,transform=ax.transAxes,\
                 head_width=0.025, head_length=0.05, color='olivedrab', facecolor='olivedrab', rasterized='True')
    plt.arrow(0.838, 0.95, -0.003, -0.1,transform=ax.transAxes,\
                 head_width=0.025, head_length=0.05, color='olivedrab', facecolor='olivedrab', rasterized='True')
    plt.plot(rolled['ra_temp'], rolled['Nu']/nu_func(rolled['ra_temp']), color='olivedrab', zorder=0, label=label)
plt.scatter(zhu_ra, zhu_nu/nu_func(zhu_ra), marker='x', s=80, c='black', label='Zhu+18')
my_ra = []
my_nu = []
my_nu_sampleMean = []
N = 5000
for ra, data in fixed_data.items():
    nu = np.mean(data['Nu'][-N:])
    stdev = np.std(data['Nu'][-N:])
    my_ra.append(float(ra))
    my_nu.append(nu)
    my_nu_sampleMean.append(stdev/np.sqrt(N))
print(my_nu, my_nu_sampleMean)
plt.errorbar(my_ra[1:], (my_nu/nu_func(my_ra))[1:], yerr=(my_nu_sampleMean/nu_func(my_ra))[1:], lw=0, elinewidth=1, capsize=1.5, c='darkorange', ms=5, marker='o', label='Dedalus-fixedT')
plt.errorbar(my_ra[0], (my_nu/nu_func(my_ra))[0], yerr=(my_nu_sampleMean/nu_func(my_ra))[0], lw=0, elinewidth=1, capsize=1.5, c='darkorange', ms=7, marker='*')
#plt.scatter(my_ra, my_nu/nu_func(my_ra), s=15, c='darkorange', marker='o', label='Dedalus-fixedT')
plt.legend(loc='best', fontsize=7)
plt.xscale('log')
plt.xlabel(r'Ra$_{\Delta T}$')
plt.ylabel(r'Nu/(0.138 Ra$_{\Delta T}^{0.285}$)')
plt.xlim(9e7, 2e9)#3e10)
plt.ylim(0.5, 1.5)


#Panel 5, Pe v. Ra
plt.sca(axs[4])
ax = axs[4]

pe_func = lambda ra: 0.45*np.array(ra)**(0.5) #Ahlers&all 2009 (prefactor mine)
#nu_guess = 0.16*ra_trace**(0.284) #Julien 2016
pe_guess = pe_func(ra_trace) 
plt.axhline(1, c='k', lw=0.5)#plt.plot(ra_trace, pe_guess/pe_guess, color='k')#, label=r'Ra$^{1/2}$')
for k, data in mixed_data.items():
    df = pd.DataFrame(data=data)
    rolled = df.rolling(window=1000, min_periods=1000).mean()
    label='Dedalus-mixedFT'
    plt.arrow(0.925, 0.2, -0.04, 0.16,transform=ax.transAxes,\
                 head_width=0.025, head_length=0.05, color='olivedrab', facecolor='olivedrab', rasterized='True')
    plt.arrow(0.937, 0.15, -0.013, 0.05,transform=ax.transAxes,\
                 head_width=0.025, head_length=0.05, color='olivedrab', facecolor='olivedrab', rasterized='True')
    plt.plot(rolled['ra_temp'], rolled['Pe']/pe_func(rolled['ra_temp']), color='olivedrab', zorder=0, label=label)
my_ra = []
my_pe = []
my_pe_sampleMean = []
N = 5000
for ra, data in fixed_data.items():
    pe = np.mean(data['Pe'][-N:])
    stdev = np.std(data['Pe'][-N:])
    my_ra.append(float(ra))
    my_pe.append(pe)
    my_pe_sampleMean.append(stdev/np.sqrt(N))

#Need to make the ones that aren't a comparison case a different color
plt.errorbar(my_ra[1:], (my_pe/pe_func(my_ra))[1:], yerr=(my_pe_sampleMean/pe_func(my_ra))[1:], lw=0, elinewidth=1, capsize=1.5, c='darkorange', ms=5, marker='o', label='Dedalus-fixedT')
plt.errorbar(my_ra[0], (my_pe/pe_func(my_ra))[0], yerr=(my_pe_sampleMean/pe_func(my_ra))[0], lw=0, elinewidth=1, capsize=1.5, c='darkorange', ms=7, marker='*', label='Dedalus-fixedT')
#plt.scatter(my_ra, my_pe/pe_func(my_ra), s=15, c='darkorange', marker='o', label='Dedalus-fixedT')
plt.xscale('log')
plt.xlabel(r'Ra$_{\Delta T}$')
plt.ylabel(r'Pe/(0.45 Ra$_{\Delta T}^{0.5}$)')
plt.xlim(9e7, 2e9)#3e10)
plt.ylim(0.5, 1.5)

for i, ra in enumerate(my_ra):
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(ra, my_nu[i], my_nu_sampleMean[i], my_pe[i], my_pe_sampleMean[i]))
print('-----------------------------------------')
for ra, data in mixed_data.items():
    nu = np.mean(data['Nu'][-N:])
    nu_stdev = np.std(data['Nu'][-N:])
    pe = np.mean(data['Pe'][-N:])
    pe_stdev = np.std(data['Pe'][-N:])
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(float(ra), nu, nu_stdev/np.sqrt(N), pe, pe_stdev/np.sqrt(N)))
    



#Get rid of bad tick labels, etc.
for i in [0, 1, 3]:
    axs[i].tick_params(labelbottom=False)
    axs[i].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs[i].get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())





fig.savefig('rbc_scalar_comparisons.png', dpi=300, bbox_inches='tight')
fig.savefig('rbc_scalar_comparisons.pdf', dpi=300, bbox_inches='tight')
