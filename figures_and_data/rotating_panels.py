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

def read_data(ra_list, dir_list, keys=['Nu', 'delta_T', 'sim_time', 'Pe', 'KE', 'left_flux', 'right_flux', 'Ro']):
    """
    Reads scalar data in from a folder containing a series of subfolders of different Ra values.
    """
    full_data = OrderedDict()
    for ra, dr in zip(ra_list, dir_list):
        data = OrderedDict()
        sub_runs = glob.glob('{:s}/run*/'.format(dr))
        if len(sub_runs) > 0:
            numbered_dirs  = [(r, int(r.split('run')[-1].split('_')[0])) for r in sub_runs]
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

# Get data from files
base_dir = './data/rotation/'
mixed_dirs = glob.glob("{:s}mixedFT/ra*/".format(base_dir))
fixed_dirs = glob.glob("{:s}fixedT/ra*/".format(base_dir))
numbered_dirs  = [(f, float(f.split("{:s}mixedFT/ra".format(base_dir))[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}fixedT/ra".format(base_dir))[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_data = read_data(mixed_ras, mixed_dirs)
fixed_data = read_data(fixed_ras, fixed_dirs)

fk = '{:.4e}'.format(2.75e9)
mk = '{:.4e}'.format(2.1e10)


# Get rolling averages of all data; Output is every 0.1 time units
avg_window = 50 #time units
fixed_trace = fixed_data[fk]
mixed_trace = mixed_data[mk]

print(mixed_trace['sim_time'])
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
            ( (50 , 530),       400,    540),
            ( (600, 450),       350,    180),
            ( (600, 630),       350,    180),
            ( (600, 810),       350,    180),
            ]
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]
cax = plt.subplot(gs.new_subplotspec((950, 580), 50, 280))


#Show times of dynamics panels
for i in range(3):
    axs[i].axvline(100   - mixed_trace['sim_time'][-1], c='k', lw=0.5)
    axs[i].axvline(5380  - mixed_trace['sim_time'][-1], c='k', lw=0.5)
    axs[i].axvline(13215 - mixed_trace['sim_time'][-1], c='k', lw=0.5)


#Panel 1, Ra evolution
plt.sca(axs[0])
ax = axs[0]
#plt.grid(which='both')
ax.plot([1, 1], [1,1], color='olivedrab', lw=2, label='mixedFT')
plt.axhline(np.mean(mixed_trace['ra_flux'])/2.75e9, color='olivedrab', lw=1)
plt.axhline(np.mean(fixed_trace['ra_temp'])/2.75e9, color='darkorange', lw=1, ls='-.')
ax.plot(rolledm['sim_time']-mixed_trace['sim_time'][-1], rolledm['ra_temp']/2.75e9, color='olivedrab', lw=2, ls='-.', label='')
ax.plot(rolledf['sim_time']-fixed_trace['sim_time'][-1], rolledf['ra_flux']/2.75e9, color='darkorange', lw=2, label='fixedT')
ax.legend(loc='center', frameon=True, fontsize=7)
ax.set_ylabel(r'Ra/$(2.75 \times 10^9)$')
plt.xlim(-mixed_trace['sim_time'][-1], 0)
plt.ylim(0.9, 10)
plt.yscale('log')


#Panel 2, Ro evolution
plt.sca(axs[1])
ax = axs[1]


plt.plot(rolledm['sim_time']-mixed_trace['sim_time'][-1], rolledm['Ro'], color='olivedrab', lw=2, label='mixedFT')
plt.plot(rolledf['sim_time']-fixed_trace['sim_time'][-1], rolledf['Ro'], color='darkorange', lw=2, label='fixedT')
plt.axhline(1, c='darkorange', lw=1)
plt.yscale('log')
plt.xlim(-mixed_trace['sim_time'][-1], 0)
plt.ylim(0.08, 2)
ax.set_ylabel('Ro')
ax.legend(loc='upper right', frameon=True, fontsize=7)


#Panel 3, Pe evolution
plt.sca(axs[2])
ax = axs[2]


Pe_final_temp = np.mean(fixed_trace['Pe'][-5000:])
plt.plot(mixed_trace['sim_time']-mixed_trace['sim_time'][-1], mixed_trace['Pe']/Pe_final_temp, color='olivedrab', lw=2, label='mixedFT')
plt.plot(fixed_trace['sim_time']-fixed_trace['sim_time'][-1], fixed_trace['Pe']/Pe_final_temp, color='darkorange', lw=2, label='fixedT')
plt.axhline(1, c='darkorange', lw=1)
plt.xlim(-mixed_trace['sim_time'][-1], 0)
ax.set_ylabel(r'Pe/Pe$_{\Delta T}$')
plt.yscale('log')
ax.set_xlabel(r'$t - t_{\mathrm{final}}$')
ax.legend(loc='upper right', frameon=True, fontsize=7)
plt.ylim(0.7, 20)



#Panel 4, Nu v. Ra
plt.sca(axs[3])
ax = axs[3]

#Literature
cheng_sims = np.genfromtxt('./data/rotation/cheng2015_tableA2.csv', delimiter=',', skip_header=1)
cheng_data = np.genfromtxt('./data/rotation/cheng2015_tableA1.csv', delimiter=',', skip_header=1)
plt.plot(cheng_data[:,1][np.isinf(cheng_data[:,0])], cheng_data[:,2][np.isinf(cheng_data[:,0])], c='k', lw=0, marker='d', ms=3, label='Cheng+2015 Experiments')
for Ek, c in [(1e-5, 'yellowgreen'), (1e-6, 'forestgreen'), (1e-7, 'teal')]:
    plt.plot(cheng_sims[:,1][cheng_sims[:,0] == Ek], cheng_sims[:,2][cheng_sims[:,0] == Ek], c=c, lw=0, marker='o', ms=3)
plt.plot([1,1], [10,10], c='k', lw=0, marker='o', ms=3, label='Cheng+2015 Sims')

for Ekmax, Ekmin, c in [(4e-6, 2e-6, 'forestgreen'), (1.5e-7, 6e-8, 'teal'), (3e-8, 1e-8, 'indigo')]:
    good = (cheng_data[:,0] < Ekmax)*(cheng_data[:,0] > Ekmin)
    plt.plot(cheng_data[:,1][good], cheng_data[:,2][good], c=c, lw=0, marker='d', ms=3)

#zhu_ra = [1e8,  2.15e8, 4.64e8, 1e9,  2.15e9, 4.64e9, 1e10, 2.15e10, 4.64e10]
#zhu_nu = [26.1, 31.2,   38.9,   48.3, 61.1,   76.3,   95.1, 120.1,   152.2]
#plt.plot(zhu_ra, zhu_nu, marker='x', ms=3, c='black', label='Zhu+18', lw=0)


ra_trace = np.logspace(7, 13, 100)
nu_func = lambda ra: 0.16*ra**(0.284)
plt.plot(ra_trace, nu_func(ra_trace), c='k')


#Our data
ra_trace = np.logspace(8, 11, 100)
nu_func = lambda ra: 0.138*np.array(ra)**(0.285) #Johnston & Doering 2009
for k, data in mixed_data.items():
    df = pd.DataFrame(data=data)
    rolled = df.rolling(window=1000, min_periods=1000).mean()
    label='mixedFT'
    plt.arrow(0.525, 0.73, -0.001, -0.04,transform=ax.transAxes,\
                 head_width=0.04, head_length=0.05, color='olivedrab', facecolor='olivedrab', rasterized='True', zorder=np.inf)
    plt.arrow(0.52, 0.635, -0.01, -0.017,transform=ax.transAxes,\
                 head_width=0.04, head_length=0.05, color='olivedrab', facecolor='olivedrab', rasterized='True', zorder=np.inf)
    plt.plot(rolled['ra_temp'], rolled['Nu'], color='olivedrab', label=label, lw=3)
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
plt.errorbar(my_ra, my_nu, yerr=(my_nu_sampleMean), lw=0, elinewidth=1, capsize=1.5, c='darkorange', ms=10, marker='*', label='fixedT')

handles, labels = ax.get_legend_handles_labels()
order = [0, 1, 2, 3]
#plt.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower right', fontsize=7)
plt.xscale('log')
plt.xlabel(r'Ra$_{\Delta T}$')
plt.ylabel(r'Nu')
plt.yscale('log')
plt.xlim(3e7, 3e12)
#plt.xlim(9e7, 2e9)#3e10)
#plt.ylim(0.5, 1.5)


#Panel 5, Colormap 1 

early_slice_f = 'data/rotation/mixedFT/ra2.1e10/slices/slices_t100_512x384x384_s1.h5'
mid_slice_f = 'data/rotation/mixedFT/ra2.1e10/slices/slices_t5380_128x384x384_s1.h5'
late_slice_f = 'data/rotation/mixedFT/ra2.1e10/slices/slices_t13215_128x384x384_s1.h5'
for i, filename, t in [(4, early_slice_f, r'100'), (5, mid_slice_f, r'5400'), (6, late_slice_f, r'13,200')]:
    plt.sca(axs[i])
    ax = axs[i]
    with h5py.File(filename, 'r') as f:
        vorticity = f['tasks']['vort_z integ'][0,:].squeeze()
        x, y = f['scales/x/1.0'], f['scales/y/1.0']
        yy, xx = np.meshgrid(y, x)
        maxv = np.abs(vorticity.max())
        c = plt.pcolormesh(xx, yy, vorticity, cmap='PuOr_r', vmin=-maxv, vmax=maxv, rasterized=True)
        plt.text(0.02, 0.05, r'$\omega_{{\mathrm{{max}}}} = {:.2f}$'.format(maxv), transform=ax.transAxes)
        plt.text(0.02, 0.88, r'$t \sim {}$'.format(t), transform=ax.transAxes)

bar = plt.colorbar(c, cax=cax, orientation='horizontal')
cax.set_xticklabels(())
bar.set_ticks(())
cax.text(0.4, -0.75, r'$\pm\omega_{\mathrm{max}}$', transform=cax.transAxes)



#Reporting
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
for i in [0, 1]:
    axs[i].tick_params(labelbottom=False)
    axs[i].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs[i].get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

for i in [4, 5, 6]:
    axs[i].set_xticks(())
    axs[i].set_yticks(())





fig.savefig('rotating_panels.png', dpi=300, bbox_inches='tight')
fig.savefig('rotating_panels.pdf', dpi=300, bbox_inches='tight')
