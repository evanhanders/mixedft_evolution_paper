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

mColor  = 'darkorange'#olivedrab'
mColor2 = 'gold'
fColor  = 'indigo'

def read_data(ra_list, dir_list, keys=['Nu', 'delta_T', 'sim_time', 'Pe', 'KE', 'left_flux', 'right_flux']):
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

# Get data from files
base_dir = './data/rbc/'
mixed_dirs = glob.glob("{:s}classic_FT_2D/ra*/".format(base_dir))
fixed_dirs = glob.glob("{:s}TT_2D/ra*/".format(base_dir))
restarted_dirs = glob.glob("{:s}TT-to-FT_2D/ra*/".format(base_dir))
numbered_dirs  = [(f, float(f.split("{:s}classic_FT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}TT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}TT-to-FT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in restarted_dirs]
restarted_dirs, restarted_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_data = read_data(mixed_ras, mixed_dirs)
fixed_data = read_data(fixed_ras, fixed_dirs)
restarted_data = read_data(restarted_ras, restarted_dirs)

# Get rolling averages of all data; Output is every 0.1 time units
avg_window = 100 #time units
fixed_trace = fixed_data['{:.4e}'.format(1.00e9)]
mixed_trace = mixed_data['{:.4e}'.format(4.83e10)]

good = mixed_trace['sim_time'] <= 1e4
for k in mixed_trace:
    mixed_trace[k] = mixed_trace[k][good]
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
ax.plot([1, 1], [1,1], color=mColor, lw=1, label='FT')
ax.plot([1, 1], [1,1], color=fColor, lw=1, label='TT')
ax.plot([1, 1], [1,1], color='k', lw=1, label=r'Ra$_{\Delta T}$')
ax.plot([1, 1], [1,1], color='k', lw=1, label=r'Ra$_{\partial_z T}$', dashes=(3,1,1,1))
plt.axhline(np.mean(mixed_trace['ra_flux'])/1e9, color=mColor, lw=1, dashes=(3,1,1,1))
plt.axhline(np.mean(fixed_trace['ra_temp'])/1e9, color=fColor, lw=1)
ax.plot(rolledf['sim_time']-fixed_trace['sim_time'][-1], rolledf['ra_flux']/1e9, color=fColor, lw=1, label='', dashes=(3,1,1,1))
ax.plot(rolledm['sim_time']-mixed_trace['sim_time'][-1], rolledm['ra_temp']/1e9, color=mColor, lw=1, label='')
ax.legend(loc='center right', frameon=True, fontsize=7, ncol=2)
ax.set_ylabel(r'Ra/$10^9$')
plt.xlim(-mixed_trace['sim_time'][-1], 0)
plt.yscale('log')
plt.ylim(0.8, 60)
plt.xlim(-mixed_trace['sim_time'][-1], 0)



##Approx function for evolution of Ra
#Nu_final_temp = np.mean(fixed_trace['Nu'][-5000:])
#t_S = np.sqrt(2.61e9 / 1298)
#t_best = t_S / 2
#t_kappa = 5e4 / (Nu_final_temp/2)**2
##plt.plot(mixed_trace['sim_time']-mixed_trace['sim_time'][-1], (1/26.1)+(1/2-1/26.1)*np.exp(-mixed_trace['sim_time']/t_kappa), c='k', lw=0.5) #1418
##plt.plot(mixed_trace['sim_time']-mixed_trace['sim_time'][-1], (1/26.1)+(1/2-1/26.1)*np.exp(-mixed_trace['sim_time']/t_S)    , c='k', lw=0.5) #1418
#plt.plot(mixed_trace['sim_time']-mixed_trace['sim_time'][-1], (1/26.1)+(1/2-1/26.1)*np.exp(-mixed_trace['sim_time']/t_best) , c='k', lw=0.5)

#plt.text(-1500, 1.3, r'Ra$_{\Delta T}$')
#plt.text(-2500, 15, r'Ra$_{\partial_z T}$')


#Panel 2, Nu evolution
plt.sca(axs[1])
ax = axs[1]

Nu_final_temp = np.mean(fixed_trace['Nu'][-5000:])

plt.plot(rolledm['sim_time']-mixed_trace['sim_time'][-1], rolledm['Nu']/Nu_final_temp, color=mColor, lw=1, label='FT')
plt.plot(rolledf['sim_time']-fixed_trace['sim_time'][-1], rolledf['Nu']/Nu_final_temp, color=fColor, lw=1, label='TT')
plt.axhline(1, c=fColor, lw=0.5)
plt.yscale('log')
plt.xlim(-mixed_trace['sim_time'][-1], 0)
plt.ylim(0.7, 3)
ax.set_ylabel(r'Nu/Nu$_{\Delta T}$')
ax.legend(loc='upper right', frameon=True, fontsize=7)


#Panel 3, Pe evolution
plt.sca(axs[2])
ax = axs[2]


Pe_final_temp = np.mean(fixed_trace['Pe'][-5000:])
plt.plot(mixed_trace['sim_time']-mixed_trace['sim_time'][-1], mixed_trace['Pe']/Pe_final_temp, color=mColor, lw=1, label='FT')
plt.plot(fixed_trace['sim_time']-fixed_trace['sim_time'][-1], fixed_trace['Pe']/Pe_final_temp, color=fColor, lw=1, label='TT')
plt.axhline(1, c=fColor, lw=0.5)
plt.xlim(-mixed_trace['sim_time'][-1], 0)
ax.set_ylabel(r'Pe/Pe$_{\Delta T}$')
#plt.yscale('log')
ax.set_yticks((1, 2, 3, 4, 5, 6))
ax.set_xlabel(r'$t - t_{\mathrm{final}}$')
ax.legend(loc='upper right', frameon=True, fontsize=7)
plt.ylim(0.5, 7)


#Panel 4, Nu v. Ra
plt.sca(axs[3])
ax = axs[3]
this_xlim = (9e7, 4e10)
this_ylim = (0.5, 1.5)

zhu_ra = [1e8,  2.15e8, 4.64e8, 1e9,  2.15e9, 4.64e9, 1e10]
zhu_nu = [26.1, 31.2,   38.9,   48.3, 61.1,   76.3,   95.1]

ra_trace = np.logspace(8, 11, 100)
nu_func = lambda ra: 0.138*np.array(ra)**(0.285) #Johnston & Doering 2009
nu_guess = nu_func(ra_trace) 
plt.axhline(1, c='k', lw=0.5, zorder=2000)#plt.plot(ra_trace, nu_guess/nu_guess, color='k')#, label='J&D09')
for k, data in mixed_data.items():
    df = pd.DataFrame(data=data)
    rolled = df.rolling(window=avg_window*10, min_periods=avg_window*10).mean()
    label='FT'

    if k == '{:.4e}'.format(2.61e9): 
        this_color=mColor2
        xy_points = [340, 353, 351, 363]
    else: 
        this_color=mColor
        xy_points = [365, 369, 370, 375]
    coords = []
    for i in xy_points:
        x = rolled['ra_temp'][avg_window*10 + i]
        y = (rolled['Nu']/nu_func(rolled['ra_temp']))[avg_window*10 + i]

        coords.append(( (np.log10(x)-np.log10(this_xlim[0]))/(np.log10(this_xlim[1])-np.log10(this_xlim[0])), 
                        (y-this_ylim[0])/(this_ylim[1]-this_ylim[0]) ))

    plt.arrow(coords[0][0], coords[0][1], coords[1][0]-coords[0][0], coords[1][1]-coords[0][1], transform=ax.transAxes,\
                 head_width=0.025, head_length=0.05, color=this_color, facecolor=mColor, rasterized='True')
    plt.arrow(coords[2][0], coords[2][1], coords[3][0]-coords[2][0], coords[3][1]-coords[2][1], transform=ax.transAxes,\
                 head_width=0.025, head_length=0.05, color=this_color, facecolor=mColor, rasterized='True')
    if k == '{:.4e}'.format(2.61e9):
        plt.plot(rolled['ra_temp'], rolled['Nu']/nu_func(rolled['ra_temp']), color=mColor2, zorder=0, label='')
    else:
        plt.plot(rolled['ra_temp'], rolled['Nu']/nu_func(rolled['ra_temp']), color=mColor, label=label)
plt.scatter(zhu_ra, zhu_nu/nu_func(zhu_ra), marker='x', s=80, c='black', label='Zhu+18', zorder=1000)
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
plt.errorbar(my_ra[1:], (my_nu/nu_func(my_ra))[1:], yerr=(my_nu_sampleMean/nu_func(my_ra))[1:], lw=0, elinewidth=1, capsize=1.5, c=fColor, ms=5, marker='o', label='TT', zorder=1001)
plt.errorbar(my_ra[0], (my_nu/nu_func(my_ra))[0], yerr=(my_nu_sampleMean/nu_func(my_ra))[0], lw=0, elinewidth=2, capsize=1.5, markeredgecolor=fColor, c=mColor2, markeredgewidth=3, ms=6, marker='o', zorder=1001)
plt.errorbar(my_ra[3], (my_nu/nu_func(my_ra))[3], yerr=(my_nu_sampleMean/nu_func(my_ra))[3], lw=0, elinewidth=2, capsize=1.5, markeredgecolor=fColor, c=mColor,  markeredgewidth=3, ms=6, marker='o', zorder=1001)

handles, labels = ax.get_legend_handles_labels()
order = [0, 2, 1]#len(mixed_data.keys())+1, len(mixed_data.keys())]
plt.legend([handles[i] for i in order], [labels[i] for i in order], loc='best', fontsize=7, markerfirst=False)
plt.xscale('log')
plt.xlabel(r'Ra$_{\Delta T}$')
plt.ylabel(r'Nu/(0.138 Ra$_{\Delta T}^{0.285}$)')
plt.xlim(this_xlim)
plt.ylim(this_ylim)


#Panel 5, Pe v. Ra
plt.sca(axs[4])
ax = axs[4]

pe_func = lambda ra: 0.43*np.array(ra)**(0.5) #Ahlers&all 2009 (prefactor mine)
#nu_guess = 0.16*ra_trace**(0.284) #Julien 2016
pe_guess = pe_func(ra_trace) 
plt.axhline(1, c='k', lw=0.5)#plt.plot(ra_trace, pe_guess/pe_guess, color='k')#, label=r'Ra$^{1/2}$')
for k, data in mixed_data.items():
    df = pd.DataFrame(data=data)
    rolled = df.rolling(window=avg_window*10, min_periods=avg_window*10).mean()
    label='FT'

    if k == '{:.4e}'.format(2.61e9): 
        this_color=mColor2
        xy_points = [140, 153, 40, 53]
    else: 
        this_color=mColor
        xy_points = [140, 153, 40, 53]
    coords = []
    for i in xy_points:
        x = rolled['ra_temp'][avg_window*10 + i]
        y = (rolled['Pe']/pe_func(rolled['ra_temp']))[avg_window*10 + i]

        coords.append(( (np.log10(x)-np.log10(this_xlim[0]))/(np.log10(this_xlim[1])-np.log10(this_xlim[0])), 
                        (y-this_ylim[0])/(this_ylim[1]-this_ylim[0]) ))

    plt.arrow(coords[0][0], coords[0][1], coords[1][0]-coords[0][0], coords[1][1]-coords[0][1], transform=ax.transAxes,\
                 head_width=0.025, head_length=0.05, color=this_color, facecolor=mColor, rasterized='True')
    plt.arrow(coords[2][0], coords[2][1], coords[3][0]-coords[2][0], coords[3][1]-coords[2][1], transform=ax.transAxes,\
                 head_width=0.025, head_length=0.05, color=this_color, facecolor=mColor, rasterized='True')

#    plt.arrow(0.925, 0.2, -0.04, 0.16,transform=ax.transAxes,\
#                 head_width=0.025, head_length=0.05, color=mColor, facecolor=mColor, rasterized='True')
#    plt.arrow(0.937, 0.15, -0.013, 0.05,transform=ax.transAxes,\
#                 head_width=0.025, head_length=0.05, color=mColor, facecolor=mColor, rasterized='True')
    if k == '{:.4e}'.format(2.61e9):
        plt.plot(rolled['ra_temp'], rolled['Pe']/pe_func(rolled['ra_temp']), color=mColor2, zorder=0)
    else:
        plt.plot(rolled['ra_temp'], rolled['Pe']/pe_func(rolled['ra_temp']), color=mColor, zorder=0, label=label)
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
plt.errorbar(my_ra[1:], (my_pe/pe_func(my_ra))[1:], yerr=(my_pe_sampleMean/pe_func(my_ra))[1:], lw=0, elinewidth=1, capsize=1.5, c=fColor, ms=5, marker='o', label='TT')
plt.errorbar(my_ra[0],  (my_pe/pe_func(my_ra))[0], yerr=(my_pe_sampleMean/pe_func(my_ra))[0], lw=0, elinewidth=2, capsize=1.5, markeredgecolor=fColor, c=mColor2, markeredgewidth=3, ms=6, marker='o', zorder=1001)
plt.errorbar(my_ra[3],  (my_pe/pe_func(my_ra))[3], yerr=(my_pe_sampleMean/pe_func(my_ra))[3], lw=0, elinewidth=2, capsize=1.5, markeredgecolor=fColor, c=mColor,  markeredgewidth=3, ms=6, marker='o', zorder=1001)
#plt.scatter(my_ra, my_pe/pe_func(my_ra), s=15, c=fColor, marker='o', label='TT')
plt.xscale('log')
plt.xlabel(r'Ra$_{\Delta T}$')
plt.ylabel(r'Pe/(0.43 Ra$_{\Delta T}^{0.5}$)')
plt.xlim(9e7, 4e10)
plt.ylim(0.5, 1.5)

print('-----------------------------------------')
print('fixed')
print('-----------------------------------------')
for i, ra in enumerate(my_ra):
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(ra, my_nu[i], my_nu_sampleMean[i], my_pe[i], my_pe_sampleMean[i]))
print('-----------------------------------------')
print('mixed')
print('-----------------------------------------')
for ra, data in mixed_data.items():
    nu = np.mean(data['Nu'][-N:])
    nu_stdev = np.std(data['Nu'][-N:])
    pe = np.mean(data['Pe'][-N:])
    pe_stdev = np.std(data['Pe'][-N:])
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(float(ra), nu, nu_stdev/np.sqrt(N), pe, pe_stdev/np.sqrt(N)))
print('-----------------------------------------')
print('f-to-m')
print('-----------------------------------------')
for ra, data in restarted_data.items():
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
