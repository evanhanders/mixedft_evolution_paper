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


mColor = 'darkorange'#olivedrab'
fColor = 'indigo'

def read_data(ra_list, dir_list, keys=['Nu', 'delta_T', 'sim_time', 'Pe', 'KE', 'left_flux', 'right_flux', 'Ro']):
    """
    Reads scalar data in from a folder containing a series of subfolders of different Ra values.
    """
    full_data = OrderedDict()
    for ra, dr in zip(ra_list, dir_list):
        data = OrderedDict()
        sub_runs = glob.glob('{:s}/run*/'.format(dr))
        if len(sub_runs) > 0:
            numbered_dirs  = [(r, int(r.split('run')[-1].split('/')[0].split('_')[0])) for r in sub_runs]
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
mixed_dirs = glob.glob("{:s}classic_FT/ra*/".format(base_dir))
fixed_dirs = glob.glob("{:s}TT/ra*/".format(base_dir))
tt_to_ft_dirs = glob.glob("{:s}TT-to-FT/ra*/".format(base_dir))
numbered_dirs  = [(f, float(f.split("{:s}classic_FT/ra".format(base_dir))[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}TT/ra".format(base_dir))[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}TT-to-FT/ra".format(base_dir))[-1].split("/")[0])) for f in tt_to_ft_dirs]
tt_to_ft_dirs, tt_to_ft_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_data = read_data(mixed_ras, mixed_dirs)
fixed_data = read_data(fixed_ras, fixed_dirs)
tt_to_ft_data = read_data(tt_to_ft_ras, tt_to_ft_dirs)

#Reporting
N = 5000
print('-----------------------------------------')
print('TT classic')
print('-----------------------------------------')
for ra, data in fixed_data.items():
    nu = np.mean(data['Nu'][-N:])
    nu_stdev = np.std(data['Nu'][-N:])
    pe = np.mean(data['Pe'][-N:])
    pe_stdev = np.std(data['Pe'][-N:])
    ro = np.mean(data['Ro'][-N:])
    ro_stdev = np.std(data['Ro'][-N:])
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(float(ra), nu, nu_stdev/np.sqrt(N), pe, pe_stdev/np.sqrt(N), ro, ro_stdev))

print('-----------------------------------------')
print('FT classic')
print('-----------------------------------------')
for ra, data in mixed_data.items():
    nu = np.mean(data['Nu'][-N:])
    nu_stdev = np.std(data['Nu'][-N:])
    pe = np.mean(data['Pe'][-N:])
    pe_stdev = np.std(data['Pe'][-N:])
    ro = np.mean(data['Ro'][-N:])
    ro_stdev = np.std(data['Ro'][-N:])
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(float(ra), nu, nu_stdev/np.sqrt(N), pe, pe_stdev/np.sqrt(N), ro, ro_stdev))
    print('-----------------------------------------')

print('TT-to-FT')
print('-----------------------------------------')
for ra, data in tt_to_ft_data.items():
    nu = np.mean(data['Nu'][-N:])
    nu_stdev = np.std(data['Nu'][-N:])
    pe = np.mean(data['Pe'][-N:])
    pe_stdev = np.std(data['Pe'][-N:])
    ro = np.mean(data['Ro'][-N:])
    ro_stdev = np.std(data['Ro'][-N:])
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(float(ra), nu, nu_stdev/np.sqrt(N), pe, pe_stdev/np.sqrt(N), ro, ro_stdev))
 
