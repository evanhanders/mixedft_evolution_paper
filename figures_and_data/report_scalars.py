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
nu_dirs = glob.glob("{:s}Nu_based_FT_2D/ra*/".format(base_dir))
numbered_dirs  = [(f, float(f.split("{:s}classic_FT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}TT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}TT-to-FT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in restarted_dirs]
restarted_dirs, restarted_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}Nu_based_FT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in nu_dirs]
nu_dirs, nu_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_data = read_data(mixed_ras, mixed_dirs)
fixed_data = read_data(fixed_ras, fixed_dirs)
restarted_data = read_data(restarted_ras, restarted_dirs)
nu_data = read_data(nu_ras, nu_dirs)

my_ra = []
my_nu = []
my_nu_sampleMean = []
my_pe = []
my_pe_sampleMean = []
N = 5000
for ra, data in fixed_data.items():
    nu = np.mean(data['Nu'][-N:])
    stdev = np.std(data['Nu'][-N:])
    my_ra.append(float(ra))
    my_nu.append(nu)
    my_nu_sampleMean.append(stdev/np.sqrt(N))

    pe = np.mean(data['Pe'][-N:])
    stdev = np.std(data['Pe'][-N:])
    my_pe.append(pe)
    my_pe_sampleMean.append(stdev/np.sqrt(N))
print('-----------------------------------------')
print('TT')
print('-----------------------------------------')
for i, ra in enumerate(my_ra):
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(ra, my_nu[i], my_nu_sampleMean[i], my_pe[i], my_pe_sampleMean[i]))
print('-----------------------------------------')
print('classic FT')
print('-----------------------------------------')
for ra, data in mixed_data.items():
    nu = np.mean(data['Nu'][-N:])
    nu_stdev = np.std(data['Nu'][-N:])
    pe = np.mean(data['Pe'][-N:])
    pe_stdev = np.std(data['Pe'][-N:])
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(float(ra), nu, nu_stdev/np.sqrt(N), pe, pe_stdev/np.sqrt(N)))
print('-----------------------------------------')
print('Nu-based FT')
print('-----------------------------------------')
for ra, data in nu_data.items():
    nu = np.mean(data['Nu'][-N:])
    nu_stdev = np.std(data['Nu'][-N:])
    pe = np.mean(data['Pe'][-N:])
    pe_stdev = np.std(data['Pe'][-N:])
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(float(ra), nu, nu_stdev/np.sqrt(N), pe, pe_stdev/np.sqrt(N)))
print('-----------------------------------------')
print('TT-to-FT')
print('-----------------------------------------')
for ra, data in restarted_data.items():
    if float(ra) == 9.51e11:
        N = 2000
    nu = np.mean(data['Nu'][-N:])
    nu_stdev = np.std(data['Nu'][-N:])
    pe = np.mean(data['Pe'][-N:])
    pe_stdev = np.std(data['Pe'][-N:])
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(float(ra), nu, nu_stdev/np.sqrt(N), pe, pe_stdev/np.sqrt(N)))



# Get data from files
base_dir = './data/rbc/3D/'
tt_to_ft_dirs = glob.glob("{:s}TT-to-FT/ra*/".format(base_dir))
fixed_dirs = glob.glob("{:s}TT/ra*/".format(base_dir))
nu_dirs = glob.glob("{:s}Nu_based_FT/ra*/".format(base_dir))
numbered_dirs  = [(f, float(f.split("{:s}TT-to-FT/ra".format(base_dir))[-1].split("/")[0])) for f in tt_to_ft_dirs]
tt_to_ft_dirs, tt_to_ft_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}TT/ra".format(base_dir))[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}Nu_based_FT/ra".format(base_dir))[-1].split("/")[0])) for f in nu_dirs]
nu_dirs, nu_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

tt_to_ft_data = read_data(tt_to_ft_ras, tt_to_ft_dirs)
fixed_data = read_data(fixed_ras, fixed_dirs)
nu_data = read_data(nu_ras, nu_dirs)

N = 3500
print('-----------------------------------------')
print('3D')
print('-----------------------------------------')
print('TT-to-FT')
print('-----------------------------------------')
for ra, data in tt_to_ft_data.items():
    nu = np.mean(data['Nu'][-N:])
    nu_stdev = np.std(data['Nu'][-N:])
    pe = np.mean(data['Pe'][-N:])
    pe_stdev = np.std(data['Pe'][-N:])
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(float(ra), nu, nu_stdev/np.sqrt(N), pe, pe_stdev/np.sqrt(N)))
print('-----------------------------------------')
print('TT')
print('-----------------------------------------')
for ra, data in fixed_data.items():
    nu = np.mean(data['Nu'][-N:])
    nu_stdev = np.std(data['Nu'][-N:])
    pe = np.mean(data['Pe'][-N:])
    pe_stdev = np.std(data['Pe'][-N:])
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(float(ra), nu, nu_stdev/np.sqrt(N), pe, pe_stdev/np.sqrt(N)))
print('-----------------------------------------')
print('Nu FT')
print('-----------------------------------------')
for ra, data in nu_data.items():
    nu = np.mean(data['Nu'][-N:])
    nu_stdev = np.std(data['Nu'][-N:])
    pe = np.mean(data['Pe'][-N:])
    pe_stdev = np.std(data['Pe'][-N:])
    print('{:.2e}\t {:.2e} +/- {:.2e}\t {:.2e} +/- {:.2e}'.format(float(ra), nu, nu_stdev/np.sqrt(N), pe, pe_stdev/np.sqrt(N)))



