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
nu_ft_dirs = glob.glob("{:s}Nu_based_FT_2D/ra*/".format(base_dir))
mixed_dirs = glob.glob("{:s}classic_FT_2D/ra*/".format(base_dir))
fixed_dirs = glob.glob("{:s}TT_2D/ra*/".format(base_dir))
restarted_dirs = glob.glob("{:s}TT-to-FT_2D/ra*/".format(base_dir))
numbered_dirs  = [(f, float(f.split("{:s}Nu_based_FT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in nu_ft_dirs]
nu_ft_dirs, nu_ft_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}classic_FT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}TT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}TT-to-FT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in restarted_dirs]
restarted_dirs, restarted_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

nu_ft_data = read_data(nu_ft_ras, nu_ft_dirs)
mixed_data = read_data(mixed_ras, mixed_dirs)
fixed_data = read_data(fixed_ras, fixed_dirs)
restarted_data = read_data(restarted_ras, restarted_dirs)
nu_ft_pdfs = read_pdfs(nu_ft_ras, nu_ft_dirs)
mixed_pdfs = read_pdfs(mixed_ras, mixed_dirs)
fixed_pdfs = read_pdfs(fixed_ras, fixed_dirs)
restarted_pdfs = read_pdfs(restarted_ras, restarted_dirs)

fk = '{:.4e}'.format(1.00e9)
mk = '{:.4e}'.format(4.83e10)

Nu_approx=48.3


keys = ['T', 'enstrophy', 'enth_flux', 'w']
labels = [r'$T/\Delta T$', r'$\omega^2 / 10^2$', r'$w T\cdot 10^3$', r'$w$']

print('pdf data: TT, FT, TT-to-FT, Nu-based FT')
Nus = []
deltaTs = []
for run_k, data in zip((fk, mk, mk, mk), (fixed_data, mixed_data, restarted_data, nu_ft_data)):
    Nus.append(np.mean(data[run_k]['Nu'][-5000:]))
    deltaTs.append(np.mean(data[run_k]['delta_T'][-5000:]))
    

for i, field in enumerate(keys):
    print('-'*50)
    print(field)
    j = 0
    for run_k, pdfs in zip((fk, mk, mk, mk), (fixed_pdfs, mixed_pdfs, restarted_pdfs, nu_ft_pdfs)):
        x, p, dx = [pdfs[run_k][field][k] for k in ['xs', 'pdf', 'dx']]

        Nu = Nus[j]
        delta_T = deltaTs[j]
        j += 1

#        print('Nu: {:.2e}  // deltaT: {:.2e}'.format(Nu, delta_T))

        if run_k != fk:
            if field == 'T' or field == 'enstrophy':
                x  /= delta_T
                dx /= delta_T
                p  *= delta_T
            elif field == 'enth_flux':
                x  /= delta_T**(3/2)
                dx /= delta_T**(3/2)
                p  *= delta_T**(3/2)
            elif field == 'w':
                x  /= delta_T**(1/2)
                dx /= delta_T**(1/2)
                p  *= delta_T**(1/2)

        mean = np.sum(x*p*dx)
        sigma= np.sqrt(np.sum((x-mean)**2*p*dx))
        skew = (1/sigma**3)*np.sum((x-mean)**3*p*dx)
        kurt = (1/sigma**4)*np.sum((x-mean)**4*p*dx)
        print('mean: {:.4e}, sigma: {:.4e}, skew: {:.4e}, kurt: {:.4e}'.format(mean, sigma, skew, kurt))



