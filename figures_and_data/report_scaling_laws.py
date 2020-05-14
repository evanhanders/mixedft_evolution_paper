import glob
from collections import OrderedDict

import h5py
import numpy as np
import matplotlib.pyplot as plt


from scipy.optimize import curve_fit
def line(x, a, b):
    return a + b*x

def read_data(ra_list, dir_list, keys=['Nu', 'delta_T', 'sim_time', 'Pe', 'KE', 'left_flux', 'right_flux', 'left_T', 'right_T']):
    """
    Reads scalar data in from a folder containing a series of subfolders of different Ra values.
    """
    full_data = OrderedDict()
    for ra, dr in zip(ra_list, dir_list):
        data = OrderedDict()
        sub_runs = glob.glob('{:s}/run*/'.format(dr))
        if len(sub_runs) > 0:
            try: 
                numbered_dirs  = [(f, float(f.split("run")[-1].split("/")[0])) for f in sub_runs]
            except:
                numbered_dirs  = [(f, float(f.split("run")[-1].split("/")[0].split('_')[0])) for f in sub_runs]
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
numbered_dirs  = [(f, float(f.split("{:s}classic_FT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("{:s}TT_2D/ra".format(base_dir))[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_data = read_data(mixed_ras, mixed_dirs)
fixed_data = read_data(fixed_ras, fixed_dirs)

N = 10
for k, data in mixed_data.items():
    fig = plt.figure()
    ra_dz = float(k)
    ra_dT = ra_dz*data['delta_T'][400:]
    finite = np.isfinite(ra_dT)
    ra_dT = ra_dT[finite]

    #Set up Ra bins
    ra_ranges = np.zeros((N,2))
    if '4.83' in k:
        max_dT = 2.5e10
    else:
        max_dT = ra_dT.max()
    ra_points = np.logspace(np.log10(ra_dT.min()), np.log10(max_dT), N)
    ra_ranges[0][0] = ra_dT.min()
    ra_ranges[-1][-1] = max_dT
    dra = np.diff(ra_points)
    for i in range(N-1):
        ra_ranges[i][1] = ra_points[i] + dra[i]/2
        ra_ranges[i+1][0] = ra_points[i+1] - dra[i]/2
    
    for i, fd in enumerate(['Nu', 'Pe']):
        ax = fig.add_subplot(1, 2, 1+i)
        plt.grid(which='both')
        values  = data[fd][400:]
        values  = values[finite]

        ras, vals, err = np.zeros(N), np.zeros(N), np.zeros(N)
        for j in range(N):
            good = (ra_dT > ra_ranges[j][0] ) * (ra_dT <= ra_ranges[j][1])
            mean_ra = np.mean(ra_dT[good])
            mean_val = np.mean(values[good])
            std_val  = np.std(values[good])
            plt.plot(ra_dT[good], values[good])
            ras[j]  = mean_ra
            vals[j] = mean_val
            err[j]  = std_val/np.sqrt(np.sum(good))
        plt.errorbar(ras, vals, yerr=err, zorder=1e2, ms=4, marker='o', c='k', capsize=1, elinewidth=1, lw=0) 
        fit = curve_fit(line, np.log10(ras), np.log10(vals), (-1, 1/3), err)
        print('(FT, Ra = {}) best fit: {} = {:.2e} * ra ^ {:.3g}'.format(k, fd, 10**fit[0][0], fit[0][1]))
        plt.plot(ras, 10**(fit[0][0])*(ras)**(fit[0][1]), c='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Ra')
        if i == 0:
            plt.ylabel('Nu')
            plt.ylim(5e0, 2e2)
        elif i == 1:
            plt.ylabel('Pe')
            plt.ylim(1e3, 1e5)
        fig.savefig('FT_scalings_{}_vs_time.png'.format(k), dpi=200, bbox_inches='tight')

fig = plt.figure()
for i, fd in enumerate(['Nu', 'Pe', 'Pe']):
    ax = fig.add_subplot(1,3, 1+i)
    fixed_ra = []
    fixed_val = []
    fixed_err = []
    for k in fixed_data.keys():
        if i == 1 and float(k) < 1e9: continue
        if i == 2 and float(k) >= 1e9: continue
        fixed_ra.append(float(k))
        fixed_val.append(np.mean(fixed_data[k][fd][-5000:]))
        fixed_err.append(np.std(fixed_data[k][fd][-5000:])/np.sqrt(5000))
    fixed_ra = np.array(fixed_ra)
    fixed_val = np.array(fixed_val)
    fixed_err = np.array(fixed_err)

    plt.errorbar(fixed_ra, fixed_val, yerr=fixed_err, zorder=1e2, ms=4, marker='o', c='k', capsize=1, elinewidth=1, lw=0) 
    fit = curve_fit(line, np.log10(fixed_ra), np.log10(fixed_val), (-1, 1/3), fixed_err)
    print('(TT) best fit: {} = {:.2e} * ra ^ {:.3g}'.format(fd, 10**fit[0][0], fit[0][1]))
    plt.plot(fixed_ra, 10**(fit[0][0])*(fixed_ra)**(fit[0][1]), c='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Ra')
    plt.ylabel(fd)


fig.savefig('TT_scalings.png', dpi=200, bbox_inches='tight')


# Get data from files
base_dir = './data/rotation/'
fixed_dirs = glob.glob("{:s}TT/ra*/".format(base_dir))
numbered_dirs  = [(f, float(f.split("{:s}TT/ra".format(base_dir))[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

rot_mixed_data = read_data([2.1e10,], ['./data/rotation/classic_FT/ra2.1e10/',])
rot_fixed_data = read_data(fixed_ras, fixed_dirs)

N = 10
for k, data in rot_mixed_data.items():
    fig = plt.figure()
    ra_dz = float(k)
    ra_dT = ra_dz*data['delta_T'][400:]
    finite = np.isfinite(ra_dT)
    ra_dT = ra_dT[finite]

    #Set up Ra bins
    max_dT = 8e9#ra_dT.max()
    ra_ranges = np.zeros((N,2))
    ra_points = np.logspace(np.log10(ra_dT.min()), np.log10(max_dT), N)
    ra_ranges[0][0] = ra_dT.min()
    ra_ranges[-1][-1] = max_dT
    dra = np.diff(ra_points)
    for i in range(N-1):
        ra_ranges[i][1] = ra_points[i] + dra[i]/2
        ra_ranges[i+1][0] = ra_points[i+1] - dra[i]/2
    
    for i, fd in enumerate(['Nu', 'Pe']):
        ax = fig.add_subplot(1, 2, 1+i)
        plt.grid(which='both')
        values  = data[fd][400:]
        values  = values[finite]
        plt.plot(ra_dT, values)

        ras, vals, err = np.zeros(N), np.zeros(N), np.zeros(N)
        for j in range(N):
            good = (ra_dT > ra_ranges[j][0] ) * (ra_dT <= ra_ranges[j][1])
            mean_ra = np.mean(ra_dT[good])
            mean_val = np.mean(values[good])
            std_val  = np.std(values[good])
            plt.plot(ra_dT[good], values[good])
            ras[j]  = mean_ra
            vals[j] = mean_val
            err[j]  = std_val/np.sqrt(np.sum(good))
        plt.errorbar(ras, vals, yerr=err, zorder=1e2, ms=4, marker='o', c='k', capsize=1, elinewidth=1, lw=0) 
        fit = curve_fit(line, np.log10(ras), np.log10(vals), (-1, 1/3), err)
#        fit = curve_fit(line, np.log10(ras[:-3]), np.log10(vals[:-3]), (-1, 1/3), err[:-3])
        print('(FT, Ra = {}) best fit: {} = {:.2e} * ra ^ {:.3g}'.format(k, fd, 10**fit[0][0], fit[0][1]))
        plt.plot(ras, 10**(fit[0][0])*(ras)**(fit[0][1]), c='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Ra')
        if i == 0:
            plt.ylabel('Nu')
            plt.ylim(5e0, 2e2)
        elif i == 1:
            plt.ylabel('Pe')
            plt.ylim(1e3, 1e5)
    fig.savefig('rot_FT_scalings_{}_vs_time.png'.format(k), dpi=200, bbox_inches='tight')

fig = plt.figure()
for i, fd in enumerate(['Nu', 'Pe']):
    ax = fig.add_subplot(1,2, 1+i)
    fixed_ra = []
    fixed_val = []
    fixed_err = []
    for k in rot_fixed_data.keys():
        fixed_ra.append(float(k))
        fixed_val.append(np.mean(rot_fixed_data[k][fd][-5000:]))
        fixed_err.append(np.std(rot_fixed_data[k][fd][-5000:])/np.sqrt(5000))
    plt.errorbar(fixed_ra, fixed_val, yerr=fixed_err, zorder=1e2, ms=4, marker='o', c='k', capsize=1, elinewidth=1, lw=0) 

    fixed_ra = []
    fixed_val = []
    fixed_err = []
    for k in rot_fixed_data.keys():
        if float(k) > 8e9: continue
        fixed_ra.append(float(k))
        fixed_val.append(np.mean(rot_fixed_data[k][fd][-5000:]))
        fixed_err.append(np.std(rot_fixed_data[k][fd][-5000:])/np.sqrt(5000))

    fixed_ra = np.array(fixed_ra)
    fixed_val = np.array(fixed_val)
    fixed_err = np.array(fixed_err)

    fit = curve_fit(line, np.log10(fixed_ra), np.log10(fixed_val), (-1, 1/3), fixed_err)
    print('(TT) best fit: {} = {:.2e} * ra ^ {:.3g}'.format(fd, 10**fit[0][0], fit[0][1]))
    plt.plot(fixed_ra, 10**(fit[0][0])*(fixed_ra)**(fit[0][1]), c='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Ra')
    plt.ylabel(fd)


fig.savefig('rot_TT_scalings.png', dpi=200, bbox_inches='tight')
