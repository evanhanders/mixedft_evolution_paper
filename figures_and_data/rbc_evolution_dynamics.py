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
from scipy.interpolate import interp1d

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

mixed_dirs = glob.glob("./data/rbc/mixedFT_2d/ra*/")
fixed_dirs = glob.glob("./data/rbc/fixedT_2d/ra*/")
restarted_dirs = glob.glob("./data/rbc/restarted_mixed_T2m/ra*/")
numbered_dirs  = [(f, float(f.split("./data/rbc/mixedFT_2d/ra")[-1].split("/")[0])) for f in mixed_dirs]
mixed_dirs, mixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("./data/rbc/fixedT_2d/ra")[-1].split("/")[0])) for f in fixed_dirs]
fixed_dirs, fixed_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))
numbered_dirs  = [(f, float(f.split("./data/rbc/restarted_mixed_T2m/ra")[-1].split("/")[0])) for f in restarted_dirs]
restarted_dirs, restarted_ras = zip(*sorted(numbered_dirs, key=lambda x: x[1]))

mixed_data = read_pdfs(mixed_ras, mixed_dirs)
fixed_data = read_pdfs(fixed_ras, fixed_dirs)
restarted_data = read_pdfs(restarted_ras, restarted_dirs)
mixed_scalars = read_data(mixed_ras, mixed_dirs)
fixed_scalars = read_data(fixed_ras, fixed_dirs)
restarted_scalars = read_data(restarted_ras, restarted_dirs)

early_pdf = './data/rbc/mixedFT_2d/ra4.83e10/early_pdfs/pdf_data.h5'
with h5py.File('{:s}'.format(early_pdf), 'r') as f:
    data = OrderedDict()
    for k in f.keys():
        data[k] = OrderedDict()
        for sk in f[k].keys():
            data[k][sk] = f[k][sk][()]
    mixed_data['{:.4e}_early'.format(4.83e10)] = data

mke = '{:.4e}_early'.format(4.83e10)
mk = '{:.4e}'.format(4.83e10)
fke = '{:.4e}'.format(1.00e10)
fkl = '{:.4e}'.format(1.00e9)

# Set up figure subplots
fig = plt.figure(figsize=(7.5, 8))
gs = gridspec.GridSpec(1000, 1000)

subplots = [( (50  ,   0),          225,    500),
            ( (275 ,   0),          225,    500),
            ( (600 ,   0),          200,    500),
            ( (50  , 500),          225,    500),
            ( (275 , 500),          225,    500),
            ( (600 , 500),          200,    500),
            ( (800 , 0),            200,    500),
            ( (800 , 500),          200,    500),
            ]
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]

cax = plt.subplot(gs.new_subplotspec((0, 650), 50, 200))

#Dynamics files
ra1e9_TT  = './data/rbc/fixedT_2d/ra1.00e9/temp_slice.h5'#slices_s50.h5'
ra1e10_TT = './data/rbc/fixedT_2d/ra1.00e10/temp_slice.h5'#run07/slices_s1.h5'
ra1e9_FT  = './data/rbc/mixedFT_2d/ra4.83e10/temp_slice_late.h5'#run03/slices_s416.h5'
ra1e10_FT = './data/rbc/mixedFT_2d/ra4.83e10/temp_slice_early.h5'#run03/slices_s22.h5' 

loop =  (   (axs[0], ra1e10_TT,   np.mean(fixed_scalars[fke]['delta_T'][-5000:]) ),
            (axs[1], ra1e10_FT,    np.mean(mixed_scalars[mk]['delta_T'][7500:12500])   ),
            (axs[3], ra1e9_TT,    np.mean(fixed_scalars[fkl]['delta_T'][-5000:]) ),
            (axs[4], ra1e9_FT,     np.mean(mixed_scalars[mk]['delta_T'][-5000:])  )
        )

for ax, filen, dT in loop:
    print('writing from file {}'.format(filen))
    print('delta T is {:.4e}'.format(dT))
    plt.sca(ax)

    with h5py.File(filen, 'r') as f:
        x = f['x'][()]
        z = f['z'][()]
        T = f['T'][()]
    print(T.max(), T.min())


    x_basis = de.Fourier(    'x', len(x), interval=[-1, 1], dealias=1)
    z_basis = de.Chebyshev(  'z', len(z), interval=[-1/2, 1/2], dealias=1)
    bases = [x_basis, z_basis]
    domain = de.Domain(bases, grid_dtype=np.float64)
    base_scale=1
    highres_n = 4096
    big_scale=int(highres_n/len(x))
    T_field = domain.new_field()
    T_field['g'] = T - T.mean(axis=0)#T_prof

    x_big = domain.grid(0, scales=big_scale)
    z_big = domain.grid(1, scales=big_scale)
    zz_b, xx_b = np.meshgrid(z_big.flatten(), x_big.flatten())

    T_field.set_scales(big_scale, keep_data=True)
    c = ax.pcolormesh(xx_b, zz_b, T_field['g'], cmap='RdBu_r', rasterized=True, vmin=-dT/3, vmax=dT/3)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 0.5)

 
for i in [1, 3, 4]:
    axs[i].set_xticks(())
    axs[i].set_yticks(())
axs[0].set_yticks((-0.5, 0, 0.5))
axs[0].set_ylabel('z', labelpad=-1)
axs[0].set_xlabel('x')
axs[0].xaxis.set_ticks_position('top')
axs[0].xaxis.set_label_position('top')


bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.75)
axs[0].text(0.18, 0.11, r"TT, Ra$_{\Delta T}\,=\,10^{10}$", ha="center", va="center", size=8, bbox=bbox_props, transform=axs[0].transAxes)
axs[1].text(0.25, 0.89, r"FT (early), Ra$_{\Delta T}\,\approx\,10^{10}$", ha="center", va="center", size=8, bbox=bbox_props, transform=axs[1].transAxes)
axs[3].text(0.83, 0.11, r"TT, Ra$_{\Delta T}\,=\,10^{9}$", ha="center", va="center", size=8, bbox=bbox_props, transform=axs[3].transAxes)
axs[4].text(0.77, 0.89, r"FT (late), Ra$_{\Delta T}\,\approx\,10^{9}$", ha="center", va="center", size=8, bbox=bbox_props, transform=axs[4].transAxes)

#Colorbar
bar = plt.colorbar(c, cax=cax, orientation='horizontal')#, rasterized=True)
cax.set_xticklabels(())
bar.set_ticks(())
cax.text(1.05, 0.35, r'$\pm\Delta T / 3$', transform=cax.transAxes, ha='left')
cax.text(-0.05,0.35,  r'$T - \bar{T}$', transform=cax.transAxes, ha='right')
#cax.annotate(r'$-|S| \times 10^{-5}$', fontsize=8,  xy=(-0.37, 0.5), va='center', annotation_clip=False)
#cax.annotate(r'$|S| \times 10^{-5}$', fontsize=8,  xy=(1.02, 0.5),  va='center',  annotation_clip=False)


# Early PDFs
axtt = axs[2]
axft = axs[6]
mx, mp, mdx = [mixed_data[mke]['T'][k] for k in ['xs', 'pdf', 'dx']]
fx, fp, fdx = [fixed_data[fke]['T'][k] for k in ['xs', 'pdf', 'dx']]
fdx = float(fdx)
mdx = float(mdx)

mmean  = np.sum(mx*mp*mdx)
msigma = np.sqrt(np.sum((mx-mmean)**2*mp*mdx))
fmean  = np.sum(fx*fp*fdx)
fsigma = np.sqrt(np.sum((fx-fmean)**2*fp*fdx))

mcdf = np.zeros_like(mp)
for i in range(len(mp)-1):
    mcdf[i+1] = mcdf[i] + mp[i]*mdx
fcdf = np.zeros_like(fp)
for i in range(len(fp)-1):
    fcdf[i+1] = fcdf[i] + fp[i]*fdx

ffunc = interp1d(fcdf, fx-fdx/2)
mfunc = interp1d(mcdf, mx-mdx/2)
axft.fill_between([mfunc(0.16), mfunc(0.84)], 1e-16, 1e5, color='grey', alpha=0.3)
axft.axvline(mfunc(0.5), c='black', lw=0.5)
axtt.fill_between([ffunc(0.16), ffunc(0.84)], 1e-16, 1e5, color='grey', alpha=0.3)
axtt.axvline(ffunc(0.5), c='black', lw=0.5)

axft.plot(mx, mp, label='FT', c=mColor)
#plt.plot(mx/dT, mp*dT, label='FT', c=mColor)
axtt.plot(fx, fp, label='TT', c=fColor)
for ax in [axft, axtt]:
    ax.set_yscale('log')
    ax.set_ylabel(r'$P(T)$', labelpad=-0.25)
    ax.set_xlabel(r'$T$')
#    ax.legend(loc='upper right', frameon=False, fontsize=7, markerfirst=False, borderpad=0.1)

mcdf = np.zeros_like(mp)
for i in range(len(mp)-1):
    mcdf[i+1] = mcdf[i] + mp[i]*mdx

axft.fill_between(mx, 1e-16, mp, color=mColor, alpha=0.5)
#ax.fill_between(mx/dT, 1e-16, mp*dT, color=mColor, alpha=0.5)
axtt.fill_between(fx, 1e-16, fp, color=fColor, alpha=0.5)

axtt.set_ylim(np.min(fp[fp > 0]), 1.5*np.max(fp[fp > 0]))
axft.set_ylim(1e-3, 1.5*np.max(mp[mp > 0]))
axtt.set_xlim(0, 1)
axft.set_xlim(0, 0.3)#2*np.sum(mx*mp*mdx))#mx[mcdf > 0.999][0])#2*loop[-1][-1])

axtt.set_xticks((0, 0.25, 0.5, 0.75))
axft.set_xticks((0, 0.05, 0.1, 0.15, 0.2, 0.25))
axtt.xaxis.set_label_position('top')


# Late PDFs
axtt = axs[5]
axft = axs[7]
mx, mp, mdx = [mixed_data[mk]['T'][k] for k in ['xs', 'pdf', 'dx']]
fx, fp, fdx = [fixed_data[fkl]['T'][k] for k in ['xs', 'pdf', 'dx']]
fdx = float(fdx)
mdx = float(mdx)

mmean  = np.sum(mx*mp*mdx)
msigma = np.sqrt(np.sum((mx-mmean)**2*mp*mdx))
fmean  = np.sum(fx*fp*fdx)
fsigma = np.sqrt(np.sum((fx-fmean)**2*fp*fdx))

mcdf = np.zeros_like(mp)
for i in range(len(mp)-1):
    mcdf[i+1] = mcdf[i] + mp[i]*mdx
fcdf = np.zeros_like(fp)
for i in range(len(fp)-1):
    fcdf[i+1] = fcdf[i] + fp[i]*fdx

ffunc = interp1d(fcdf, fx-fdx/2)
mfunc = interp1d(mcdf, mx-mdx/2)
axft.fill_between([mfunc(0.16), mfunc(0.84)], 1e-16, 1e5, color='grey', alpha=0.3)
axft.axvline(mfunc(0.5), c='black', lw=0.5)
axtt.fill_between([ffunc(0.16), ffunc(0.84)], 1e-16, 1e5, color='grey', alpha=0.3)
axtt.axvline(ffunc(0.5), c='black', lw=0.5)

axft.plot(mx, mp, label='FT', c=mColor)
#plt.plot(mx/dT, mp*dT, label='FT', c=mColor)
axtt.plot(fx, fp, label='TT', c=fColor)
for ax in [axft, axtt]:
    ax.set_yscale('log')
    ax.set_ylabel(r'$P(T)$', labelpad=-0.25)
    ax.set_xlabel(r'$T$')
#    ax.legend(loc='upper right', frameon=False, fontsize=7, markerfirst=False, borderpad=0.1)


axft.fill_between(mx, 1e-16, mp, color=mColor, alpha=0.5)
#ax.fill_between(mx/dT, 1e-16, mp*dT, color=mColor, alpha=0.5)
axtt.fill_between(fx, 1e-16, fp, color=fColor, alpha=0.5)

axtt.set_ylim(np.min(fp[fp > 0]), 1.5*np.max(fp[fp > 0]))
axft.set_ylim(mp[mcdf > 0.9999][0], 1.5*np.max(mp[mp > 0]))
axtt.set_xlim(0, 1.35)
axft.set_xlim(0, mx[mcdf > 0.9999][0])#3*np.sum(mx*mp*mdx))#mx[mcdf > 0.999][0])#2*loop[-1][-1])


axtt.set_xticks((0, 0.5, 1))
axft.set_xticks((0, 0.005, 0.01, 0.015, 0.02, 0.025))
axtt.xaxis.set_label_position('top')
axft.yaxis.set_label_position('right')
axtt.yaxis.set_label_position('right')


axs[2].text(0.18, 0.87, r"TT, Ra$_{\Delta T}\,=\,10^{10}$", ha="center", va="center", size=8, bbox=bbox_props, transform=axs[2].transAxes)
axs[6].text(0.25, 0.87, r"FT (early), Ra$_{\Delta T}\,\approx\,10^{10}$", ha="center", va="center", size=8, bbox=bbox_props, transform=axs[6].transAxes)
axs[5].text(0.83, 0.87, r"TT, Ra$_{\Delta T}\,=\,10^{9}$", ha="center", va="center", size=8, bbox=bbox_props, transform=axs[5].transAxes)
axs[7].text(0.77, 0.87, r"FT (late), Ra$_{\Delta T}\,\approx\,10^{9}$", ha="center", va="center", size=8, bbox=bbox_props, transform=axs[7].transAxes)


for i in (0, 1, 3, 4):
    [s.set_linewidth(2) for k,s in axs[i].spines.items()]
#for i, side in zip((0, 1, 3, 4), (('right', 'bottom'), ('top', 'right'), ('left', 'bottom'), ('left', 'top'))):
#    for k in side:
#        axs[i].spines[k].set_linewidth(2)


for i in (5, 7):
    axs[i].yaxis.set_ticks_position('right')
for i in (2, 5):
    axs[i].xaxis.set_ticks_position('top')


fig.savefig('rbc_evolution_dynamics.png', dpi=150, bbox_inches='tight')
fig.savefig('rbc_evolution_dynamics.pdf', dpi=300, bbox_inches='tight')
