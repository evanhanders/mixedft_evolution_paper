"""
Dedalus script for calculating the maximum growth rates in no-slip
Rayleigh Benard convection over a range of horizontal wavenumbers.

This script can be ran serially or in parallel, and produces a plot of the
highest growth rate found for each horizontal wavenumber.

To run using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)


# Global parameters
Nz = 64
Nk = 200
zoom = 0.1

data = {}

for case in [0,1,2,3]:

    if case == 0:
        velocity_bc    = "stress-free"
        temperature_bc = "TT"
        kc = 2.22

    if case == 1:
        velocity_bc    = "no-slip"
        temperature_bc = "TT"
        kc = 3.12

    if case == 2:
        velocity_bc    = "stress-free"
        temperature_bc = "FT"
        kc = 1.76

    if case == 3:
        velocity_bc    = "no-slip"
        temperature_bc = "FT"
        kc = 2.55

    kx_global = np.linspace(kc*(1-zoom),kc*(1+zoom), Nk)

    # Create bases and domain
    # Use COMM_SELF so keep calculations independent between processes
    z_basis = de.Chebyshev('z', Nz, interval=(-1/2, 1/2))
    domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

    # 2D Boussinesq hydrodynamics, with various boundary conditions
    # Use substitutions for x derivatives
    problem = de.EVP(domain, variables=['p','T','u','w','F','ω'], eigenvalue='Ra')
    problem.parameters['kx'] = 1
    problem.substitutions['dx(A)'] = "1j*kx*A"

    problem.add_equation("  dx(u) + dz(w) = 0")
    problem.add_equation("  dx(dx(T)) - dz(F) + w  = 0")
    problem.add_equation("   dz(ω) + dx(p)         = 0")
    problem.add_equation(" - dx(ω) + dz(p) - Ra*T  = 0")
    problem.add_equation("F + dz(T) = 0")
    problem.add_equation("ω - dx(w) + dz(u) = 0")

    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0")

    if velocity_bc == "no-slip":
        problem.add_bc("right(u) = 0")
        problem.add_bc("left(u) = 0")

    if velocity_bc == "stress-free":
        problem.add_bc("right(ω) = 0")
        problem.add_bc("left(ω) = 0")

    if temperature_bc == "TT":
        problem.add_bc("right(T) = 0")
        problem.add_bc("left(T) = 0")

    if temperature_bc == "FT":
        problem.add_bc("right(F) = 0")
        problem.add_bc("left(T) = 0")

    solver = problem.build_solver()

    # Create function to compute max growth rate for given kx
    def min_rayleigh(kx):
        logger.info('Computing min Rayleigh number for kx = %f' %kx)
        # Change kx parameter
        problem.namespace['kx'].value = kx
        # Solve for eigenvalues with sparse search near zero, rebuilding NCCs
        solver.solve_sparse(solver.pencils[0], N=10, target=0, rebuild_coeffs=True)
        # Return largest imaginary part
        return np.min(solver.eigenvalues.real)

    # Compute growth rate over local wavenumbers
    kx_local = kx_global[CW.rank::CW.size]
    t1 = time.time()
    ra_local = np.array([min_rayleigh(kx) for kx in kx_local])
    t2 = time.time()
    logger.info('Elapsed solve time: %f' %(t2-t1))

    # Reduce growth rates to root process
    ra_global = np.zeros_like(kx_global)
    ra_global[CW.rank::CW.size] = ra_local
    if CW.rank == 0:
        CW.Reduce(MPI.IN_PLACE, ra_global, op=MPI.SUM, root=0)
    else:
        CW.Reduce(ra_global, ra_global, op=MPI.SUM, root=0)

    # Plot growth rates from root process
    if CW.rank == 0:
        
        
        ra_min, ra_max = np.min(ra_global),np.max(ra_global)
        
        i = np.where(ra_global == ra_min)[0][0]
        
        (k0,km,k1) = kx_global[i-1:i+2]
        (r0,rm,r1) = ra_global[i-1:i+2]
        
        kmin  = (r1-rm)*k0**2 + (r0-r1)*km**2 + (rm-r0)*k1**2
        kmin /= 2*(k0*(r1-rm) + km*(r0-r1) + k1*(rm-r0))
        
        rmin  = r0*(kmin-k1)*(kmin-km)/(k0-k1)/(k0-km)
        rmin += rm*(kmin-k0)*(kmin-k1)/(k0-km)/(k1-km)
        rmin += r1*(kmin-k0)*(kmin-km)/(k0-k1)/(km-k1)
        
        data[(velocity_bc,temperature_bc)] = (kmin,rmin)
        
        plt.plot(kx_global/np.pi, ra_global, '.')
        plt.xlabel(r'aspect ratio ($2 L_{z}/L_{x}$)')
        plt.ylabel(r'Rayleigh number')
        
        plt.xlim([kx_global[0]/np.pi,kx_global[-1]/np.pi])
        plt.ylim([ra_min//1,ra_max//1])
        
        title = 'Boundary conditions: ' + velocity_bc + ', ' + temperature_bc
        plt.title(title)
        
        file = 'critical_Ra_' + velocity_bc + '_' + temperature_bc + '.png'
        plt.savefig(file,dpi=300)

if CW.rank == 0:

    for bc in data:
        print('')
        print(bc)
        print(25*'-')
        print('aspect ratio:    ',data[bc][0]/np.pi)
        print('Rayleigh number: ',data[bc][1])

    print('')
