import os
import sys

import numpy as np
import pandas as pd
import math

from tqdm import tqdm

sys.path.insert(0, '../../icenumerics/')
sys.path.insert(0, './auxnumerics/')
import icenumerics as ice

from parameters import params
import auxiliary as aux
import montecarlo as mc
import energy as eg
from numba import jit
from numba_progress import ProgressBar

ureg = ice.ureg
idx = pd.IndexSlice

def is_accepted(dE,kbT):
    arg = dE/kbT
    if dE<0:
        return True
    else:
        r = np.random.rand()
        if r<np.exp(arg):
            return True
        else:
            return False

def save_state(centers,dirs,rels,energy,step,field,create=True):
    df = mc.numpy2trj(centers,dirs,rels,vanilla=True)
    df['frame'] = [step]*len(centers)
    df['field'] = [field]*len(centers)

    if create:
        df.to_csv(TRJ_SAVE_FILE,index=False)
    else:
        df.to_csv(TRJ_SAVE_FILE,mode='a',header=False,index=False)


def metropolis(centers, dirs, rels, prefactor, mcsteps=1000):

    atoms = centers + rels
    a = np.round(params['lattice_constant'].magnitude, 5)
    Eold = eg.calculate_energy(prefactor, N*a, atoms)

    print(Eold)
    for i in tqdm(range(mcsteps)):
        # generate new state and compute energy
        dirsnew, relsnew, pos = mc.flip_loop(a,N,centers,dirs,rels)
        dirsnew, relsnew = mc.flip_spin(dirsnew,relsnew, np.random.randint(0,high=len(centers)))
        newatoms = centers + relsnew
        Enew = eg.calculate_energy(prefactor, N*a, newatoms)

        # energy difference
        dE = Enew - Eold

        if is_accepted(dE,kbT):
            dirs = dirsnew.copy()
            rels = relsnew.copy()
            Eold = Enew
        else:
            Eold=Eold

        if i==0:
            save_state(centers,dirs,rels,Eold,i,B,create=True)
        else:
            save_state(centers,dirs,rels,Eold,i,B,create=False)


## GLOBAL CONSTANTS
mu0 = (4*np.pi)*1e-7 * ureg.H/ureg.m
kb = 1.380649e-23 * ureg.J / ureg.K
kbT = (params['kb'] * params['sim_temp']).to(ureg.nm * ureg.pN).magnitude

## GETTING SOME RANDOM COLLOIDAL ICE OBJECT
N = 10
a = params['lattice_constant']

sp = ice.spins()
sp.create_lattice('square', [N,N], lattice_constant=a, border='periodic')

particle = ice.particle(radius=params['particle_radius'],
                        susceptibility=params['particle_susceptibility'],
                        diffusion=params['particle_diffusion'],
                        temperature=params['particle_temperature'],
                        density=params['particle_density']
)

trap = ice.trap(trap_sep=params['trap_sep'],
                height=params['trap_height'],
                stiffness=params['trap_stiffness']
)


col = ice.colloidal_ice(sp, particle, trap,
                       height_spread=0,
                       susceptibility_spread=0.1,
                       periodic=True)

particle_radius = params['particle_radius']
col.region = np.array([[0,0,-3*(particle_radius/a/N).magnitude],[1,1,3*(particle_radius/a/N).magnitude]])*N*a

## COMPUTING ENERGY SCALE PREFACTOR
centers, dirs, rels = mc.trj2numpy(col.to_ctrj())

B = 1*ureg.mT
m = np.pi * (2*params['particle_radius'])**3 *params['particle_susceptibility']*B/6/mu0
prefactor = -(mu0*m**2/4/np.pi).to(ureg.pN * ureg.nm * ureg.um**3).magnitude

## DOING METROPOLIS
TRJ_SAVE_FILE = '/media/frieren/BIG/stuckgs/data/metropolis/trj.csv'
metropolis(centers, dirs, rels, prefactor, mcsteps=1000)
