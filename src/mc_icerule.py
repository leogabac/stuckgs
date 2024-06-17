import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
from IPython.display import clear_output

import matplotlib as mpl 
import matplotlib.pyplot as plt

sys.path.insert(0, '../../icenumerics/')
sys.path.insert(0, './auxnumerics/')
import icenumerics as ice


from parameters import params
import auxiliary as aux
import montecarlo as mc
import chirality_tools as chir
from numba import jit

ureg = ice.ureg
idx = pd.IndexSlice


def simulated_annealing(file_path, mcsteps, centers, dirs, rels, realization):

    # compute the old 'energy'
    Eold = mc.obj_icerule(indices_matrix,dirs,N)

    T = 3000

    for i in tqdm(range(mcsteps)):

        # generate new configuration
        dirs_new, rels_new = mc.flip_loop(params['lattice_constant'].magnitude, N, centers,dirs,rels)
        # some flips
        dirs_new,rels_new = mc.flip_spin(dirs_new,rels_new,np.random.randint(0,len(dirs)))

        # compute the new energy
        Enew = mc.obj_icerule(indices_matrix,dirs_new,N)

        # compute dE
        dE = Enew - Eold

        
        # Accept or reject the change
        if mc.is_accepted(dE,T):
            dirs = dirs_new.copy()
            rels = rels_new.copy()

            Eold = Enew
        else:
            
            Eold = Eold


        T = 0.95*T
    
    # save the stuff
    print(mc.obj_icerule(indices_matrix,dirs,N))
    print('saving...')
    df = mc.numpy2trj(centers,dirs,rels)
    df['realization'] = [realization]*len(df)

    if realization == 1:
        df.to_csv(file_path)
    else:
        df.to_csv(file_path,mode='a',header=False)



## initializing data types
N = 10
a = params["lattice_constant"]
print(f'Size: {N}')

# initialize neutral charge colloid
sp = ice.spins()
sp.create_lattice("square",[N,N],lattice_constant=a, border="periodic")


particle = ice.particle(radius = params["particle_radius"],
            susceptibility = params["particle_susceptibility"],
            diffusion = params["particle_diffusion"],
            temperature = params["particle_temperature"],
            density = params["particle_density"]
            )


trap = ice.trap(trap_sep = params["trap_sep"],
            height = params["trap_height"],
            stiffness = params["trap_stiffness"]
            )


col = ice.colloidal_ice(sp, particle, trap,
                        height_spread = params["height_spread"], 
                        susceptibility_spread = params["susceptibility_spread"],
                        periodic = params["isperiodic"])

    
col.region = np.array([[0,0,-3*(params["particle_radius"]/a/N).magnitude],[1,1,3*(params["particle_radius"]/a/N).magnitude]])*N*a


## stuff before the simulated annealing process
trj = col.to_ctrj()
centers, dirs, rels = mc.trj2numpy(trj)
vrt_lattice = mc.create_lattice(a.magnitude,N,spos=(0,0))
indices_matrix = mc.indices_lattice(vrt_lattice,centers, a.magnitude, N)


file_path = f'../data/q2_degeneracy/ice{N}.csv'

for i in range(1,2):
    print(f'===== realization {i} =====')
    simulated_annealing(file_path,int(1e7), centers, dirs, rels,i)