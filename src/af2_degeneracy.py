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
    Eold = mc.get_objective_function(indices_matrix,dirs,N)

    T = 3000

    for i in tqdm(range(mcsteps)):

        # generate new configuration
        dirs_new, rels_new = mc.flip_loop(params['lattice_constant'].magnitude, N, centers,dirs,rels)
        #dirs_new,rels_new = mc.flip_spin(dirs_new,rels_new,np.random.randint(0,len(dirs)))

        # compute the new energy
        Enew = mc.get_objective_function(indices_matrix,dirs_new,N)

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
    print('saving...')
    df = mc.numpy2trj(centers,dirs,rels)
    df['realization'] = [realization]*len(df)

    if realization == 1:
        df.to_csv(file_path)
    else:
        df.to_csv(file_path,mode='a',header=False)



## initializing data types
N = 50
print(f'Size: {N}')
a = params["lattice_constant"]
afstate_path = '../data/states/af2'

## stuff before the simulated annealing process
trj = pd.read_csv(os.path.join(afstate_path,f'{N}.csv'))
centers, dirs, rels = mc.trj2numpy(trj)
vrt_lattice = mc.vertices_lattice(a.magnitude,N,spos=(0,0))
indices_matrix = mc.indices_lattice(vrt_lattice,centers, a.magnitude, N)


file_path = f'../data/q2_degeneracy/s{N}.csv'

for i in range(1,2):
    print(f'===== realization {i} =====')
    simulated_annealing(file_path,int(1e5), centers, dirs, rels,i)