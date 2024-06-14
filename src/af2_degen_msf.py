import os
import sys

import numpy as np
import math
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
from numba_progress import ProgressBar


ureg = ice.ureg
idx = pd.IndexSlice


def reciprocal_space(N,a):
    kx = 2*np.pi*np.fft.fftshift( np.fft.fftfreq(N,d=a) )
    ky = 2*np.pi*np.fft.fftshift( np.fft.fftfreq(N,d=a) ) 

    KX, KY = np.meshgrid(kx,ky)
    N = len(kx)
    reciprocal_space = np.zeros((N,N,2))
    reciprocal_space[:,:,0] = KX
    reciprocal_space[:,:,1] = KY

    return reciprocal_space



# initialization
N = 20
print(f"N: {N}")
a = params["lattice_constant"]

file_path = f'../data/q2_degeneracy/s{N}.csv'
trj_final = pd.read_csv(file_path, index_col=['realization','frame','id'])
trj = trj_final.loc[idx[1,:,:]]


# topology
print("making topology...")
centers, dirs, rels = mc.trj2numpy(trj)
vrt_lattice = mc.vertices_lattice(a.magnitude,N,spos=(0,0))
indices_matrix = mc.indices_lattice(vrt_lattice,centers,a.magnitude,N)
arrow_lattice = mc.dipole_lattice(centers,dirs,rels, vrt_lattice, indices_matrix)
reciprocal_lattice = reciprocal_space(N,a.magnitude)


# good stuff
print("computing magnetic structure factor...")
pairwise_indices = np.array([[i,j] for i in range(N) for j in range(N)])

with ProgressBar(total=len(pairwise_indices)) as progress:
    msf = mc.magnetic_structure_factor(
        reciprocal_lattice,
        arrow_lattice,
        vrt_lattice,
        N,
        a.magnitude,
        pairwise_indices,
        progress)

print("saving...")
pd.DataFrame(msf).to_csv(f'../data/q2_degeneracy/msf{N}.csv', header=False, index=False)
