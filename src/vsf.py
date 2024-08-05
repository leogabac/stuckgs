import os
import sys

import numpy as np
import math
import pandas as pd

from tqdm import tqdm

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
from itertools import combinations


ureg = ice.ureg
idx = pd.IndexSlice


def reciprocal_space(N,a):
    amount_bz = 3.5
    klim = np.pi/a * amount_bz
    kx = np.linspace(-klim,klim,120) 
    ky = np.linspace(-klim,klim,120) 

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
reciprocal_lattice = reciprocal_space(N,a.magnitude)
rs_indices = np.array(np.meshgrid(
    np.arange(reciprocal_lattice.shape[0]),
    np.arange(reciprocal_lattice.shape[0])
    )).T.reshape(-1,2)
pairs = np.asarray(list(combinations(np.arange(len(centers)),2)) + [(i,i) for i in range(len(centers))] )


# good stuff
print("computing magnetic structure factor...")
with ProgressBar(total=len(rs_indices)) as progress:
    msf = mc.vector_msf(
        pairs,
        centers,
        dirs,
        N,
        a.magnitude,
        reciprocal_lattice,
        rs_indices,
        progress)

print("saving...")
pd.DataFrame(msf).to_csv(f'../data/q2_degeneracy/msf{N}.csv', header=False, index=False)
