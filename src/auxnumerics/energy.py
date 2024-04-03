# ============================================================= 
# This is a module aimed at computing energies.
# It is separated since it is supposed to be precompiled
# Author: leogabac
# ============================================================= 


import os
import sys

sys.path.insert(0, '../icenumerics/')
import icenumerics as ice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit

ureg = ice.ureg
idx = pd.IndexSlice

@jit(nopython=True)
def precompiled_calculate_energy(magic_number,L,sel_particles):
    n = len(sel_particles)
    
    H = 0
    for i in range(n):
        for j in range(i+1,n):
            
            xi = sel_particles[i]
            xj = sel_particles[j]
            
            xij = xi - xj
            ox = np.array([xij[0], xij[0]+L, xij[0]-L])
            oy = np.array([xij[1], xij[1]+L, xij[1]-L])
            oz = np.array([xij[2], xij[2]+L, xij[2]-L])
            
            ix = np.argmin(np.abs(ox))
            iy = np.argmin(np.abs(oy))
            iz = np.argmin(np.abs(oz))
            
            xij_pbc = np.array([ox[ix], oy[iy], oz[iz]])
            
            distance = np.sqrt((xij_pbc**2).sum())
            rhat = xij_pbc/distance
            
            Bhat = np.array([1,0,0])
            dimensional = (magic_number/distance**3)
            
            # yes, this is a dot product done by hand
            Bdotr = Bhat[0]*rhat[0] + Bhat[1]*rhat[1] + Bhat[2]*rhat[2] 
        
            adimensional = 3*Bdotr**2 - 1
            
            H += dimensional*adimensional
    
    return H
            
            
    

