# =============================================================

# God bless whoever reads this code
# Author: leogabac
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
import math
import itertools

from tqdm import tqdm
from math import isclose
from numba import jit,prange,float64,int64,complex128

# own
sys.path.insert(0, '../../icenumerics/')
sys.path.insert(0, './auxnumerics/')
sys.path.insert(0, './testing/')
import icenumerics as ice
import vertices as vrt

# parameters
from parameters import params

ureg = ice.ureg
idx = pd.IndexSlice

DRIVE_MOUNT = '/mnt/BIG/'
PROJECT = 'stuckgs/data'
DATA_PATH = os.path.join(DRIVE_MOUNT,PROJECT,'sims_superslow_short')
all_files = [x for x in os.listdir(os.path.join(DATA_PATH)) if x.startswith('trj')]

for file in all_files:

    trj_file = os.path.join(DATA_PATH,file)
    vrt_file = os.path.join(DATA_PATH,file.replace('trj','vertices'))
    params['size'] = 30

    print('working with... ',trj_file)

    # creating the topology
    vrt_lattice = vrt.create_lattice(params['lattice_constant'].magnitude,params['size'])

    # Importing files
    trj = pd.read_csv(trj_file, index_col=['frame','id'])

    # Doing shit with the vertices
    frames = trj.index.get_level_values('frame').unique().to_list()

    for frame in tqdm(frames[::10]):

        # here the idea is to go frame by frame computing the topological charges
        # and generate the same structure than the vertices module from icenumerics

        # select the current frame, i could have done a group_by('frame') tehee :p
        sel_trj = trj.loc[idx[frame,:]]
        centers, dirs, rels = vrt.trj2numpy(sel_trj)

        # here i make sure the directions are normalized
        dirs = dirs / np.max(dirs)

        # topology shenanigans
        idx_lattice = vrt.indices_lattice(vrt_lattice,centers, params['lattice_constant'].magnitude, params['size'])
        q_frame = vrt.get_charge_lattice(idx_lattice,dirs)
        mask = np.where(q_frame == 0, 1, 0) # put 0 in nonzero, and 1 in zero
        # make all charged vertices have zero dipole
        dipoles = vrt.dipole_lattice(centers,dirs,rels,vrt_lattice,idx_lattice) * mask[:,:,np.newaxis]

        # now is time to reshape
        vrt_coord_list = vrt_lattice.reshape(params['size']**2,3)
        dip_list = dipoles.reshape(params['size']**2,3)
        q_list = q_frame.reshape(-1)

        # put together
        N = len(q_list)
        data = np.column_stack((
            [frame]*N,
            list(range(N)),
            vrt_coord_list[:,0],
            vrt_coord_list[:,1],
            [4]*N,
            q_list,
            dip_list[:,0],
            dip_list[:,1]
        ))

        df = pd.DataFrame(data,columns=['frame','vertex','x','y','coordination','charge','dx','dy'])

        if frame == 0:
            df.to_csv(vrt_file,index=False)
        else:
            df.to_csv(vrt_file,mode='a',index=False,header=False)
