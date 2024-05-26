# ============================================================= 
# Script to compute the vertices for all sizes
# God bless whoever reads this code
# Author: leogabac
# ============================================================= 


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

import auxiliary as aux
import montecarlo_tools as mc
import chirality_tools as chir

from parameters import params

ureg = ice.ureg

idx = pd.IndexSlice

sizes = ['30']
data_path = r'../data/simstair/'

print("COMPUTING VERTICES")
for strsize in sizes:
    print(f"===== size {strsize} =====")
    params['size'] = int(strsize)
    
    trj_path = os.path.join(data_path,strsize,"trj")
    ctrj_path = os.path.join(data_path,strsize,"ctrj")
    vrt_path = os.path.join(data_path,strsize,"vertices")
    
    try:
        os.mkdir(vrt_path)
    except:
        pass

    # Get the number of realizations

    for i in range(1,10+1):

        ctrj_file = os.path.join(ctrj_path,f"xtrj{i}.csv")
        vrt_file = os.path.join(vrt_path,f"vertices{i}.csv")
        
        
        if os.path.isfile(vrt_file):
            continue
        
        # Importing files
        print(f"- realization {i} -")
        try:
            ctrj_raw = pd.read_csv(ctrj_file, index_col=[0,1])
        except:
            print(f"There is no such trj. Skipping")
            continue

        # Doing shit with the vertices
        v = ice.vertices()
        frames = ctrj_raw.index.get_level_values("frame").unique()

        v.trj_to_vertices(ctrj_raw.loc[frames[::1]])

        print(f"Saving vertices to " + vrt_file)
        v.vertices.to_csv(vrt_file)
