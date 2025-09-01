import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
from IPython.display import clear_output

import matplotlib.pyplot as plt

sys.path.insert(0, '../../icenumerics/')
sys.path.insert(0, './auxnumerics/')
import icenumerics as ice

from numba import jit
import auxiliary as aux
import montecarlo_tools as mc
import chirality_tools as chir
import energy
from parameters import params
ureg = ice.ureg

def get_energy_from_last_frame(magic_number,size,a,trjpath,filename):
        
        L = size*a
        
        filepath = os.path.join(trjpath,filename)
        trj = pd.read_csv(filepath, index_col=['frame','id'])
        
        last_frame = trj.index.get_level_values('frame').unique()[-1]
        
        sel_particles = aux.get_coordinates_at_frame(trj,last_frame).to_numpy()
        cenergy = energy.calculate_energy(magic_number,L,sel_particles)
        return cenergy
    

idx = pd.IndexSlice

sim_path = '/mnt/e/stuckgs/data/sims'
save_path = '../data/sims'

sizes = [i for i in range(10,30+1,2)]
magic_number = params['freedom']

size_energy = []
for i,size in enumerate(sizes):
    print(f'===== Size {size} =====')
    trjpath = os.path.join(sim_path,f'{size}/trj')
    renergy = [ 
               get_energy_from_last_frame(magic_number,int(size),params['lattice_constant'].magnitude,trjpath,cfile) 
               for cfile in tqdm(os.listdir(trjpath)) 
               ]
    
    print(renergy)
    size_energy.append(renergy)

edf = pd.DataFrame(size_energy)
edf['size'] = [int(s) for s in sizes]

edf.to_csv(os.path.join(save_path,'energiessize.csv'),index=False)
        
        