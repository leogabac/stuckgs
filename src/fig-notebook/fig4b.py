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

idx = pd.IndexSlice

sim_path = '../data/sims/'

L = params['size']*params['lattice_constant'].magnitude
magic_number = params['freedom']

def get_energy_at_frame(trj,frame,quench_rate):
    """
        Calculate the energy of a trj at a given frame
        Assumes the framerate.
    """
    time  = frame/20
    field = quench_rate * time * ureg.mT
    
    dip_moment = np.pi * (2*params['particle_radius'])**3 *params['particle_susceptibility']*field/6/params['mu0']
    magic_number =  - (params['mu0']*dip_moment**2/4/np.pi).to(ureg.pN * ureg.nm * ureg.um**3).magnitude
    
    L  = 30 * params['lattice_constant'].magnitude
    sel_particles = aux.get_coordinates_at_frame(trj,frame).to_numpy()
    return energy.calculate_energy(magic_number,L,sel_particles) 

for i,realization in enumerate(range(1,10+1)):
    print(f'===== Realization {realization} =====')
    filepath = os.path.join(sim_path,'30','trj',f'trj{realization}.csv')
    
    if not os.path.isfile(filepath):
        continue
    
    print(f'Loading... {filepath}')
    trj = pd.read_csv(filepath, index_col=['frame','id'])
    
    frames = trj.index.get_level_values('frame').unique().to_list()
    
    # These frames are each second, cuz framerate is 20
    sframes = frames[::20]
    goodframes = sframes[:301]
    qrate = params['max_field'].magnitude / 300
    # The 300 at the end means the first 300s
    energies = [get_energy_at_frame(trj,frame,qrate) for frame in tqdm(goodframes)]
    
    energydf = pd.DataFrame(energies, columns=['energy'])
    energydf['frame'] = goodframes
    energydf['realization'] = [realization] * len(goodframes)
    
    savename = os.path.join(sim_path,'energiestime.csv',)
    if i == 0:
        energydf.to_csv(savename, index=False)
    else:
        energydf.to_csv(savename, mode='a',index=False,header=False)



# This is completely a copy-paste and is a mess
#print('===== Computing GS =====')

#energies = pd.read_csv(os.path.join(sim_path,'energiestime.csv'), index_col=['realization','frame'])
#frames = energies.index.get_level_values('frame').unique().to_list()
#realizations = energies.index.get_level_values('realization').unique().to_list()
#time = np.array(frames) / params['framespersec'].magnitude
#field = 10/300 * time
#
#gs = pd.read_csv('../data/states/af4/30.csv', index_col='id')
#gs_energies = []
#for cfield in tqdm(field):
#    ufield = cfield*ureg.mT
#    dip_moment = np.pi * (2*params['particle_radius'])**3 *params['particle_susceptibility']*ufield/6/params['mu0']
#    magic_number =  - (params['mu0']*dip_moment**2/4/np.pi).to(ureg.pN * ureg.nm * ureg.um**3).magnitude
#    L  = 30 * params['lattice_constant'].magnitude
#    sel_particles = aux.get_positions_from_ctrj(gs).to_numpy()
#    gs_energy = energy.calculate_energy(magic_number,L,sel_particles)
#    gs_energies.append(gs_energy)
    
