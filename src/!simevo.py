import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
from IPython.display import clear_output

import matplotlib as mpl 
#mpl.use('pgf')
import matplotlib.pyplot as plt

sys.path.insert(0, '../../icenumerics/')
sys.path.insert(0, './auxnumerics/')

import icenumerics as ice

import auxiliary as aux
import montecarlo_tools as mcb
import chirality_tools as chir
from parameters import params

import concurrent.futures

ureg = ice.ureg
idx = pd.IndexSlice


data_path = '../data/afevo'

def create_simulation(params,ctrj,size,realization):
    sp = ice.spins()
    
    N = size
    a = params["lattice_constant"]
    
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
    
    params['particle'] = particle
    params['trap'] = trap
    
    col = aux.get_colloids_from_ctrj2(ctrj,params)

    world = ice.world(
            field = params["max_field"],
            temperature = params["sim_temp"],
            dipole_cutoff = params["sim_dipole_cutoff"],
            boundaries=['p', 'p', 'p'])


    col.simulation(world,
                name = f"./lammps_files/ctrj{realization}",
                include_timestamp = False,
                targetdir = r".",
                framerate = params["framespersec"],
                timestep = params["dt"],
                run_time = params["total_time"],
                output = ["x","y","z","mux","muy","muz"],
                processors=1)

    # Field
    

    col.sim.field.fieldx = "v_Bmag/300e6*time*(time<300e6)+v_Bmag*(time>=300e6)"
    col.sim.field.fieldy = "0"
    col.sim.field.fieldz = "0"

    return col

def run_simulation(params,ctrj,size,realization):

    col = create_simulation(params,ctrj,size,realization)
    col.run_simulation()

def load_simulation(params,data_path,ctrj,size,realization):

    col = create_simulation(params,ctrj,size,realization)
    col.sim.base_name = os.path.join(col.sim.dir_name,col.sim.file_name)
    col.sim.script_name = col.sim.base_name+'.lmpin'
    col.sim.input_name = col.sim.base_name+'.lmpdata'
    col.sim.output_name =  col.sim.base_name+'.lammpstrj'
    col.sim.log_name =  col.sim.base_name+'.log'

    col.load_simulation()

    trj_path = os.path.join(data_path,'trj')
    ctrj_path = os.path.join(data_path,'ctrj')

    try:
        os.mkdir(trj_path)
        os.mkdir(ctrj_path)
    except:
        pass
    

    filename = f"trj{realization}.csv"
    col.trj.to_csv(os.path.join(trj_path,filename))

    #filename = f"ctrj{realization}.csv"
    ice.get_ice_trj_low_memory(col,dir_name=ctrj_path)
    #trj = ice.get_ice_trj(col.trj, bounds = col.bnd)
    #trj.to_csv(ctrj_path + filename)
    

size = 30
realizations = [1,2,3,4,5,6,7,8,9,10]
CTRJ_PATH = '../data/states/af4/30.csv'
DATA_PATH = '../data/afevo/'

ctrj = pd.read_csv(CTRJ_PATH,index_col='id')



try:
    SIZE_PATH = os.path.join(DATA_PATH,str(size))
    os.mkdir(SIZE_PATH)
except:
    pass

# with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#     results = list(
#         executor.map(
#             run_simulation,
#             [params] * len(realizations),
#             [ctrj] * len(realizations),
#             [size] * len(realizations),
#             realizations,
#         )
#     )

for i in range(1,10+1):
    print(f'===== Realization {i} =====')
    load_simulation(params,SIZE_PATH,ctrj,size,i)