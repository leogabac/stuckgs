import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm

sys.path.insert(0, '../../../icenumerics/')
sys.path.insert(0, '../')

import icenumerics as ice
from parameters import params

import concurrent.futures

ureg = ice.ureg
idx = pd.IndexSlice


def create_simulation(params,ctrj,size,realization):
    N = size
    a = params["lattice_constant"]

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

    params['particle'] = particle
    params['trap'] = trap

    col = aux.trj2col(params,ctrj)

    world = ice.world(
        field = params["max_field"],
        temperature = params["sim_temp"],
        dipole_cutoff = params["sim_dipole_cutoff"],
        boundaries=['p', 'p', 'p'])

    col.simulation(world,
            name = f"./lammps_files/trj{realization}",
            include_timestamp = False,
            targetdir = r".",
            framerate = params["framespersec"],
            timestep = params["dt"],
            run_time = params["total_time"],
            output = ["x","y","z","mux","muy","muz"],
            processors=1)

    # Field

    # here the idea is to start from a 4in/4out state
    col.sim.field.fieldx = f'v_Bmag/6000e6*time*(time<6000e6)+v_Bmag*(time>=6000e6)'
    col.sim.field.fieldy = '0'
    col.sim.field.fieldz = '0'

    return col

def run_simulation(params,ctrj,size,realization):

    col = create_simulation(params,ctrj,size,realization)
    col.run_simulation()

def load_simulation(params,ctrj,data_path,size,realization):

    print(f'Saving {realization}...')
    col = create_simulation(params,size,realization)
    col.sim.base_name = os.path.join(col.sim.dir_name,col.sim.file_name)
    col.sim.script_name = col.sim.base_name+'.lmpin'
    col.sim.input_name = col.sim.base_name+'.lmpdata'
    col.sim.output_name = col.sim.base_name+'.lammpstrj'
    col.sim.log_name = col.sim.base_name+'.log'

    # saving trj
    ice.get_ice_trj_low_memory(col,dir_name=data_path)


SIZE = 30
REALIZATIONS = [1,2,3,4,5,6,7,8,9,10]

CTRJ_PATH = '/media/frieren/BIG/stuckgs/data/states/af4/30.csv'
DATA_PATH = '/media/frieren/BIG/stuckgs/data/sims_melting/'
GS = pd.read_csv(CTRJ_PATH,index_col='id')


params['max_field'] = 2*ureg.mT
params['total_time'] = 6100*ureg.s
params['framespersec'] = 1*ureg.Hz

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(
        executor.map(
            run_simulation,
            [params] * len(REALIZATIONS),
            [SIZE] * len(REALIZATIONS),
            REALIZATIONS,
        )
    )

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(
        executor.map(
            run_simulation,
            [params] * len(REALIZATIONS),
            [GS] * len(REALIZATIONS),
            [SIZE] * len(REALIZATIONS),
            REALIZATIONS,
        )
    )


for i in range(1,11):
    print(f'===== Realization {i} =====')
    load_simulation(params,DATA_PATH,SIZE,i)
