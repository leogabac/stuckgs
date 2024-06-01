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


data_path = '../data/simstair_detailed'

def create_simulation(params,size,realization):
    
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
    
    
    col = ice.colloidal_ice(sp, particle, trap,
                            height_spread = params["height_spread"], 
                            susceptibility_spread = params["susceptibility_spread"],
                            periodic = params["isperiodic"])

        
    col.randomize()
    col.region = np.array([[0,0,-3*(params["particle_radius"]/a/N).magnitude],[1,1,3*(params["particle_radius"]/a/N).magnitude]])*N*a


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
    

    # Here i have a rate of 10/300 mT, to 
    # 30s will increment 1mT, but 3s will increment 0.1 mT
    # I want to make those tiny increments between 1-2

    stair=[ 
        "v_Bmag/300e6*time*(time<30e6)+0.1*v_Bmag*(time>=30e6)*(time<330e6)+", 
        "(v_Bmag/300e6*(time-330e6)+0.1*v_Bmag)*(time>=330e6)*(time<333e6)+0.11*v_Bmag*(time>=333e6)*(time<633e6)+",
        "(v_Bmag/300e6*(time-633e6)+0.11*v_Bmag)*(time>=633e6)*(time<636e6)+0.12*v_Bmag*(time>=636e6)*(time<936e6)+",
        "(v_Bmag/300e6*(time-936e6)+0.12*v_Bmag)*(time>=936e6)*(time<939e6)+0.13*v_Bmag*(time>=939e6)*(time<1239e6)+",
        "(v_Bmag/300e6*(time-1239e6)+0.13*v_Bmag)*(time>=1239e6)*(time<1242e6)+0.14*v_Bmag*(time>=1242e6)*(time<1542e6)+",
        "(v_Bmag/300e6*(time-1542e6)+0.14*v_Bmag)*(time>=1542e6)*(time<1545e6)+0.15*v_Bmag*(time>=1545e6)*(time<1845e6)+",
        "(v_Bmag/300e6*(time-1845e6)+0.15*v_Bmag)*(time>=1845e6)*(time<1848e6)+0.16*v_Bmag*(time>=1848e6)*(time<2148e6)+",
        "(v_Bmag/300e6*(time-2148e6)+0.16*v_Bmag)*(time>=2148e6)*(time<2151e6)+0.17*v_Bmag*(time>=2151e6)*(time<2451e6)+",
        "(v_Bmag/300e6*(time-2451e6)+0.17*v_Bmag)*(time>=2451e6)*(time<2454e6)+0.18*v_Bmag*(time>=2454e6)*(time<2754e6)+",
        "(v_Bmag/300e6*(time-2754e6)+0.18*v_Bmag)*(time>=2754e6)*(time<2757e6)+0.19*v_Bmag*(time>=2757e6)*(time<3057e6)+",
        "(v_Bmag/300e6*(time-3057e6)+0.19*v_Bmag)*(time>=3057e6)*(time<3060e6)+0.2*v_Bmag*(time>=3060e6)*(time<3360e6)+",
        "(v_Bmag/300e6*(time-3360e6)+0.2*v_Bmag)*(time>=3360e6)*(time<3390e6)+0.3*v_Bmag*(time>=3390e6)*(time<3690e6)+",
        "(v_Bmag/300e6*(time-3690e6)+0.3*v_Bmag)*(time>=3690e6)*(time<3720e6)+0.4*v_Bmag*(time>=3720e6)*(time<4020e6)+",
        "(v_Bmag/300e6*(time-4020e6)+0.4*v_Bmag)*(time>=4020e6)*(time<4050e6)+0.5*v_Bmag*(time>=4050e6)*(time<4350e6)+",
        "(v_Bmag/300e6*(time-4350e6)+0.5*v_Bmag)*(time>=4350e6)*(time<4380e6)+0.6*v_Bmag*(time>=4380e6)*(time<4680e6)+",
        "(v_Bmag/300e6*(time-4680e6)+0.6*v_Bmag)*(time>=4680e6)*(time<4710e6)+0.7*v_Bmag*(time>=4710e6)*(time<5010e6)+",
        "(v_Bmag/300e6*(time-5010e6)+0.7*v_Bmag)*(time>=5010e6)*(time<5040e6)+0.8*v_Bmag*(time>=5040e6)*(time<5340e6)+",
        "(v_Bmag/300e6*(time-5340e6)+0.8*v_Bmag)*(time>=5340e6)*(time<5370e6)+0.9*v_Bmag*(time>=5370e6)*(time<5670e6)+",
        "(v_Bmag/300e6*(time-5670e6)+0.9*v_Bmag)*(time>=5670e6)*(time<5700e6)+1.0*v_Bmag*(time>=5700e6)*(time<6000e6)"
]


    col.sim.field.fieldx = "".join(stair)
    col.sim.field.fieldy = "0"
    col.sim.field.fieldz = "0"

    return col

def run_simulation(params,size,realization):

    col = create_simulation(params,size,realization)
    col.run_simulation()

def load_simulation(params,data_path,size,realization):

    print(f'Saving {realization}...')
    col = create_simulation(params,size,realization)
    col.sim.base_name = os.path.join(col.sim.dir_name,col.sim.file_name)
    col.sim.script_name = col.sim.base_name+'.lmpin'
    col.sim.input_name = col.sim.base_name+'.lmpdata'
    col.sim.output_name =  col.sim.base_name+'.lammpstrj'
    col.sim.log_name =  col.sim.base_name+'.log'

    ctrj_path = os.path.join(data_path,'ctrj')

    try:
        os.mkdir(ctrj_path)
    except:
        pass
    

    ice.get_ice_trj_low_memory(col,dir_name=ctrj_path)
    


size = 30
realizations = [1,2,3,4,5,6,7,8,9,10]
DATA_PATH = '../data/simstair_detailed/'

try:
    SIZE_PATH = os.path.join(DATA_PATH,str(size))
    os.mkdir(SIZE_PATH)
except:
    pass


with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
    results = list(
        executor.map(
            run_simulation,
            [params] * len(realizations),
            [size] * len(realizations),
            realizations,
        )
    )

for i in range(1,11):
     print(f'===== Realization {i} =====')
     load_simulation(params,SIZE_PATH,size,i)
