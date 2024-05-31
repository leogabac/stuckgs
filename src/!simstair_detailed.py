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
    # I want to make those tiny increments between 0-1 and 1-2

    rate = "v_Bmag/300e6"

    stair=[ 
        "v_Bmag/300e6*time*(time<30e6)+1*(time>=30e6)*(time<330e6)+", 
        "(v_Bmag/300e6*(time-330e6)+1)*(time>=330e6)*(time<333e6)+1.1*(time>=333e6)*(time<633e6)+",
        "(v_Bmag/300e6*(time-633e6)+1.1)*(time>=633e6)*(time<636e6)+1.2*(time>=636e6)*(time<936e6)+",
        "(v_Bmag/300e6*(time-936e6)+1.2)*(time>=936e6)*(time<939e6)+1.3*(time>=939e6)*(time<1239e6)+",
        "(v_Bmag/300e6*(time-1239e6)+1.3)*(time>=1239e6)*(time<1242e6)+1.4*(time>=1242e6)*(time<1542e6)+",
        "(v_Bmag/300e6*(time-1542e6)+1.4)*(time>=1542e6)*(time<1545e6)+1.5*(time>=1545e6)*(time<1845e6)+",
        "(v_Bmag/300e6*(time-1845e6)+1.5)*(time>=1845e6)*(time<1848e6)+1.6*(time>=1848e6)*(time<2148e6)+",
        "(v_Bmag/300e6*(time-2148e6)+1.6)*(time>=2148e6)*(time<2151e6)+1.7*(time>=2151e6)*(time<2451e6)+",
        "(v_Bmag/300e6*(time-2451e6)+1.7)*(time>=2451e6)*(time<2454e6)+1.8*(time>=2454e6)*(time<2754e6)+",
        "(v_Bmag/300e6*(time-2754e6)+1.8)*(time>=2754e6)*(time<2757e6)+1.9*(time>=2757e6)*(time<3057e6)+",
        "(v_Bmag/300e6*(time-3057e6)+1.9)*(time>=3057e6)*(time<3060e6)+2*(time>=3060e6)*(time<336e6)+",
        "(v_Bmag/300e6*(time-3360e6)+2)*(time>=3360e6)*(time<3390e6)+3*(time>=3390e6)*(time<3690e6)+",
        "(v_Bmag/300e6*(time-3690e6)+3)*(time>=3690e6)*(time<3720e6)+4*(time>=3720e6)*(time<4020e6)+",
        "(v_Bmag/300e6*(time-4020e6)+4)*(time>=4020e6)*(time<4050e6)+5*(time>=4050e6)*(time<4350e6)+",
        "(v_Bmag/300e6*(time-4350e6)+5)*(time>=4350e6)*(time<4380e6)+6*(time>=4380e6)*(time<4680e6)+",
        "(v_Bmag/300e6*(time-4680e6)+6)*(time>=4680e6)*(time<4710e6)+7*(time>=4710e6)*(time<5010e6)+",
        "(v_Bmag/300e6*(time-5010e6)+7)*(time>=5010e6)*(time<5040e6)+8*(time>=5040e6)*(time<5340e6)+",
        "(v_Bmag/300e6*(time-5340e6)+8)*(time>=5340e6)*(time<5370e6)+9*(time>=5370e6)*(time<5670e6)+",
        "(v_Bmag/300e6*(time-5670e6)+9)*(time>=5670e6)*(time<5700e6)+10*(time>=5700e6)*(time<6000e6)"
]

    # Then with 3 s increments up to 2 mT

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

    #print('load simulation')
    #col.load_simulation()

    print('making paths')
    trj_path = os.path.join(data_path,'trj')
    ctrj_path = os.path.join(data_path,'ctrj')

    try:
        os.mkdir(trj_path)
        os.mkdir(ctrj_path)
    except:
        pass
    

   #filename = f"trj{realization}.csv"
   
    #print('saving usual trj')
    #col.trj.to_csv(os.path.join(trj_path,filename))

    #filename = f"ctrj{realization}.csv"
    print('low memory ctrj')
    ice.get_ice_trj_low_memory(col,dir_name=ctrj_path)
    #trj = ice.get_ice_trj(col.trj, bounds = col.bnd)
    #trj.to_csv(os.path.join(ctrj_path,filename))
    


size = 2
realizations = [1]
DATA_PATH = '../data/simstair_detailed/'

try:
    SIZE_PATH = os.path.join(DATA_PATH,str(size))
    os.mkdir(SIZE_PATH)
except:
    pass


with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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
