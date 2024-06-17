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


data_path = '../data/simstair'

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
    

    # pt1 = "v_Bmag/300e6*time*(time<30e6)+1*(time>=30e6)*(time<50e6)+"


    pt1 = "v_Bmag/300e6*time*(time<30e6)+0.1*v_Bmag*(time>=30e6)*(time<330e6)+"
    pt2 = "(v_Bmag/300e6*(time-330e6)+0.1*v_Bmag)*(time>=330e6)*(time<360e6)+0.2*v_Bmag*(time>=360e6)*(time<660e6)+"
    pt3 = "(v_Bmag/300e6*(time-660e6)+0.2*v_Bmag)*(time>=660e6)*(time<690e6)+0.3*v_Bmag*(time>=690e6)*(time<990e6)+"
    pt4 = "(v_Bmag/300e6*(time-990e6)+0.3*v_Bmag)*(time>=990e6)*(time<1020e6)+0.4*v_Bmag*(time>=1020e6)*(time<1320e6)+"
    pt5 = "(v_Bmag/300e6*(time-1320e6)+0.4*v_Bmag)*(time>=1320e6)*(time<1350e6)+0.5*v_Bmag*(time>=1350e6)*(time<1650e6)+"
    pt6 = "(v_Bmag/300e6*(time-1650e6)+0.5*v_Bmag)*(time>=1650e6)*(time<1680e6)+0.6*v_Bmag*(time>=1680e6)*(time<1980e6)+"
    pt7 = "(v_Bmag/300e6*(time-1980e6)+0.6*v_Bmag)*(time>=1980e6)*(time<2010e6)+0.7*v_Bmag*(time>=2010e6)*(time<2310e6)+"
    pt8 = "(v_Bmag/300e6*(time-2310e6)+0.7*v_Bmag)*(time>=2310e6)*(time<2340e6)+0.8*v_Bmag*(time>=2340e6)*(time<2640e6)+"
    pt9 = "(v_Bmag/300e6*(time-2640e6)+0.8*v_Bmag)*(time>=2640e6)*(time<2670e6)+0.9*v_Bmag*(time>=2670e6)*(time<2970e6)+"
    pt10 = "(v_Bmag/300e6*(time-2970e6)+0.9*v_Bmag)*(time>=2970e6)*(time<3000e6)+1.0*v_Bmag*(time>=3000e6)*(time<3300e6)"
    col.sim.field.fieldx = pt1+pt2+pt3+pt4+pt5+pt6+pt7+pt8+pt9+pt10
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
    


size = 30
realizations = [1,2,3,4,5,6,7,8,9,10]
DATA_PATH = '../data/simstair/'

try:
    SIZE_PATH = os.path.join(DATA_PATH,str(size))
    os.mkdir(SIZE_PATH)
except:
    pass


#with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#    results = list(
#        executor.map(
#            run_simulation,
#            [params] * len(realizations),
#            [size] * len(realizations),
#            realizations,
#        )
#    )

for i in range(1,11):
     print(f'===== Realization {i} =====')
     load_simulation(params,SIZE_PATH,size,i)
