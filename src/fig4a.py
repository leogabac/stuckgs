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

print("COMPUTING RPARALLELS")

sim_path = '../data/simstair_detailed/'


def get_rp_on_realization(sim_path,realization):
    ctrj = pd.read_csv(os.path.join(sim_path,'30','ctrj',f'xtrj{realization}.csv'), index_col=[0,1])
    particles = ctrj.index.get_level_values('id').unique().to_list()
    frames = ctrj.index.get_level_values('frame').unique().to_list()
    sframes = frames[::1]
    
    ts = []

    for cur_particle in tqdm(particles):

        single_ts = [aux.get_rparalell(ctrj,cur_particle,frame) for frame in sframes]
        ts.append(single_ts)
            
    dfts = pd.DataFrame(data = np.array(ts).transpose())
    pnumbers = list(dfts.columns)
    dfts['frame'] = sframes
    dfts['realization'] = [realization]*len(dfts)
    dfts = dfts[['realization','frame'] + pnumbers]
    
    file2make = os.path.join(sim_path,'rparallels.csv')
    
    if realization == 1:
        dfts.to_csv(file2make,index=False)
    else:
        dfts.to_csv(file2make,mode='a',index=False, header=False)
    
    return None


realizations = list(range(6,11))

for i in range(1,11):
    print(f'===== Realization {i} =====')
    get_rp_on_realization(sim_path,i)
    
