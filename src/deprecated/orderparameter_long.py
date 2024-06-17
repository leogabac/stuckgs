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

data_path = '../data/simstair/30'
vrt_temp = pd.read_csv(os.path.join(data_path,'vertices','vertices1.csv'),index_col=['frame','vertex'])
vrt_frames = vrt_temp.index.get_level_values('frame').unique().to_list()

h = 300
intervals = [
 (30, 330),
 (360, 660),
 (690, 990),
 (1020, 1320),
 (1350, 1650),
 (1680, 1980),
 (2010, 2310),
 (2340, 2640),
 (2670, 2970),
 (3000, 3300),
]

def is_between(x,low,high):
    return (x >= low) and (x<=high)

# start looping the intervals
for i,(stime,etime) in enumerate(intervals):
    print(i,stime, etime)

    # get the relevant frames to focus in (relaxation)
    startframe = stime * params['framespersec'].magnitude
    endframe = etime * params['framespersec'].magnitude
    analysis_frames = [i for i in vrt_frames if is_between(i,startframe,endframe)]

    # get the order parameter in this interval
    psi = []
    for frame in tqdm(analysis_frames):
        results = [chir.get_charge_order_on_frame_on_realization(params,data_path,frame,realization,tol=1.6) for realization in range(1,10+1) ]
        psi.append(results)
    
    cur_df = pd.DataFrame(psi)
    cur_df['frame'] = analysis_frames
    cur_df['field'] = [i+1]*len(psi) 

    file2make = os.path.join(data_path,'orderparameters.csv')

    if i == 0:
        cur_df.to_csv(file2make,index=False)
    else:
        cur_df.to_csv(file2make,mode='a',index=False, header=False)
