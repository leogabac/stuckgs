import os
import sys
import numpy


def reduce_trj(ogfile,tgfile,save_each):

    fh = open(ogfile,'r')

    # loop the lines
    for line in fh:

        # split with , and get the frame number
        line_split = line.strip().split(',')
        frame_no = line_split[-2]

        # write the header
        # or write the correct frame
        if frame_no=='frame' or int(frame_no) % save_each==0 or int(frame_no)==LAST_FRAME:
            with open(tgfile,'a') as tg:
                tg.write(line)

    fh.close()
    return 0


LAST_FRAME = 999999
datapath = '/media/frieren/BIG/stuckgs/data/metropolis/1M/'
all_files = [x for x in os.listdir(os.path.join(datapath,'OG')) if x.startswith('trj')]

for og_filename in all_files:
    print('working with... ',og_filename)
    og_file = os.path.join(datapath,'OG',og_filename)
    tg_file = os.path.join(datapath,og_filename)
    reduce_trj(og_file,tg_file,50)
