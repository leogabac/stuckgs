from tqdm import tqdm

def is_convertible_to_integer(s):
    try:
        float_value = float(s)
        int_value = int(float_value)
        # Check if converting to integer doesn't lose precision
        return float_value == int_value
    except ValueError:
        return False

def reduce_realization(realization):
    input_fh = open(f'ctrj{realization}.csv','r')
    output_fh = open(f'lmctrj{realization}.csv','a')


    cur_line = 0 # initialize
    for line in tqdm(input_fh):

        # write the headers
        if cur_line==0:
            output_fh.write(line)
            cur_line += 1
            continue
        
        # get the current time, and check if it is an integer
        cur_time = line.split(',')[8]
        if is_convertible_to_integer(cur_time):
            output_fh.write(line)
        else:
            pass

        # update the line counter
        cur_line += 1

    

    input_fh.close()
    output_fh.close()


for realization in range(1,11):
    print(f'===== Realization {realization} =====')
    reduce_realization(realization)
