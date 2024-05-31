

def evolve_field(rate,stime,dinc,devo,sB,dB):
    return f"({rate}*(time-{stime}e6)+{sB})*(time>={stime}e6)*(time<{stime+dinc}e6)+{sB+dB}*(time>={stime+dinc}e6)*(time<{stime+dinc+devo}+e6)+"


rate = "v_Bmag/300e6"



starting_field = 2 
field_increment = 1 
ramp_time = 30
evo_time = 300 
starting_time = 3360  

for _ in range(9):
    print(evolve_field(rate, starting_time , ramp_time, evo_time ,starting_field, field_increment))
    starting_time = starting_time + ramp_time + evo_time
    starting_field = starting_field + field_increment
