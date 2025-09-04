import os
import sys

sys.path.insert(0, "../../icenumerics/")
sys.path.insert(0, "../auxnumerics/")
sys.path.insert(0, "../")  # for parameters.py


from tqdm import tqdm
from parameters import params
import vertices as vrt
import auxiliary as aux
import concurrent.futures
import icenumerics as ice
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path


ureg = ice.ureg
idx = pd.IndexSlice

# ==============================================================================
# SIMULATION FUNCTIONS
# ==============================================================================


def create_simulation(params, trj, size, realization):
    # magnetic field definition
    fx = ["v_Bmag"]
    fy = ["0"]
    fz = ["0"]

    params["size"] = size
    N = size
    a = params["lattice_constant"]

    print(f"time sanity check: {params['total_time']}")

    sp = ice.spins()
    sp.create_lattice("square", [N, N], lattice_constant=a, border="periodic")

    particle = ice.particle(
        radius=params["particle_radius"],
        susceptibility=params["particle_susceptibility"],
        diffusion=params["particle_diffusion"],
        temperature=params["particle_temperature"],
        density=params["particle_density"],
    )

    trap = ice.trap(
        trap_sep=params["trap_sep"],
        height=params["trap_height"],
        stiffness=params["trap_stiffness"],
    )

    params["particle"] = particle
    params["trap"] = trap

    col = aux.trj2col(params, trj)

    world = ice.world(
        field=params["max_field"],
        temperature=params["sim_temp"],
        dipole_cutoff=params["sim_dipole_cutoff"],
        boundaries=["p", "p", "p"],
    )

    col.simulation(
        world,
        name=os.path.join(LAMMPS_DIR, f"trj{realization}"),
        include_timestamp=False,
        targetdir=r".",
        framerate=params["framespersec"],
        timestep=params["dt"],
        run_time=params["total_time"],
        output=["x", "y", "z", "mux", "muy", "muz"],
        processors=1,
    )

    col.sim.field.fieldx = "".join(fx)
    col.sim.field.fieldy = "".join(fy)
    col.sim.field.fieldz = "".join(fz)

    return col


def run_simulation(params, trj, size, realization):
    col = create_simulation(params, trj, size, realization)
    col.run_simulation()


def load_simulation(params, trj, data_path, size, realization):
    print(f"Saving {realization}...")
    col = create_simulation(params, trj, size, realization)
    col.sim.base_name = os.path.join(col.sim.dir_name, col.sim.file_name)
    col.sim.script_name = col.sim.base_name + ".lmpin"
    col.sim.input_name = col.sim.base_name + ".lmpdata"
    col.sim.output_name = col.sim.base_name + ".lammpstrj"
    col.sim.log_name = col.sim.base_name + ".log"
    trj_path = os.path.join(data_path, "trj")

    try:
        os.mkdir(trj_path)
    except:
        pass

    ice.get_ice_trj_low_memory(col, dir_name=trj_path)


def load_initial_condiiton(filepath):
    trj = pd.read_csv(filepath,index_col = ['id'])
    return trj


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================


REPO_ROOT = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip()

SCRIPT = os.path.basename(__file__).split(".")[0]
DATA_DIR = os.path.join(REPO_ROOT, "data", SCRIPT)
INIT_COND_DIR = os.path.join(DATA_DIR, "initial-conditions")  # initial conditions
LAMMPS_DIR = os.path.join(DATA_DIR, "lammps-files")


# creating data directories
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(LAMMPS_DIR).mkdir(parents=True, exist_ok=True)

print(f"[INFO] \t saving data to: {DATA_DIR}")
print(f"[INFO] \t lammps files: {LAMMPS_DIR}")

# raise error if there is no dir with initial conditions
if not os.path.isdir(INIT_COND_DIR):
    raise FileNotFoundError(f"Directory does not exist: {INIT_COND_DIR}")


SIZE = 30
REALIZATIONS = list(range(1, 11))
FIELD = 20

# running the simulations

params["max_field"] = FIELD * ureg.mT
params["total_time"] = 3600 * ureg.s

print("=" * 80)
print("INFO")
print("=" * 80)

print(f"max field: \t {params['max_field']}")
print(f"total time: \t {params['total_time']}")

# TODO:
# load those initial conditions
# pass the initial conditions as a list for parallelization
# run and see what happens

# this should be a list of 10 trajectories of the last simulation frame
sim_type = 'fast'
initial_conditions = [load_initial_condiiton(os.path.join(INIT_COND_DIR, f"{sim_type}-{realization}.csv")) for realization in REALIZATIONS]

with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
    results = list(
        executor.map(
            run_simulation,
            [params] * len(REALIZATIONS),
            initial_conditions,
            [int(SIZE)] * len(REALIZATIONS),
            REALIZATIONS,
        )
    )
