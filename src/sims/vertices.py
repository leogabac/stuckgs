import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "../../icenumerics/")
sys.path.insert(0, "../auxnumerics/")
sys.path.insert(0, "../")  # for parameters.py

import icenumerics as ice
import vertices as vrt
from parameters import params


def compute_vertices(data_dir, size, realizations):
    """
    Compute the vertices' CSV file with schema
    [frame, vertex, x, y, coordination, charge, dx, dy]

    Parameters:
    ----------
    - data_dir: str, directory where the trj files are located.
    - size: int, vertices per side.
    - realization: int
    """
    df_cols = [
        "frame",
        "vertex",
        "x",
        "y",
        "coordination",
        "charge",
        "dx",
        "dy",
    ]

    vrt_lattice = vrt.create_lattice(params["lattice_constant"].magnitude, size)
    for i in realizations:
        trj_file = os.path.join(data_dir, f"xtrj{i}.csv")
        vrt_file = os.path.join(data_dir, f"vertices{i}.csv")

        if os.path.isfile(vrt_file):
            print(f"[INFO] \t {vrt_file} exists, skipping")
            continue

        if not os.path.isfile(trj_file):
            raise FileExistsError(f"Cannot load {trj_file}")

        trj_obj = ice.trajectory(trj_file)
        trj_obj.load()

        # Doing shit with the vertices
        frames = trj_obj.trj.index.get_level_values("frame").unique().to_list()[::20]

        for frame in tqdm(frames):
            # here the idea is to go frame by frame computing the topological charges
            # and generate the same structure than the vertices module from icenumerics

            # select the current frame, i could have done a group_by('frame') tehee :p
            sel_trj = trj_obj.slice(frame)
            centers, dirs, rels = vrt.trj2numpy(sel_trj)

            # here i make sure the directions are normalized
            dirs = dirs / np.max(dirs)

            # topology shenanigans
            idx_lattice = vrt.indices_lattice(
                vrt_lattice,
                centers,
                params["lattice_constant"].magnitude,
                params["size"],
            )
            q_frame = vrt.get_charge_lattice(idx_lattice, dirs)
            dip_lattice = vrt.dipole_lattice(
                centers, dirs, rels, vrt_lattice, idx_lattice
            )
            # the dipole lattice will assign some total dipole to charged vertices
            # ( since they still have some direction )
            # so we need to make a mask that assigns 0 in nonzero vertices, and 1 in zeros
            mask = np.where(q_frame == 0, 1, 0)
            dipoles = dip_lattice * mask[:, :, np.newaxis]

            # now is time to reshape
            vrt_coord_list = vrt_lattice.reshape(size**2, 3)
            dip_list = dipoles.reshape(size**2, 3)
            q_list = q_frame.reshape(-1)

            # put together
            num_vertices = len(q_list)
            data = np.column_stack(
                (
                    [frame] * num_vertices,
                    list(range(num_vertices)),
                    vrt_coord_list[:, 0],
                    vrt_coord_list[:, 1],
                    [4] * num_vertices,
                    q_list,
                    dip_list[:, 0],
                    dip_list[:, 1],
                )
            )

            df = pd.DataFrame(data, columns=df_cols)
            if frame == 0:
                df.to_csv(vrt_file, index=False)
            else:
                df.to_csv(vrt_file, mode="a", index=False, header=False)
