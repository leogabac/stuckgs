# ============================================================= 
# Modifications to montecarlo_tools
# The idea is to not rely on pandas, and make it faster
# God bless whoever reads this code
# Author: leogabac
# ============================================================= 

import os
import sys

sys.path.insert(0, '../icenumerics/')
import icenumerics as ice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from math import isclose

from auxiliary import *

from numba import jit


ureg = ice.ureg
idx = pd.IndexSlice

def trj2numpy(trj):
    """
        Takes a trj df and converts it to numpy arrays separated by centers, directions, and relative coordinate.
        ----------
        Parameters
        * trj: trj dataframe
    """
    centers = trj[['x','y','z']].to_numpy()
    dirs = trj[['dx','dy','dz']].to_numpy()
    rels = trj[['cx','cy','cz']].to_numpy()
    return centers,dirs,rels

def numpy2trj(centers,dirs,rels):
    """
        Takes numpy arrays and converts them into a trj dataframe
        ----------
        Parameters
        * centers: center of the trap
        * dirs: normalized direction
        * rels: relative position to the center of the trap
    """

    trj = np.concatenate([centers,dirs,rels],axis=1)
    trj = pd.DataFrame(trj,columns=['x','y','z','dx','dy','dz','cx','cy','cz'])
    trj['id'] = list(range(len(trj)))
    trj['frame'] = [0]*len(trj)
    trj = trj.set_index(['frame', 'id'])
    return trj

@jit(nopython=True)
def flip_spin(dirs,rels,idx):
    """
        Flips the spin at idx.
        ----------
        Parameters
        * centers: center of the trap
        * dirs: normalized direction
        * rels: relative position to the center of the trap
    """

    # sanity check
    dirs1 = dirs.copy()
    rels1 = rels.copy()
    
    dirs1[idx] = -dirs1[idx]
    rels1[idx] = -rels1[idx]

    return dirs1, rels1

@jit(nopython=True)
def get_idx_from_position(centers,pos,tol=0.1):
    """
        Get the index in the centers array from a position vector.
        ----------
        * centers: centers of the traps
        * pos: np array with a 3D coordinate
    """
    
    for i,center in enumerate(centers):
        distance = np.linalg.norm(center - pos)
        if np.isclose(0,distance,atol=tol):
            return i

def is_horizontal(direction):
    """
        Checks if a given direction is horizontal.
        ----------
        Parameters:
        * direction
    """
    x = np.array([1,0,0])
    dotP = np.dot(direction,x)

    if isclose(abs(dotP),1,rel_tol=1e-3):
        return True
    else:
        return False

@jit(nopython=True)
def fix_position(position,a,size):
    """
        Fixes the position to fit in the box
        0 < x < size*a, and 
        0 < y < size*a
        ----------
        Parameters:
        * position: Position vector in 3D
        * a: lattice constant
        * size: size of the system
    """
    L = size*a

    # Apply BC to x
    position[0] = position[0] % L
    if position[0]<0:
        position[0] += L

    # Apply BC to y
    position[1] = position[1] % L
    if position[1]<0:
        position[1] += L

    return position


def flip_loop(a,size,centers,dirs,rels):
    
    # choose one random colloid
    sel = np.random.randint(len(centers))

    # decide if horizontal or vertical & get displacements in loop
    if is_horizontal(dirs[sel]):
        # down, up, right, left
            shift = [
            np.array([0,0,0]),
            np.array([0,a,0]),
            np.array([a/2,a/2,0]),
            np.array([-a/2,a/2,0]) ]
    else:
        # right, left, up, down
            shift = [
            np.array([0,0,0]),
            np.array([-a,0,0]),
            np.array([-a/2,a/2,0]),
            np.array([-a/2,-a/2,0]) ]

    # get those positions & fix pbc
    pos_pbc = np.zeros((4,3))
    for j in range(len(shift)):
        cpos = fix_position(centers[sel] + shift[j],a,size)
        pos_pbc[j,:] = cpos

    # get the indices
    indices = [get_idx_from_position(centers,x) for x in pos_pbc]

    # flip them (kinda hard coded oops)
    dirs, rels = flip_spin(dirs,rels,indices[0])
    dirs, rels = flip_spin(dirs,rels,indices[1])
    dirs, rels = flip_spin(dirs,rels,indices[2])
    dirs, rels = flip_spin(dirs,rels,indices[3])

    return dirs, rels

def vertices_lattice(a,L,spos=(15,15)):
    """
        Create a matrix of shape (L,L,3) that stores the coordinates of the vertices
        ----------
        Parameters:
        * a: lattice constant
        * L: size of the system
        * spos: Offcenter tuple.
    """
    xstart,ystart = spos
    xcoords = np.linspace(xstart,L*a-xstart,L+1)[:-1]
    ycoords = np.linspace(ystart,L*a-ystart,L+1)[:-1]
    zcoords = [0]

    global_pos = [ np.array(element) for element in itertools.product(xcoords,ycoords,zcoords) ]

    return np.reshape(global_pos,(L,L,3))

def indices_lattice(vrt_lattice,centers,a,N):
    """
        Make a matrix of size (L,L,4) where the (i,j,:) element
        points to the 4 colloids associated to vertex (i,j)
        ----------
        Parameters:
        * vrt_lattice: (L,L,3) array with the vertex positions in real space
        * centers: centers of the traps
    """

    rows, cols = vrt_lattice.shape[:2]
    indices_matrix = np.zeros((rows,cols,4)) # intialize

    for i in range(rows):
        for j in range(cols):
            # current position at (i,j)           
            cur_vrt = vrt_lattice[i,j,:] 

            # get the positions with pbc
            up = fix_position(cur_vrt + np.array([0,a/2,0]),a,N)
            down = fix_position( cur_vrt + np.array([0,-a/2,0]), a, N)
            left = fix_position( cur_vrt + np.array([-a/2,0,0]), a, N)
            right = fix_position(cur_vrt + np.array([a/2,0,0]), a, N)

            # get the indices
            up_idx = get_idx_from_position(centers,up)
            down_idx = get_idx_from_position(centers,down)
            left_idx = get_idx_from_position(centers,left)
            right_idx = get_idx_from_position(centers,right)

            indices_matrix[i,j,:] = np.array([up_idx,down_idx,left_idx,right_idx])
    
    return indices_matrix

@jit(nopython=True)
def get_topological_charge_at_vertex(indices,dirs):
    """
        Computes the topological charge at a given vertex.
        ----------
        Parameters:
        * indices: the 4 indices that point to the 4 colloids in the vertes
        * dirs: directions of all vertices
    """
    
    towards = np.array([
        [0,-1,0], #up
        [0,1,0], #down
        [1,0,0], #left
        [-1,0,0] #right
    ])

    charge = 0
    for i in range(len(indices)):
        idx = int(indices[i])

        # this is a dot product between direction \cdot towards
        # this will give 1 if the spin points towards the vertex
        # this will give -1 if the spin points away
        charge += dirs[idx][0]*towards[i][0] + dirs[idx][1]*towards[i][1] + dirs[idx][2]*towards[i][2] 
        

    return charge

@jit(nopython=True)
def get_charge_lattice(indices_lattice,dirs):
    """
        Computes the topological charge in the current lattice
    """

    rows, cols = indices_lattice.shape[:2]
    charges = np.zeros((rows,cols))

    for i in range(rows):
        for j in range(cols):
            charges[i,j] = get_topological_charge_at_vertex(indices_lattice[i,j,:],dirs)
    
    return charges

@jit(nopython=True)
def is_accepted(dE,T, kB =1):
    """
        Acceptance function for simulated annealing.
        ----------
        Parameters:
        * dE: Energy difference
        * T: Temperature
        * kB (obtional): Bolzman constant, defaults to 1.
    """
    division = (dE/kB/T)
    if dE < 0:
        return True
    else:
        r = np.random.rand()
        if r < np.exp(-division):
            return True
        else:
            return False

@jit(nopython=True)
def charge_op(charged_vertices):
    """
        Computes the kappa order parameter
        ----------
        Parameters:
        * charged_vertices: Array (N,N) where (i,j) has the charge of vertex (i,j)
    """

    kappa = 0
    rows,cols = charged_vertices.shape 
    for i in range(rows):
        for j in range(cols):
            kappa += charged_vertices[i,j]*(-1)**(i+j)

    return kappa


def get_objective_function(indices_matrix,dirs,N):

    q = get_charge_lattice(indices_matrix,dirs)
    kappa = charge_op(q)
    return np.abs(2*N**2 - np.abs(kappa))


def display_vertices(trj,N,a,ax):

    """
        Plots the topological charges of a given trj.
        ----------
        Parameters:
        * trj: trajectory dataframe
        * N: vertices per side
        * a: lattice constant
        * ax: matplotlib.Axes object
    """
    
    # generate the topology
    centers, dirs, rels = trj2numpy(trj)

    # lattice with the position of the vertices
    vrt_lattice = vertices_lattice(a.magnitude,N,spos=(0,0))

    # matrix with association vertex-colloid
    indices_matrix = indices_lattice(vrt_lattice,centers, a.magnitude, N)

    # lattice with the topological charges
    q = get_charge_lattice(indices_matrix,dirs)

    rows, cols = q.shape

    for i in range(rows):
        for j in range(cols):

            if q[i,j] < 0:
                c = 'blue'
            elif q[i,j] >0:
                c = 'red'
            else:
                c='k'

            ax.add_artist( plt.Circle(
                vrt_lattice[i,j,:2], # position
                0.9*np.abs(q[i,j]), # radius
                color=c
                ))


def normalize_spin(x):
    return x/np.linalg.norm(x)

def display_arrows(trj,N,a,ax):


    # some plotting parameters
    offset = 2.5


    # generate the topology
    centers, dirs, rels = trj2numpy(trj)

    # lattice with the position of the vertices
    vrt_lattice = vertices_lattice(a.magnitude,N,spos=(0,0))

    # matrix with association vertex-colloid
    indices_matrix = indices_lattice(vrt_lattice,centers, a.magnitude, N)


    rows, cols = indices_matrix.shape[:2]

    # testing purposes
    for i in range(rows):
        for j in range(cols):

            # get the position
            x,y,z = tuple(vrt_lattice[i,j,:])
            # get the directions of the colloids related to the vertices
            cidxs = [int(k) for k in  indices_matrix[i,j,:]]
            # get the total direction of the arrow at vertex
            # this is only the vector sum of all of the directions at the vetex
            arrow_direction = normalize_spin( np.sum(dirs[cidxs], axis=0) )
            dx,dy,dz= tuple(arrow_direction)

            ax.add_artist( plt.Arrow(x-offset*dx,y-offset*dy,2*offset*dx,2*offset*dy, width=5, color='black'))


def display_lines(trj,N,a,ax):

    from matplotlib.lines import Line2D

    # some plotting parameters
    offset = 5


    # generate the topology
    centers, dirs, rels = trj2numpy(trj)

    # lattice with the position of the vertices
    vrt_lattice = vertices_lattice(a.magnitude,N,spos=(0,0))

    # matrix with association vertex-colloid
    indices_matrix = indices_lattice(vrt_lattice,centers, a.magnitude, N)


    rows, cols = indices_matrix.shape[:2]

    # testing purposes
    for i in range(rows):
        for j in range(cols):

            # get the position
            x,y,z = tuple(vrt_lattice[i,j,:])
            # get the directions of the colloids related to the vertices
            cidxs = [int(k) for k in  indices_matrix[i,j,:]]
            # get the total direction of the arrow at vertex
            # this is only the vector sum of all of the directions at the vetex
            arrow_direction = normalize_spin( np.sum(dirs[cidxs], axis=0) )
            dx,dy,dz= tuple(arrow_direction)


            ax.add_line( Line2D([x-offset*dx, x+offset*dx ],[y-offset*dy, y+offset*dy ], color='#d10014', linewidth=3)    )

            #ax.add_artist( plt.Arrow(x-offset*dx,y-offset*dy,2*offset*dx,2*offset*dy, width=5, color='red'))