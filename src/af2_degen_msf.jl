using FFTW
using LinearAlgebra
using  DataFrames, CSV

include("./auxnumerics/montecarlo.jl")


# initialize parameters
N = 10;
a = 8.374011537017761;
file_path =  "./data/q2_degeneracy/s$(N).csv";
df = CSV.read(file_path,DataFrame);
trj = df[ df[:,:realization] .== 1, : ];

# create the topology
@time begin 
    centers, dirs, rels = trj2array(trj);
    vrt_space = vrt_lattice(a,N);
    indices_matrix = indices_lattice(vrt_space,centers,a,N);
    arrow_lattice = dipole_lattice(centers,dirs,rels,vrt_space,indices_matrix);
end;
