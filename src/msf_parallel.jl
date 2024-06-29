using DataFrames, CSV
using LinearAlgebra, ProgressMeter
include("auxnumerics/msf.jl")
using Base.Threads
println("using $(nthreads()) threads...")

println("loading...")
a = 8.374011537017761 # um
N = 50

file_path = "../data/q2_degeneracy/s$(N).csv"
df = CSV.read(file_path,DataFrame)
println("choosing...")
trj = df[ df[:,:realization] .== 1, :]

centers, dirs, rels = trj2array(trj);
kx,ky,mesh = reciprocal_space(a;amount_bz=3);
pre_indices = [(i,j) for i in 1:length(kx) for j in 1:length(ky)];

msf = zeros(size(mesh)...)

@showprogress Threads.@threads for idx in pre_indices
    
    q = mesh[idx...]
    msf[idx...] = single_msf(centers,dirs,rels,N,a,q)
end
