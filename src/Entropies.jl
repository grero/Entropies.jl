module Entropies
import StatsBase
using Docile
@docstrings

include("types.jl")
include("nsb.jl")

function estimate(::Type{NSBEstimator}, counts::Array{Int64,1};K::Integer=1000)
	μ, σ² = nsb_entropy(counts,K)
	ShannonEntropy(μ, σ², sum(counts), 0.0)
end

function estimate(::Type{MaEstimator}, counts::Array{Int64,1})
	ntrials = sum(counts)
	pp = counts./ntrials
	SE  = ShannonEntropy(pp, ntrials)
end
end
