module Entropies
import StatsBase
using Docile
@docstrings

include("types.jl")
include("nsb.jl")

function estimate{T<:EntropyEstimator}(Q::Type{T}, counts::Array{Int64,1}, α::Real;K::Integer=1000)
	if α == 1
		return estimate(Q, counts;K=K)
	end
	ntrials = sum(counts)
	p = counts/ntrials
	ee = entropy(p, α)
	return RenyiEntropy(ee, α, ntrials, 0.0)
end

function estimate(::Type{NSBEstimator}, counts::Array{Int64,1};K::Integer=1000)
	μ, σ² = nsb_entropy(counts,K)
	ShannonEntropy(μ, σ², sum(counts), 0.0)
end

function estimate(::Type{MaEstimator}, counts::Array{Int64,1})
	ntrials = sum(counts)
	pp = counts./ntrials
	SE  = ShannonEntropy(pp, ntrials)
end

@doc meta("Compute tne Renyi entropy of order `α` of the discrete variable `x`. `α` = 1 corresponds to Shannon entropy")->
function entropy(x::Array{Int64,1},α::Real)
  n = StatsBase.counts(x,minimum(x):maximum(x))
  p = n./sum(n)
  if α == 1
	  return ShannonEntropy(p, length(x))
  else
	  ee = entropy(p,α)
	  return RenyiEntropy(ee,α, length(x),0.0)
  end
end

function entropy(p::Array{Float64,1},α::Real)
  if α == 1
	  return entropy(p)
  end
  E = 0.0
  for i in 1:length(p)
    @inbounds E += p[i]^α
  end
  E = (1/(1-α))*log2(E)
end

function entropy(p::Array{Float64,1})
	ee = 0.0
	nnz = 0
	for _p in p
		if _p > 0
			ee += _p*log(_p)
			nnz += 1
		end
	end
	-ee
end


end
