module Entropies
using StatsBase
import StatsBase.entropy

include("utils.jl")
include("types.jl")
include("counting.jl")
include("nsb.jl")
include("information.jl")

g(x,α) = 2.^((1-α)*x) - 1.0
g(x) = g(x,2.0)

h(x,α) = log2((x+1))/(1-α) #g-inverse
h(x) = h(x,2.0)

function estimate{T<:EntropyEstimator}(Q::Type{T}, counts::AbstractArray{Int64,1}, α::Real,K::Integer=1000)
	ntrials = sum(counts)
	if α == 1
		return estimate(Q, counts,K)
	end
	n = length(counts)
	p = zeros(Float64,n)
	for i in 1:n
    @inbounds p[i] = counts[i]/ntrials
	end
	ee = renyientropy(p, α)
	return RenyiEntropy(ee, α, ntrials, 0.0)
end

function estimate{T<:EntropyEstimator}(Q::Type{T}, X::AbstractArray{Int64,2}, α::Real=1.0,K::Integer=1000)
    counts = collect(values(StatsBase.countmap(vhash(X))))
    estimate(Q, counts, α,max(maximum(counts), K))
end

function estimate(::Type{NSBEstimator}, counts::Array{Int64,1},K::Integer=1000)
	μ, σ²,ntrials = 0,0,0
	try
		μ, σ² = nsb_entropy(counts,K)
		ntrials = sum(counts)
	catch DomainError

	end
	ShannonEntropy(μ/log(2), σ²/(log(2)*log(2)), ntrials, 0.0)
end

function estimate(::Type{MaEstimator}, counts::Array{Int64,1},K::Integer=1000)
	ntrials = sum(counts)
	pp = counts./ntrials
	SE  = ShannonEntropy(pp, ntrials)
end

"""
Compute tne Renyi entropy of order `α` of the discrete variable `x`. `α` = 1 corresponds to Shannon entropy")
"""
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

"""
Compute the conditional entropy of x, conditioning on each row in Y")
"""
function conditional_entropy{T<:EntropyEstimator}(Q::Type{T}, x::AbstractArray{Int64,1}, Y::AbstractArray{Int64,2},s::AbstractArray{Int64,1}=Int64[];α::Float64=1.0)
	groups,_ = size(Y)
	CC = Entropies.get_conditional_counts(x,Y,s)
	conditional_entropy(Q, CC,α)
end

function conditional_entropy{T<:EntropyEstimator}(Q::Type{T}, x::Array{Int64,1}, Y::Array{Int64,1}...;α::Float64=1.0)
	CC = Entropies.get_conditional_counts(x,Y...)
	conditional_entropy(Q, CC,α)
end

function conditional_entropy{T<:EntropyEstimator}(::Type{T}, CC::ConditionalCounts,α::Float64=1.0)
	q = 0.0
	σ² = 0.0
	ntrials = 0
	if α == 1
		for i in 1:length(CC.ny)
			if CC.ny[i] > 0
				#_e = entropy(P.pxy[:,i],α)
				_e = estimate(T,CC.nxy[:,i])
				q += _e.ntrials*_e.H
				σ² += _e.ntrials*_e.σ²
				ntrials += _e.ntrials
			end
		end
		return q/ntrials,σ²/ntrials
	end
	ρ = zeros(Float64,size(CC.ny))
	n = sum(CC.ny)
	if n == 0
		return 0.0, 0.0
	end
	@inbounds for i in 1:length(CC.ny)
		_pi =(CC.ny[i]/n)^α
		q += _pi
		ρ[i] = _pi
	end
	q2 = 0.0
	@inbounds for i in 1:length(CC.ny)
    ρ[i] = ρ[i]/q
		_e = estimate(T,CC.nxy[:,i],α)
    if ρ[i] > 0
      q2 += ρ[i]*g(_e.H,α)
		end
	end
	hh = h(q2,α), 0.0
end

function conditional_entropy(CC::ConditionalCounts,α::Float64=1.0)
    conditional_entropy(MaEstimator, CC,α)
end

function conditional_entropy{T<:EntropyEstimator}(Q::Type{T}, x::Array{Int64,2}, Y::Array{Int64,3};α::Float64=1.0,nruns::Int64=100)
	nbins = size(x,2)
        SE = Array(ShannonEntropy,nbins)
        conditional_entropy!(SE,Q, x, Y;α=α, nruns=nruns)
	H = zeros(nbins)
	σ² = zeros(nbins)
        for i in 1:nbins
            H[i] = SE[i].H
            σ²[i] = SE[i].σ²
        end
        H, σ²
end

function conditional_entropy!{T<:EntropyEstimator, TE<:Entropy}(SE::Array{TE,1}, ::Type{T}, x::Array{Int64,2}, Y::Array{Int64,3};α::Float64=1.0,nruns::Int64=100)
	nbins = size(x,2)
        ntrials = size(x,1)
	for i in 1:nbins
		h,σ² = conditional_entropy(T, x[:,i], Y[:,:,i];α=α)
                SE[i] = ShannonEntropy(h, σ², ntrials, 0.0)
	end
end

end
