
abstract Entropy
abstract EntropyEstimator
abstract ShannonEntropyEstimator <: EntropyEstimator

type PluginEstimator <: ShannonEntropyEstimator end
type MaEstimator <: ShannonEntropyEstimator end
type NSBEstimator <: ShannonEntropyEstimator end

type ShannonEntropy <: Entropy
	H::Float64
	σ²::Float64
	ntrials::Integer
	bias::Float64
end

function ShannonEntropy(p::Array{Float64,1}, ntrials::Integer)
	np = length(p)
	_e = 0.0
	nnz = 0
	#compute entropy
	for i in 1:np
		@inbounds _pi = p[i]
		if _pi > 0.0
			_e -= _pi*log(_pi)
			nnz +=1
		end
	end
	#compute variance
	_vv = 0.0
	for i in 1:np
		@inbounds _pi = p[i]
		if _pi > 0.0
			q = log(_pi) + _e
			_vv += q*q*_pi*(1.0 - _pi)
		end
	end
	_vv = _vv/ntrials
	bias = (nnz - 1)/(2*ntrials*log(2))
	ShannonEntropy(_e+bias, _vv, ntrials,bias)
end

type RenyiEntropy <: Entropy
	H::Float64
	α::Real
	ntrials::Integer
	bias::Float64
end
