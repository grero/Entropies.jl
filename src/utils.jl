using Distributions

function vhash{T<:Integer}(X::AbstractArray{T,2})
	ndims = size(X,1)
	base = maximum(X)+1
	x = base.^(0:ndims-1)
	return X'*x
end

type SimpleConditional <: Sampleable{Multivariate,Discrete}
    P::Array{Float64,2}
    n::Int64
end

function SimpleConditional(PP::Array{Float64,2})
    SimpleConditional(PP,size(PP,1))
end


type JointPoisson <: Sampleable{Multivariate,Discrete}
    λ::Array{Float64,1}
    n::Int64
end

JointPoisson(λ::Array{Float64,1}) = JointPoisson(λ,length(λ))

Base.length(s::Union{SimpleConditional,JointPoisson}) = s.n

function Distributions._rand!{T<:Real}(s::SimpleConditional, x::AbstractVector{T})
    m = size(s.P,1)
    x[1] = rand(0:m-1)
    rr = rand()
    idx = searchsortedfirst(s.P[:,x[1]+1], rr)
    x[2] = m-idx
    x
end

function Distributions._rand!{T<:Real}(s::JointPoisson,x::AbstractVector{T})
    for i in 1:s.n
        x[i] = rand(Poisson(s.λ[i]))
    end
    x
end
