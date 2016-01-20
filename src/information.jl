Docile.@doc "computes the mutual information between X and Y"->
function information{T<:EntropyEstimator}(Q::Type{T}, X::Array{Int64,2}, Y::Array{Int64,3},nruns::Int64=1,α::Real=1.0)
    nbins = size(X,2)
    H = zeros(nbins,nruns)
    σ² = zeros(nbins,nruns)
    Hs = zeros(nbins)
    σ²s = zeros(nbins)
    Xs = copy(X)
    for i in 1:nbins
        SE = estimate(Q,X[:,i]')
        Hs[i] = SE.H
        σ²s[i] = SE.σ²
        for r in 1:nruns
            H[i,r],σ²[i,r] = conditional_entropy(Q,Xs[:,i], Y[:,:,i];α=α)
            shuffle!(ArrayViews.view(Xs, :, i))
        end
    end
    if nruns > 1
        I = Hs .- H
        M = mean(I[:,2:end],2)[:]
        S = std(I[:,2:end],2)[:]
        I[:,1], M, S
    else
        Hs .- H[:,1], σ²[:,1].+σ²s
    end
end

"""
Computes H(X|S) - H(X|S,Y),
"""
function information{T<:EntropyEstimator}(Q::Type{T}, X::Array{Int64,2}, Y::Array{Int64,2},s::Array{Int64,1}, nruns::Int64=1,α::Real=1.0)
  ntrials,nbins = size(X)
  ntrials == length(s) == size(Y,1) || throw(ArgumentError("All variables must have the same number of trials"))
  Hx_s = zeros(1,nbins) #H(X|S)
  σx_s = zeros(Hx_s)
  Hx_sy = zeros(nruns+1,nbins) #H(X|S,Y)
  σx_sy = zeros(Hx_sy)
  Ys = copy(Y)
  for i in 1:nbins
    Hx_s[1,i], σx_s[1,i] = conditional_entropy(Q, X[:,i], s;α=α)
    Hx_sy[1,i], σx_s[1,i] = conditional_entropy(Q, X[:,i], Y[:,i],s;α=α)
  end
  for r in 2:nruns+1
    for i in 1:nbins
      shuffle!(ArrayViews.view(Ys,:,i))
      Hx_sy[r,i], σx_sy[r,i] = conditional_entropy(Q, X[:,i], Ys[:,i],s;α=α)
    end
  end
  I = Hx_s - Hx_sy[1,:]
  Is = Hx_s .- Hx_sy
  M = mean(Is[2:end,:],1)
  S = std(Is[2:end,:],1)
  I, M, S, Is
end

function information{T<:EntropyEstimator}(Q::Type{T}, s::Array{Int64,1}, N::Array{Int64,3},nruns::Int64=1,α::Real=1.0)
    ntrials,nbins,ncells = size(N)
    x = repmat(s, 1, nbins)
    I = zeros(nbins, ncells)
    M = zeros(nbins, ncells)
    S = zeros(nbins, ncells)
    for i in 1:ncells
        Y = permutedims(cat(3, N[:, :, i]), (3,1,2))
        I[:,i], M[:,i], S[:,i] = Entropies.information(Entropies.MaEstimator, x, Y, nruns,α)
    end
    I,M,S
end
