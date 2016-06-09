"""
Computes H(X) - H(X|Y)
"""
function information{T<:EntropyEstimator}(Q::Type{T}, X::Array{Int64,2}, Y::Array{Int64,2},nruns::Int64=1,α::Real=1.0)
    nbins = size(X,2)
    H = zeros(nbins,nruns)
    σ² = zeros(nbins,nruns)
    Hs = zeros(nbins)
    σ²s = zeros(nbins)
    Ys = copy(X)
    for i in 1:nbins
        SE = estimate(Q,X[:,i]',α)
        Hs[i] = SE.H
        σ²s[i] = SE.σ²
        for r in 1:nruns
            H[i,r],σ²[i,r] = conditional_entropy(Q,X[:,i], Ys[:,i];α=α)
            shuffle!(ArrayViews.view(Ys, :, i))
        end
    end
    if nruns > 1
        I = Hs .- H
        M = mean(I[:,2:end],2)[:]
        S = std(I[:,2:end],2)[:]
        return I[:,1], M, S
    else
        return Hs .- H[:,1], σ²[:,1].+σ²s
    end
end

"""
Computes H(X|Y) - H(X|Y,Z)

  function information{T<:EntropyEstimator}(Q::Type{T}, X::Array{Int64,2}, Y::Array{Int64,2},Z::Array{Int64,2}, nruns::Int64=1,α::Real=1.0)

"""
function information{T<:EntropyEstimator}(Q::Type{T}, X::Array{Int64,2}, Y::Array{Int64,2},Z::Array{Int64,2}, nruns::Int64=1,α::Real=1.0)
  ntrials,nbins = size(X)
  ntrials == size(Y,1) == size(Z,1) || throw(ArgumentError("All variables must have the same number of trials"))
  Hx_y = zeros(1,nbins) #H(X|S)
  σx_s = zeros(Hx_y)
  Hx_yz = zeros(nruns+1,nbins) #H(X|S,Y)
  σx_sy = zeros(Hx_yz)
  Zs = copy(Z)
  for i in 1:nbins
    Hx_y[1,i], σx_s[1,i] = conditional_entropy(Q, X[:,i], Y[:,i];α=α)
    Hx_yz[1,i], σx_sy[1,i] = conditional_entropy(Q, X[:,i], Y[:,i], Z[:,i];α=α)
  end
  for r in 2:nruns+1
    for i in 1:nbins
      shuffle!(ArrayViews.view(Zs,:,i))
      Hx_yz[r,i], σx_sy[r,i] = conditional_entropy(Q, X[:,i], Y[:,i], Zs[:,i];α=α)
    end
  end
  I = Hx_y - Hx_yz[1,:]
  Is = Hx_y .- Hx_yz
  M = mean(Is[2:end,:],1)
  S = std(Is[2:end,:],1)
  I, M, S
end

"""
Computes H(S|Y) - H(S|Y,Z) when S is constant across bins
"""
function information{T<:EntropyEstimator}(Q::Type{T}, s::Array{Int64,1}, Y::Array{Int64,2},Z::Array{Int64,2}, nruns::Int64=1,α::Real=1.0)
  ntrials,nbins = size(Y)
  X = repmat(s,1,nbins)
  information(Q, X, Y, Z, nruns, α)
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
