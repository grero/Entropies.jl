Docile.@doc "computes the mutual information between X and Y"->
function information{T<:EntropyEstimator}(Q::Type{T}, X::Array{Int64,2}, Y::Array{Int64,3},nruns::Int64=1)
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
            H[i,r],σ²[i,r] = conditional_entropy(Q,Xs[:,i], Y[:,:,i])
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

function information{T<:EntropyEstimator}(Q::Type{T}, s::Array{Int64,1}, N::Array{Int64,3},nruns::Int64=1)
    ntrials,nbins,ncells = size(N)
    x = repmat(s, 1, nbins)
    I = zeros(nbins, ncells)
    M = zeros(nbins, ncells)
    S = zeros(nbins, ncells)
    for i in 1:ncells
        Y = permutedims(cat(3, N[:, :, i]), (3,1,2))
        I[:,i], M[:,i], S[:,i] = Entropies.information(Entropies.MaEstimator, x, Y, nruns)
    end
    I,M,S
end
