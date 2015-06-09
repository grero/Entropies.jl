function normalize!(PP::ConditionalProbability)
	for i in 1:length(PP.nxy)
		PP.py[i] /= PP.ny
		if PP.nxy[i] > 0
			for j in 1:size(PP.pxy,1)
				PP.pxy[j,i] /= PP.nxy[i]
			end
		end
	end
end

function normalize(CC::ConditionalCounts)
	for i in 1:length(PP.nxy)
		PP.py[i] /= PP.ny
		if PP.nxy[i] > 0
			for j in 1:size(PP.pxy,1)
				PP.pxy[j,i] /= PP.nxy[i]
			end
		end
	end
end

function ConditionalCounts(nx::Int64, ng::Int64...)
	nn = prod(ng)
	nxy = zeros(Int64, nx,nn)
	ny = zeros(Int64,nn)
	xybins = Array(Array{Int64,1}, length(ng))
	for i in 1:length(ng)
		xybins[i] = [0:ng[i]-1]
	end
	ybins = [0:nx-1]
	ConditionalCounts(nxy, ny, xybins, ybins)
end

function get_conditional_counts!(PP::ConditionalCounts, x::Array{Int64,1}, groups::Array{Int64,2})
	ngroups = size(groups,1)
	ng = zeros(Int64,ngroups)
	for gg in 1:ngroups
		ng[gg] = length(PP.xybins[gg])
	end
	for i in 1:length(x)
		q = sub2ind(ng,(groups[:,i]+1)...)
		PP.nxy[x[i]+ 1,q] += 1
		PP.ny[q] += 1
	end
end

function get_conditional_counts(x::Array{Int64,1}, groups::Array{Int64,2})
	mx = maximum(x)+1
	mxy = maximum(groups,2)[:]+1
	PP = ConditionalCounts(mx, mxy...)
	get_conditional_counts!(PP, x, groups)
	PP
end
