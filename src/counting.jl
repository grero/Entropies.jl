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
		xybins[i] = collect(0:ng[i]-1)
	end
	ybins = collect(0:nx-1)
	ConditionalCounts(nxy, ny, xybins, ybins)
end

function get_conditional_counts!(PP::ConditionalCounts, x::AbstractArray{Int64,1}, groups::AbstractArray{Int64,2},sgroup::AbstractArray{Int64,1}=Int64[])
	ngroups = size(groups,1)
	ng = zeros(Int64,ngroups)
	for gg in 1:ngroups
		ng[gg] = length(PP.xybins[gg])
	end
	if !isempty(sgroup)
		push!(ng, length(PP.xybins[end]))
	end
	for i in 1:length(x)
		#inline sub2in
		q = groups[1,i]+1
		s = 1
		for j in 2:ngroups
			s *= ng[j-1]
			q += (groups[j,i])*s
		end
		if !isempty(sgroup)
			s *= ng[ngroups]
			q += (sgroup[i])*s
		end
		#q = sub2ind(ng,ArrayViews.view(groups,:,i)+1)
		PP.nxy[x[i]+ 1,q] += 1
		PP.ny[q] += 1
	end
end

function get_conditional_counts!(PP::ConditionalCounts, x::Array{Int64,1}, groups::Array{Int64,1}...)
	ngroups = length(groups)
	ng = zeros(Int64,ngroups)
	for gg in 1:ngroups
		ng[gg] = length(PP.xybins[gg])
	end
    gi = zeros(Int64,ngroups)
	for i in 1:length(x)
		q = groups[1][i]+1
		s = 1
		for j in 2:ngroups
				s *= ng[j-1]
				q += (groups[j][i])*s
		end
		#q = sub2ind(ng,gi)
		PP.nxy[x[i]+ 1,q] += 1
		PP.ny[q] += 1
	end
end

function get_conditional_counts(x::AbstractArray{Int64,1}, groups::AbstractArray{Int64,2}, sgroup::AbstractArray{Int64,1}=Int64[])
	mx = maximum(x)+1
	mxy = maximum(groups,2)[:]+1
	if !isempty(sgroup)
		push!(mxy, maximum(sgroup)+1)
	end
	PP = ConditionalCounts(mx, mxy...)
	get_conditional_counts!(PP, x, groups, sgroup)
	PP
end

function get_conditional_counts(x::Array{Int64,1}, groups::Array{Int64,1}...)
	mx = maximum(x)+1
	ngroups = length(groups)
	mxy = zeros(Int64,ngroups)
	for (i,g) in enumerate(groups)
			mxy[i] = maximum(g)+1
	end
	PP = ConditionalCounts(mx, mxy...)
	get_conditional_counts!(PP, x, groups...)
	PP
end
