function vhash{T<:Integer}(X::AbstractArray{T,2})
	ndims = size(X,1)
	base = maximum(X)+1
	x = base.^[0:ndims-1]
	return X'*x
end
