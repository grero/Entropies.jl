import Distributions
import Winston
using Entropies

using Docile
@docstrings

Docile.@doc meta("Test conditional entropy")->
function generate_data(;ntrials=1000,β = 0.003)
	λ₁ = 5.0
	β₀ = 0.01
	X1 = zeros(Int64,ntrials)
	X2 = zeros(Int64,ntrials)
	P1 = Distributions.Poisson(λ₁)
	for i in 1:ntrials
		X1[i] = rand(P1)
		λ₂ = exp(β₀ + β*X1[i])
		X2[i] = rand(Distributions.Poisson(λ₂))
	end
	return X1, X2
end

function example1(;β=0.003)
	ntrials = 2.^[5:17]
	n = length(ntrials)
	H_nsb = zeros(n)
	σ²_nsb = zeros(n)
	H_ma = zeros(n)
	σ²_ma = zeros(n)
	for i in 1:n
		X1, X2 = generate_data(;ntrials=ntrials[i],β=β)
		H_ma[i], σ²_ma[i] = Entropies.conditional_entropy(Entropies.MaEstimator, X1, X2')
		H_nsb[i], σ²_nsb[i] = Entropies.conditional_entropy(Entropies.NSBEstimator, X1, X2')
	end
	return ntrials, H_ma, σ²_ma, H_nsb, σ²_nsb
end

function plot_example1(ntrials, H_ma, σ²_ma, H_nsb, σ²_nsb
)
	#plotting
	p = Winston.FramedPlot()
	Winston.add(p,Winston.Points(ntrials,H_ma;color="red",symbolkind="circle"))
	Winston.add(p, Winston.ErrorBarsY(ntrials, H_ma - √σ²_ma,H_ma + √σ²_ma;color="red"))
	Winston.add(p,Winston.Points(ntrials,H_nsb ;color="blue"))
	Winston.add(p, Winston.ErrorBarsY(ntrials, H_nsb - √σ²_nsb,H_nsb + √σ²_nsb;color="blue",symbolkind="circle"))
	Winston.setattr(p, "xlog", true)
	Winston.add(p, Winston.LineY(H_ma[end]))
	p
end

function plot_entropy_estimate(ntrials, H, σ²)
	p = Winston.FramedPlot()
	y = (H-H[end])/H[end]
	dy = sqrt(σ²)/H[end]
	Winston.add(p,Winston.Points(ntrials,y;color="blue",symbolkind="circle"))
	Winston.add(p, Winston.ErrorBarsY(ntrials, y - dy,y + dy;color="blue"))
	Winston.setattr(p, "xlog", true)
	p
end

function explore_β()
	ta = Winston.Table(5,1)
	β = [0.001, 0.005, 0.01, 0.05, 0.1]
	for (i,βᵢ) in enumerate(β)
		ntrials, H_ma, σ²_ma, H_nsb, σ²_nsb = example1(;β=βᵢ)
		ta[i,1] = plot_example1(ntrials, H_ma, σ²_ma, H_nsb, σ²_nsb)
	end
	ta
end

Docile.@doc meta("Generate q's from Eq. (5) in Nemenman et al. (2008)")->
function generate_q(β, K)
       q = zeros(K)
       q[1] = rand(Distributions.Beta(β, (K-1)*β))
       for i in 2:K-1
          q[i] = (1-sum(q[1:i-1]))*rand(Distributions.Beta(β, (K-i)*β))
       end
       q[end] = 1-sum(q)
       q
end

function generate_n(q::Array{Float64,1},N::Integer)
	K = length(q)
	n = zeros(Int64,K)
	Q = cumsum(q)
	for i in 1:N
		idx = searchsortedfirst(Q, rand())
		n[idx] +=1
	end
	n
end

function test_nsb(β, K)
	q = generate_q(β, K)
	ntrials = 2.^[5:17]
	nn = length(ntrials)
	H = zeros(nn)
	σ² = zeros(nn)
	for i in 1:nn
		#n = generate_n(q, ntrials[i])
		n = rand(Distributions.Multinomial(ntrials[i], q))
		nsb = Entropies.find_nsb_entropy(n, K, 1e-5)
		H[i] = nsb[1].S_nsb/log(2)
		σ²[i] = nsb[1].dS_nsb/(log(2)*log(2))
	end
	H, σ²
end



