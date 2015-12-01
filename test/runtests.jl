import Entropies
import StatsBase
import Distributions
using Base.Test

function generate_data(;ntrials::Int64=1000,β₀::Real=0.19,β₁::Real=0.43, λ₁::Real=5.0)
	#log(λ₁) = log(r) + (β₀ + β₁*λ₁)
	#Q = β₀/(β₁λ₁) ; ratio between background firing and influence from cell 1
	#β₁ = log(λ₁/r)/(λ₁(Q+1));

	X1 = zeros(Int64,ntrials)
	X2 = zeros(Int64,ntrials)
	#X3 = zeros(Int64,ntrials)
	P1 = Distributions.Poisson(λ₁)
	for i in 1:ntrials
		X1[i] = rand(P1)
		#λ₂ = exp(β₀ + β₁*X1[i])
		λ₂ = log(1 + exp(β₀ + β₁*X1[i]))
		#λ₃ = log(1 + exp(β₀ + β₁*X1[i]))
		X2[i] = rand(Distributions.Poisson(λ₂))
	end
	return X1, X2
end

function test_hash()
    srand(1234)
    Y = rand(0:1,(5,100))
    E_nsb = Entropies.estimate(Entropies.NSBEstimator, Y, 1.0)
    @test_approx_eq E_nsb.H 5.121949001404041
    @test_approx_eq E_nsb.σ² 0.030257361952552006
    println("Test 2D entropy with NSB-estimator passed")

    E_ma = Entropies.estimate(Entropies.MaEstimator, Y, 1.0)
    @test_approx_eq E_ma.H 4.946936297034786
    @test_approx_eq E_ma.σ² 0.005368400644084358
    println("Test 2D entropy with MA-estimator passed")
end

function test_ma_estimator()
	srand(1234)
	X = rand(0:5, 1000)
	n = StatsBase.countmap(X)
	SH = Entropies.estimate(Entropies.MaEstimator, collect(values(n)))
	@test_approx_eq SH.H 2.5850538223964947
	@test_approx_eq SH.bias 0.0036067376022224087
	@test_approx_eq SH.σ² 8.414455994916174e-6
	println("MA estimator test passed")
end

function test_renyi_estimator()
	srand(1234)
	X = rand(0:5, 1000)
	n = StatsBase.countmap(X)
	SH = Entropies.estimate(Entropies.MaEstimator, collect(values(n)), 0.5)
	@test_approx_eq SH.H 2.583197825555981
	println("Renyi estimator for α = 0.5 passed")
	SH = Entropies.estimate(Entropies.MaEstimator, collect(values(n)), 2.0)
	@test_approx_eq SH.H 2.57799095742199
	println("Renyi estimator for α = 2.0 passed")
end

function test_nsb_estimator()
	srand(1234)
	X = rand(0:5, 1000)
	n = StatsBase.countmap(X)
	SH = Entropies.estimate(Entropies.NSBEstimator, collect(values(n)))
	@test_approx_eq SH.H 2.585903905550866
	@test_approx_eq SH.σ² 8.76643342960793e-5
	println("NSB estimator test passed")
end

function test_nsb_conditional_entropy()
	srand(1234)
	X1, X2 = generate_data()
	H, σ² = Entropies.conditional_entropy(Entropies.NSBEstimator,X2, X1')
	@test_approx_eq H 2.6303445189858787
	@test_approx_eq σ² 0.03910289523194087
    println("NSB conditional entropy estimator passed")
end

function test_renyi_conditional_entropy()
	srand(1234)
	X1, X2 = generate_data()
	H, σ² = Entropies.conditional_entropy(Entropies.MaEstimator,X2, X1';α=0.5)
	@test_approx_eq H 2.55525429245648
	H, σ² = Entropies.conditional_entropy(Entropies.MaEstimator,X2, X1;α=0.5)
	@test_approx_eq H 2.55525429245648
    println("Renyi conditional entropy estimator with α = 0.5 passed")
	H, σ² = Entropies.conditional_entropy(Entropies.MaEstimator,X2, X1';α=2.0)
	@test_approx_eq H 2.4195163659791508
	println("Renyi conditional entropy estimator with α = 2.0 passed")
end

function init()
	srand(1234)
	x = rand(1:100,50)
	n = collect(values(StatsBase.countmap(x)))
	S_nsb, _ = Entropies.find_nsb_entropy(n,100,1e-5)
	return S_nsb
end

function test_B_xiK()
	S_nsb = init()
	B =  Entropies..B_xiK(1.2, S_nsb)
	return @test_approx_eq B 1.3853866920075761
end

function test_find_nsb_entropy()
	S_nsb = init()
	Entropies.find_nsb_entropy(S_nsb,1e-5)
	return @test_approx_eq S_nsb.S_nsb 4.484256684719789
end

function test_mlog_evidence()
	S_nsb = init()
	nsb_mlog = Entropies.mlog_evidence(200*S_nsb.K,S_nsb)
	return @test_approx_eq nsb_mlog 46.05806318661056
end

test_ma_estimator()
test_renyi_estimator()
test_nsb_estimator()
test_nsb_conditional_entropy()
test_renyi_conditional_entropy()
test_hash()

S_nsb = init()
ms2 = Entropies.meanS2(370.0850,S_nsb)
@test_approx_eq ms2 20.049019410263792
B =  Entropies.B_xiK(1.2, S_nsb)
@test_approx_eq B 1.3853866920075761

nsb_mlog = Entropies.mlog_evidence(200*S_nsb.K,S_nsb)
@test_approx_eq nsb_mlog 59.85364833069893

Entropies.find_nsb_entropy(S_nsb,1e-5)
@test_approx_eq S_nsb.S_nsb 4.3952671710926685


