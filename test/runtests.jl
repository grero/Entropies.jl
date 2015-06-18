import Entropies
import StatsBase
using Base.Test

#Testing nsb_entropy

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

S_nsb = init()
ms2 = Entropies.meanS2(370.0850,S_nsb)
@test_approx_eq ms2 20.049019410263792
B =  Entropies.B_xiK(1.2, S_nsb)
@test_approx_eq B 1.3853866920075761

nsb_mlog = Entropies.mlog_evidence(200*S_nsb.K,S_nsb)
@test_approx_eq nsb_mlog 59.85364833069893

Entropies.find_nsb_entropy(S_nsb,1e-5)
@test_approx_eq S_nsb.S_nsb 4.3952671710926685

