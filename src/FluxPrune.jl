module FluxPrune

using MaskedArrays
using MaskedArrays: mask, freeze, MaskedArray, MaskedSliceArray
using Functors: isleaf, functor
using LinearAlgebra
using NNlib
using ChainRulesCore
using Flux

include("nnlib.jl")
include("flux.jl")

include("functor.jl")
export prune, keepprune

include("unstructured.jl")
export LevelPrune, ThresholdPrune

include("structured.jl")
export ChannelPrune

include("strategies.jl")
export iterativeprune

end
