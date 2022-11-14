module FluxPrune

using MaskedArrays
using MaskedArrays: mask, bitmask, freeze, MaskedArray, MaskedSliceArray
using Functors: isleaf, functor
using LinearAlgebra
using NNlib, NNlibCUDA
using ChainRulesCore
using Flux
using Zygote

include("fastpaths.jl")
include("flux.jl")

include("functor.jl")
export prune, keepprune

include("strategies.jl")
export LevelPrune, ThresholdPrune, ChannelPrune

include("recipes.jl")
export iterativeprune

end
