module FluxPrune

export prune, iterativeprune,
       LevelPrune, ThresholdPrune,
       ChannelPrune

using MaskedArrays
using MaskedArrays: mask, freeze, MaskedArray, MaskedSliceArray
using Functors: isleaf, functor
using LinearAlgebra
using NNlib
using ChainRulesCore
using Flux

include("nnlib.jl")
include("functor.jl")
include("unstructured.jl")
include("structured.jl")
include("strategies.jl")

end
