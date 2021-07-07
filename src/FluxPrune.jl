module FluxPrune

export prune, iterativeprune,
       LevelPrune, ThresholdPrune,
       ChannelPrune

using MaskedArrays
using MaskedArrays: mask
using Functors: isleaf, functor
using LinearAlgebra
using Flux

include("functor.jl")
include("unstructured.jl")
include("structured.jl")
include("prune.jl")

end
