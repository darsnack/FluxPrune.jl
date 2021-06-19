module FluxPrune

export prune,
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

prune(strategy, m) = mappruneable(strategy, m; exclude = x -> isleaf(x) && x isa AbstractArray)

end
