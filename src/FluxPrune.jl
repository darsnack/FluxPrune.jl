module FluxPrune

export prune,
       LevelPrune, ThresholdPrune

using MaskedArrays
using MaskedArrays: mask
using Functors: isleaf, functor
using Flux

include("functor.jl")
include("unstructured.jl")

prune(strategy, m) = mappruneable(strategy, m; exclude = x -> isleaf(x) && x isa AbstractArray)

end
