module FluxPrune

export prune, levelprune

using MaskedArrays
using MaskedArrays: mask
using Functors: isleaf, functor
using Flux

include("functor.jl")

function levelprune(w::AbstractArray, level)
    p = sortperm(vec(w))
    n = ceil(Int, (1 - level) * length(p))

    return mask(w, p[1:n])
end
levelprune(level) = w -> levelprune(w, level)

prune(strategy, m) = mappruneable(strategy, m; exclude = x -> isleaf(x) && x isa AbstractArray)

end
