_default_prune_exclude(x) = isleaf(x) && (x isa AbstractArray)

pruneable(m) = Flux.trainable(m)
pruneable(m::Dense) = (weight = m.weight,)
pruneable(m::Conv) = (weight = m.weight,)
pruneable(::BatchNorm) = (;)

map_filtered(f, subset, x, ys...) =
    map((_x, _ys...) -> _x ∈ subset ? f(_x, _ys...) : _x, x, ys...)

function walkpruneable(recurse, x, prune_fns)
    children, re = functor(x)
    pchildren = pruneable(x)
    prune_subfns, _ = functor(prune_fns)

    return re(map_filtered(recurse, pchildren, children, prune_subfns))
end
function walkpruneable(recurse, x)
    children, re = functor(x)
    pchildren = pruneable(x)

    return re(map_filtered(recurse, pchildren, children))
end
function walkpruneable_structure(recurse, x)
    children, _ = functor(x)
    pchildren = pruneable(x)

    return map_filtered(recurse, pchildren, children)
end

mappruneable(f, x; exclude = _default_prune_exclude, kwargs...) =
    fmap((_x, _f) -> _f(_x), x, f; walk = walkpruneable, exclude = exclude, kwargs...)
mappruneable_structure(f, x; exclude = _default_prune_exclude, kwargs...) =
    fmap(f, x; walk = walkpruneable_structure, exclude = exclude, kwargs...)

prune(strategies::Union{<:Tuple, <:NamedTuple}, m) = mappruneable(strategies, m)
prune(strategy, m) = prune(mappruneable_structure(_ -> strategy, m), m)
prune(strategies::AbstractVector, m::Chain) = Chain(prune.(strategies, m)...)

keepprune(m) = fmap(freeze, m; exclude = _default_prune_exclude, walk = walkpruneable)
