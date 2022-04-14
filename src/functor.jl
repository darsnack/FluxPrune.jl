_default_prune_exclude(x) = isleaf(x) && (x isa AbstractArray)

pruneable(m) = Flux.trainable(m)
pruneable(m::Dense) = (weight = m.weight,)
pruneable(m::Conv) = (weight = m.weight,)

function walkpruneable(f, x, s)
    children, re = functor(x)
    tchildren = pruneable(x)
    schildren, _ = functor(s)

    return re(map((x, s) -> x ∈ tchildren ? f(x, s) : x, children, schildren))
end
function walkpruneable(f, x)
    children, re = functor(x)
    tchildren = pruneable(x)

    return re(map(x -> x ∈ tchildren ? f(x) : x, children))
end
function walkpruneable_structure(f, x)
    children, _ = functor(x)
    tchildren = pruneable(x)

    return map(x -> x ∈ tchildren ? f(x) : x, children)
end

mappruneable(f, x; exclude = _default_prune_exclude, kwargs...) =
    fmap((_x, _f) -> _f(_x), x, f; walk = walkpruneable, exclude = exclude, kwargs...)
mappruneable_structure(f, x; exclude = _default_prune_exclude, kwargs...) =
    fmap(f, x; walk = walkpruneable_structure, exclude = exclude, kwargs...)

prune(strategies::Union{<:Tuple, <:NamedTuple}, m) = mappruneable(strategies, m)
prune(strategy, m) = prune(mappruneable_structure(_ -> strategy, m), m)
prune(strategies::AbstractVector, m::Chain) = Chain(prune.(strategies, m)...)

keepprune(m) = fmap(freeze, m; exclude = _default_prune_exclude, walk = walkpruneable)
