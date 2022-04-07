pruneable(m) = Flux.trainable(m)
pruneable(m::Dense) = (weight = m.weight,)
pruneable(m::Conv) = (weight = m.weight,)

function walkpruneable(f, x)
    children, re = functor(x)
    tchildren = pruneable(x)
    re(map(x -> x ∈ tchildren ? f(x) : x, children))
end

mappruneable(f, x; kwargs...) = fmap(f, x; walk = walkpruneable, kwargs...)
