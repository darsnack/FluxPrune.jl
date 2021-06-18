pruneable(m) = Flux.trainable(m)
pruneable(m::Dense) = (weight = m.weight,)
pruneable(m::Conv) = (weight = m.weight,)

function mappruneable1(f, x)
    children, re = functor(x)
    tchildren = pruneable(x)
    re(map(x -> x ∈ tchildren ? f(x) : x, children))
end

  # See https://github.com/FluxML/Functors.jl/issues/2 for a discussion regarding the need for
  # cache.
function mappruneable(f, x; exclude = isleaf, cache = IdDict())
    haskey(cache, x) && return cache[x]
    y = exclude(x) ? f(x) :
                     mappruneable1(x -> mappruneable(f, x, cache = cache, exclude = exclude), x)
    cache[x] = y

    return y
end
