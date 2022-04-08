function channelprune(x::AbstractArray{<:Any, 4}, level;
                      magnitude = x -> norm(x, 1))
    sⱼs = [magnitude(x[:, :, :, j]) for j in axes(x, 4)]
    p = sortperm(sⱼs; rev = true)
    n = ceil(Int, (1 - level) * length(p))

    return mask(x, :, :, :, p[1:n])
end

struct ChannelPrune{T, S}
    level::T
    magnitude::S
end
ChannelPrune(level; magnitude = x -> norm(x, 1)) = ChannelPrune(level, magnitude)
(p::ChannelPrune)(w) = channelprune(w, p.level; magnitude = p.magnitude)

function prune(strategy::ChannelPrune, m)
    exclude(x) = x isa Conv || _default_prune_exclude(x)
    strategies = mappruneable_structure(m; exclude = exclude) do _m
        (_m isa Conv) ? walkpruneable_structure(_ -> strategy, _m) : identity
    end

    return prune(strategies, m)
end
