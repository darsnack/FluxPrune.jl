function channelprune(x::AbstractArray{<:Any, 4}, level;
                      magnitude = x -> norm(x, 1))
    sⱼs = [magnitude(x[:, :, :, j]) for j in axes(x, 4)]
    p = sortperm(sⱼs; rev = true)
    n = ceil(Int, (1 - level) * length(p))

    return mask(x, :, :, :, p[1:n])
end
channelprune(m::Conv, level; kwargs...) =
    mappruneable1(w -> channelprune(w, level; kwargs...), m)

struct ChannelPrune{T, S}
    level::T
    magnitude::S
end
ChannelPrune(level; magnitude = x -> norm(x, 1)) = ChannelPrune(level, magnitude)
(p::ChannelPrune)(m) = channelprune(m, p.level; magnitude = p.magnitude)

prune(strategy::ChannelPrune, m) =
    mappruneable(strategy, m; exclude = x -> x isa Conv)
