function levelprune(w::AbstractArray, level)
    p = sortperm(vec(w); rev = true)
    n = ceil(Int, (1 - level) * length(p))

    return mask(w, p[1:n])
end

struct LevelPrune{T<:Real}
    level::T
end
(p::LevelPrune)(w) = levelprune(w, p.level)


thresholdprune(w::AbstractArray, threshold) = mask(w, abs.(w) .>= threshold)

struct ThresholdPrune{T<:Real}
    threshold::T
end
(p::ThresholdPrune)(w) = thresholdprune(w, p.threshold)


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

function copy_channel_mask(dst::AbstractVector, src::MaskedSliceArray{<:Any, 4})
    m = zeros(Bool, length(dst))
    m[src.slices[4]] .= true

    return mask(dst, m)
end
function copy_channel_mask(dst::AbstractVector, src::MaskedArray{<:Any, 4})
    m = zeros(Bool, length(dst))
    slices = [all(iszero, w) for w in eachslice(bitmask(src); dims = 4)]
    m[slices] .= true

    return mask(dst, m)
end

function propagate_pruning_convbn(conv::Conv, bn::BatchNorm)
    γ = copy_channel_mask(bn.γ, conv.weight)
    β = copy_channel_mask(bn.β, conv.weight)
    _bn = BatchNorm(bn.λ, β, γ, bn.μ, bn.σ², bn.ϵ,
                    bn.momentum,
                    bn.affine,
                    bn.track_stats,
                    bn.active,
                    bn.chs)

    return conv, _bn
end

function propagate_pruning_convbn(current_layer::Chain, next_layer)
    layers = collect(current_layer.layers)
    for i in 1:(length(layers) - 1)
        layers[i], layers[i + 1] = propagate_pruning_convbn(layers[i], layers[i + 1])
    end

    return Chain(layers...), next_layer
end

propagate_pruning_convbn(current_layer, next_layer) = current_layer, next_layer
propagate_pruning_convbn(model) = propagate_pruning_convbn(model, nothing)[1]

function prune(strategy::ChannelPrune, m)
    exclude(x) = x isa Conv || _default_prune_exclude(x)
    strategies = mappruneable_structure(m; exclude = exclude) do _m
        (_m isa Conv) ? walkpruneable_structure(_ -> strategy, _m) : identity
    end

    return prune(strategies, m) |> propagate_pruning_convbn
end
