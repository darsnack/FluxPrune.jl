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
