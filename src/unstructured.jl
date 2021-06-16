function levelprune(w::AbstractArray, level)
    p = sortperm(vec(w))
    n = ceil(Int, (1 - level) * length(p))

    return mask(w, p[1:n])
end
levelprune(level) = w -> levelprune(w, level)

thresholdprune(w::AbstractArray, threshold) = mask(w, w .>= threshold)
thresholdprune(threshold) = w -> thresholdprune(w, threshold)
