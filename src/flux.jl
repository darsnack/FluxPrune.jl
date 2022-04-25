function Flux.Optimise.update!(opt::Flux.Optimise.AbstractOptimiser, x::T, dx) where T<:Union{<:MaskedArray, <:MaskedSliceArray}
    Flux.Optimise.update!(opt, freeze(x), dx)
    x .-= dx

    return x
end
