function Flux.Optimise.update!(opt::Flux.Optimise.AbstractOptimiser, x::T, dx) where T<:Union{<:MaskedArray, <:MaskedSliceArray}
    Flux.Optimise.update!(opt, freeze(x), dx)
    x.data .-= dx

    return x
end
