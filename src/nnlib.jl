for fconv in (:conv, :depthwiseconv), WT in (MaskedArray, MaskedSliceArray)
    local ∇conv_data, ∇conv_filter = Symbol.(:∇, fconv, [:_data, :_filter])
    local conv_pullback, ∇conv_data_pullback = Symbol.([fconv, ∇conv_data], :_pullback)

    @eval begin
        NNlib.$fconv(x::AbstractArray{xT, N}, w::$WT{wT, N}; kwargs...) where {xT, wT, N} =
            NNlib.$fconv(x, freeze(w); kwargs...)
        NNlib.$fconv(x::AbstractArray{xT, N}, w::$WT{wT, N}, cdims::ConvDims;
                     kwargs...) where {xT, wT, N} = NNlib.$fconv(x, freeze(w), cdims; kwargs...)

        NNlib.$fconv(x::AbstractArray{Flux.NilNumber.Nil, N}, w::$WT{wT, N};
                     kwargs...) where {xT, wT, N} = NNlib.$fconv(x, freeze(w); kwargs...)
        NNlib.$fconv(x::AbstractArray{Flux.NilNumber.Nil, N}, w::$WT{wT, N}, cdims::DenseConvDims;
                     kwargs...) where {xT, wT, N} = NNlib.$fconv(x, freeze(w); kwargs...)

        function ChainRulesCore.rrule(::typeof($fconv), x, w::$WT, cdims; kwargs...)
            y, pb = ChainRulesCore.rrule($fconv, x, freeze(w), cdims; kwargs...)
            function $conv_pullback(Δ)
                Δs = pb(Δ)
                return (Δs[1], Δs[2], w.mask .* Δs[3], Δs[4])
            end
            return y, $conv_pullback
        end

        function ChainRulesCore.rrule(::typeof($∇conv_data), x, w::$WT, cdims; kwargs...)
            y, pb = ChainRulesCore.rrule($∇conv_data, x, freeze(w), cdims; kwargs...)
            function $∇conv_data_pullback(Δ)
                Δs = pb(Δ)
                return (Δs[1], Δs[2], w.mask .* Δs[3], Δs[4])
            end
            return y, $∇conv_data_pullback
        end
    end
end
