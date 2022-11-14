for T in (MaskedArray, MaskedSliceArray)
    @eval begin
        function ChainRulesCore.rrule(::typeof(Base.:(*)),
                                      x::$T,
                                      y::AbstractVecOrMat{<:Union{Real, Complex}})
            y, pb = ChainRulesCore.rrule(*, freeze(x), y)
            function _pb(Δ)
                Δs = pb(Δ)
                return (Δs[1], Δs[2] .* bitmask(x), Δs[3])
            end
            return y, _pb
        end
        function ChainRulesCore.rrule(::typeof(Base.:(*)),
                                      x::AbstractVecOrMat{<:Union{Real, Complex}},
                                      y::$T)
            y, pb = ChainRulesCore.rrule(*, x, freeze(y))
            function _pb(Δ)
                Δs = pb(Δ)
                return (Δs[1], Δs[2], Δs[3] .* bitmask(y))
            end
            return y, _pb
        end
        function Zygote._pullback(ctx::Zygote.AContext,
                                  ::typeof(Base.:(*)),
                                  x::$T,
                                  y::AbstractVecOrMat{<:Union{Real, Complex}})
            return Zygote.chain_rrule(Zygote.ZygoteRuleConfig(ctx), *, x, y)
        end
        function Zygote._pullback(ctx::Zygote.AContext,
                                  ::typeof(Base.:(*)),
                                  x::AbstractVecOrMat{<:Union{Real, Complex}},
                                  y::$T)
            return Zygote.chain_rrule(Zygote.ZygoteRuleConfig(ctx), *, x, y)
        end
    end
end

for T in (MaskedArray, MaskedSliceArray)
    @eval begin
        NNlibCUDA.batchnorm(g::$T, b::$T, x,
                            running_mean, running_var, momentum; kwargs...) =
            NNlibCUDA.batchnorm(freeze(g), freeze(b), x,
                                running_mean, running_var, momentum; kwargs...)

        function NNlibCUDA.∇batchnorm(g::$T, b::$T, x, dy,
                                      running_mean, running_var, momentum; kwargs...)
            dg, db, dx = NNlibCUDA.∇batchnorm(freeze(g), freeze(b), x, dy,
                                              running_mean, running_var, momentum; kwargs...)
            return dg .* bitmask(g), db .* bitmask(b), dx
        end
    end
end

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
                     kwargs...) where {xT, wT, N} = NNlib.$fconv(x, freeze(w), cdims; kwargs...)

        function ChainRulesCore.rrule(::typeof($fconv), x, w::$WT, cdims; kwargs...)
            y, pb = ChainRulesCore.rrule($fconv, x, freeze(w), cdims; kwargs...)
            function $conv_pullback(Δ)
                Δs = pb(Δ)
                return (Δs[1], Δs[2], bitmask(w) .* Δs[3], Δs[4])
            end
            return y, $conv_pullback
        end

        function ChainRulesCore.rrule(::typeof($∇conv_data), x, w::$WT, cdims; kwargs...)
            y, pb = ChainRulesCore.rrule($∇conv_data, x, freeze(w), cdims; kwargs...)
            function $∇conv_data_pullback(Δ)
                Δs = pb(Δ)
                return (Δs[1], Δs[2], bitmask(w) .* Δs[3], Δs[4])
            end
            return y, $∇conv_data_pullback
        end
    end
end
