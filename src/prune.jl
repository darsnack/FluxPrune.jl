prune(strategy, m) = mappruneable(strategy, m; exclude = x -> isleaf(x) && x isa AbstractArray)
prune(strategies::Union{<:Tuple, <:AbstractVector}, m::Chain) = Chain(prune.(strategies, m)...)

function iterativeprune(finetune!, stages, m; maxtries = 5)
    for stage in stages
        # prune to current stage
        m̄ = prune(stage, m)
        @show count(iszero, m̄[end].weight) / length(m̄[end].weight)

        # finetune pruned model as needed
        attempts = 0
        while (attempts < maxtries) && !finetune!(m̄)
            attempts += 1
        end
        m = m̄
    end

    return m
end
