function iterativeprune(finetune!, stages, m; maxtries = 5)
    for stage in stages
        # prune to current stage
        m̄ = prune(stage, m)

        # finetune pruned model as needed
        attempts = 0
        while (attempts < maxtries) && !finetune!(m̄)
            attempts += 1
        end
        m = m̄
    end

    return m
end
