function iterativeprune(finetune!, stages, m; maxtries = 5)
    for stage in stages
        # prune to current stage
        m̄ = prune(stage, m)

        # finetune pruned model as needed
        attempts = 0
        while (attempts < maxtries) && !finetune!(m̄)
            attempts += 1
        end

        # exit early if we failed
        if attempts >= maxtries
            @warn """
            Failed to finetune model to target, exiting pruning early.
            Returning finetuned model from last successful stage.
            """
            break
        end

        m = m̄
    end

    return m
end
