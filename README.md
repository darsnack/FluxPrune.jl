# FluxPrune

[![Build Status](https://github.com/darsnack/FluxPrune.jl/workflows/CI/badge.svg)](https://github.com/darsnack/FluxPrune.jl/actions)

FluxPrune.jl provides iterative pruning algorithms for Flux models. Pruning strategies can be _unstructured_ or _structured_. Unstructured strategies operate on arrays, while structured strategies operate on layers.

## Examples

Unstructured edge pruning:
```julia
using Flux, FluxPrune

m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), Dense(512, 10))
# prune all weights to 70% sparsity
m̄ = prune(LevelPrune(0.7), m)
# prune all weights with magnitude lower than 0.5
m̄ = prune(ThresholdPrune(0.5), m)
# prune each layer in a Chain at a different rate
# (just uses broadcasting then re-Chains)
m̄ = prune([LevelPrune(0.4), LevelPrune(0.6), LevelPrune(0.7)], m)
```

Structured channel pruning:
```julia
using Flux, FluxPrune

m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), Dense(512, 10))
# prune all conv layer channels to 30% sparsity
m̄ = prune(ChannelPrune(0.3), m)
```

Mixed pruning:
```julia
using Flux, FluxPrune

m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), Dense(512, 10))
# apply channel and edge pruning
m̄ = prune([ChannelPrune(0.3), ChannelPrune(0.4), LevelPrune(0.8)], m)
```

Iterative pruning:
```julia

using Flux, FluxPrune

m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), Dense(512, 10))
# target pruning levels step-by-step
# the first argument to iterativeprune will finetune the model
#  and return true to indicate moving onto the next stage,
#  or false to indicate that finetune must be called again
stages = [[ChannelPrune(0.1), ChannelPrune(0.1), LevelPrune(0.4)],
          [ChannelPrune(0.2), ChannelPrune(0.3), LevelPrune(0.7)],
          [ChannelPrune(0.3), ChannelPrune(0.5), LevelPrune(0.9)]]
m̄ = iterativeprune(stages, m) do m̄
    opt = Momentum()
    ps = Flux.params(m̄)
    Flux.@epochs 10 Flux.train!(loss, ps, data, opt)

    return accuracy(data, m̄) > target_accuracy
end
```
