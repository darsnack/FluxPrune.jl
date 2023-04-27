# FluxPrune

[![Build Status](https://github.com/darsnack/FluxPrune.jl/workflows/CI/badge.svg)](https://github.com/darsnack/FluxPrune.jl/actions)

FluxPrune.jl provides iterative pruning algorithms for Flux models. Pruning strategies can be _unstructured_ or _structured_. Unstructured strategies operate on arrays, while structured strategies operate on layers.

## Examples

### Unstructured edge pruning
```julia
using Flux, FluxPrune
using MLUtils: flatten

m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), flatten, Dense(512, 10))
# prune all weights to 70% sparsity
m̄ = prune(LevelPrune(0.7), m)
# prune all weights with magnitude lower than 0.5
m̄ = prune(ThresholdPrune(0.5), m)
# prune each layer in a Chain at a different rate
# (just uses broadcasting then re-Chains)
m̄ = prune([LevelPrune(0.4), LevelPrune(0.6), identity, LevelPrune(0.7)], m)
```

### Structured channel pruning
```julia
using Flux, FluxPrune
using MLUtils: flatten

m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), flatten, Dense(512, 10))
# prune all conv layer channels to 30% sparsity
m̄ = prune(ChannelPrune(0.3), m)
```

### Mixed pruning
```julia
using Flux, FluxPrune
using MLUtils: flatten

m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), flatten, Dense(512, 10))
# apply channel and edge pruning
m̄ = prune([ChannelPrune(0.3), ChannelPrune(0.4), identity, LevelPrune(0.8)], m)
```

### Iterative pruning

Target pruning levels step-by-step.
The first argument to iterativeprune (or the function block after the `do` statement) will finetune the model and return true to indicate moving onto the next stage, or false to indicate that finetune must be called again.
```julia
using Flux, FluxPrune
using MLUtils: flatten
using Statistics: mean

features = rand(Float32, 8, 8, 3, 100);
labels = Flux.onehotbatch(rand(0:9, 100), 0:9);
data = (features, labels);
loss(m, x, y) = Flux.Losses.mse(m(x), y)
accuracy(m, data) = mean(Flux.onecold(m(data[1]), 0:9) .== Flux.onecold(data[2], 0:9))
target_accuracy = 0.08 # random data, so this is a low target

m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), flatten, Dense(512, 10), softmax)
opt_state = Flux.setup(Momentum(), m);

stages = [
    [ChannelPrune(0.1), ChannelPrune(0.1), identity, LevelPrune(0.4), identity],
    [ChannelPrune(0.2), ChannelPrune(0.3), identity, LevelPrune(0.7), identity],
    [ChannelPrune(0.3), ChannelPrune(0.5), identity, LevelPrune(0.9), identity]
]
m̄ = iterativeprune(stages, m) do m̄
    for epoch in 1:10
        Flux.train!(loss, m̄, [data], opt_state)
    end
    return accuracy(m̄, data) > target_accuracy
end
```
