using FluxPrune
using Flux
using MaskedArrays: bitmask
using Test

sparsity(x) = count(iszero, x) / length(x)
count_slices(f, x; dims) = count([f(slice) for slice in eachslice(x; dims = dims)])

@testset "LevelPrune" begin
    m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), Dense(512, 10))
    m̄ = prune(LevelPrune(0.5), m)
    for layer in m̄
        @test sparsity(layer.weight) ≈ 0.5 rtol=0.1
        @test layer.bias isa Array
    end
end

@testset "ThresholdPrune" begin
    m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), Dense(512, 10))
    m̄ = prune(ThresholdPrune(0.3), m)
    for layer in m̄
        @test sparsity(layer.weight) ≈ sparsity(count(>(0.3), layer.weight)) rtol=0.1
        @test layer.bias isa Array
    end
end

@testset "ChannelPrune" begin
    m = Chain(Conv((3, 3), 3 => 16), BatchNorm(16), Conv((3, 3), 16 => 32), Dense(512, 10))
    m̄ = prune(ChannelPrune(0.25), m)
    for layer in m̄
        if layer isa Conv
            @test count_slices(iszero, layer.weight; dims = 4) ≈ 0.25 * size(layer.weight, 4)
            @test layer.bias isa Array
        elseif layer isa BatchNorm
            @test sparsity(bitmask(layer.γ)) ≈ 0.25
            @test sparsity(bitmask(layer.β)) ≈ 0.25
        else
            @test layer.weight isa Array
            @test layer.bias isa Array
        end
    end
end

@testset "Selective pruning" begin
    m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), Dense(512, 10))
    m̄ = prune([LevelPrune(0.3), ChannelPrune(0.25), LevelPrune(0.6)], m)
    @test sparsity(m̄[1].weight) ≈ 0.3 rtol=0.1
    @test count_slices(iszero, m̄[2].weight; dims = 4) ≈ 0.25 * 32
    @test sparsity(m̄[3].weight) ≈ 0.6 rtol=0.1
end

@testset "keepprune" begin
    m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32), Dense(512, 10))
    m̄ = keepprune(prune(LevelPrune(0.5), m))
    for layer in m̄
        @test sparsity(layer.weight) ≈ 0.5 rtol=0.1
        @test layer.weight isa Array
        @test layer.bias isa Array
    end
end

@testset "Training" begin
    m = Chain(Conv((3, 3), 3 => 16), GlobalMeanPool(), Flux.flatten, Dense(16, 10))
    m̄ = prune([ChannelPrune(0.1), identity, identity, LevelPrune(0.1)], m)
    ps = Flux.params(m̄)
    gs = Flux.gradient(() -> sum(m(rand(Float32, 12, 12, 3, 2))), ps)
    @test (Flux.Optimise.update!(Momentum(), ps, gs); true)
end
