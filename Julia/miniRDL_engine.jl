using LinearAlgebra
using Random

mutable struct RDFState
    S::Vector{Float64}      # 状態
    M::Matrix{Float64}      # 整合慣性
    F::Vector{Float64}      # 素流圧
    H_error::Float64        # 誤差熱
    H_boredom::Float64      # 退屈熱
    θ::Float64              # 跳躍閾値
    K::Float64              # 揺らぎ総量
end

function rdf_noise(state::RDFState, E; T=0.01)
    M_strength = norm(state.M) + 1e-6
    Dξ = state.K / M_strength
    return T .* Dξ .* randn(length(state.S)) .* (1 .+ abs.(E))
end

function base_step(state::RDFState; T=0.01)
    E = state.F - state.M * state.S
    ξ = rdf_noise(state, E; T=T)
    S_new = state.M * state.S .+ state.F .+ ξ
    return S_new, E, ξ
end

function update_heat!(state::RDFState, E;
                      α_error=0.1,
                      α_boredom=0.05,
                      ε=0.01,
                      boredom_rate=0.02)

    err = norm(E)

    # 誤差熱
    state.H_error += err^2
    state.H_error *= (1 - α_error)

    # 退屈熱：誤差が小さすぎると増える
    if err < ε
        state.H_boredom += boredom_rate
    else
        state.H_boredom *= (1 - α_boredom)
    end
end

function update_M!(state::RDFState, S_new, E;
                   η=0.01,
                   H_opt=1.0,
                   width=0.8)

    err = norm(E)

    # 誤差が小さいほど整合成功
    success = exp(-err)

    # 熱ゲート：中熱で学習しやすい
    H = state.H_error
    heat_gate = exp(-((H - H_opt)^2) / (2 * width^2))

    s = S_new / (norm(S_new) + 1e-6)
    ΔM = s * s'

    state.M .+= η * success * heat_gate .* ΔM
end

function leap!(state::RDFState)
    # 最小跳躍：Mを少し緩めて、熱を残しつつ下げる
    state.M .*= 0.98
    state.H_error *= 0.3
    state.H_boredom *= 0.5
end

function step!(state::RDFState; T=0.01)
    S_new, E, ξ = base_step(state; T=T)

    update_heat!(state, E)
    update_M!(state, S_new, E)

    H_total = state.H_error + state.H_boredom

    if H_total > state.θ
        leap!(state)
    end

    state.S = S_new

    return (
        S = copy(state.S),
        E = copy(E),
        ξ = copy(ξ),
        H_error = state.H_error,
        H_boredom = state.H_boredom,
        H_total = H_total,
        M_norm = norm(state.M)
    )
end

function main()
    S = [1.0, 0.5]
    M = [0.9 0.0;
         0.0 0.8]
    F = [0.1, -0.05]

    state = RDFState(
        S,
        M,
        F,
        0.0,   # H_error
        0.0,   # H_boredom
        2.0,   # θ
        1.0    # K
    )

    for t in 1:100
        log = step!(state; T=0.02)

        if t % 10 == 0
            println(
                "t=", t,
                " S=", round.(log.S, digits=3),
                " H=", round(log.H_total, digits=3),
                " |M|=", round(log.M_norm, digits=3)
            )
        end
    end
end

main()