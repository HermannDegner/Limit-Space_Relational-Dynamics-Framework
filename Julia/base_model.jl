# RDFEngine.jl
# ------------------------------------------------------------
# non-euclidean-warp.html の力学部分を Julia 側へ切り出すための核。
# JS/p5.js は描画・入力だけを担当し、Julia は世界更新 step! を担当する。
#
# 対応関係:
# particles     -> 粒子状態 S 群
# flowField     -> 整合慣性 M の局所場
# heatField     -> 熱 H
# heatVecField  -> 熱ベクトル H_vec
# goal          -> 外部素流圧 F_goal
# noise         -> 揺らぎ ξ
# leap/death    -> 局所跳躍・再配置
# ------------------------------------------------------------

module RDFEngine

using LinearAlgebra
using Random
using JSON3
using StructTypes

# ------------------------------------------------------------
# 基本型
# ------------------------------------------------------------

const Vec2 = NTuple{2, Float64}

mutable struct Particle
    x::Float64
    y::Float64
    vx::Float64
    vy::Float64
    heat_phase::Float64
    stuck::Int
    deaths::Int
    immunity::Int
end

mutable struct EngineParams
    cell_size::Int
    width::Int
    height::Int
    num_particles::Int

    noise_scale::Float64
    max_speed::Float64

    heat_decay::Float64
    heat_diffuse::Float64
    heat_wall_gain::Float64
    heat_stuck_gain::Float64
    heat_back_gain::Float64
    heat_exp::Float64
    death_threshold::Float64
    death_rate::Float64

    flow_learn::Float64
    flow_decay::Float64
    flow_min::Float64

    goal_pull_init::Float64
    goal_pull_gain::Float64
    goal_pull_max::Float64
    goal_pull_decay::Float64
    goal_gravity_ref::Float64

    phase_gas_threshold::Float64
    phase_noise_scale::Float64
    phase_speed_scale::Float64
    phase_warp_scale::Float64
    phase_learn_scale::Float64
end

function default_params(; width=800, height=600, cell_size=20, num_particles=80)
    EngineParams(
        cell_size, width, height, num_particles,
        0.09, 2.2,
        0.97, 0.10, 0.25, 0.06, 0.04, 0.28, 18.0, 0.25,
        0.07, 0.998, 0.018,
        0.004, 0.004, 0.06, 0.9995, 180.0,
        8.0, 5.0, 1.7, 0.05, 0.08
    )
end

mutable struct EngineState
    params::EngineParams
    cols::Int
    rows::Int

    # grid[y,x] = true なら壁
    grid::Matrix{Bool}

    # flowField[y,x,2] = M 的な局所流れ
    flow::Array{Float64,3}

    # heatVecField[y,x,2]
    heat_vec::Array{Float64,3}

    # heatField[y,x]
    heat::Matrix{Float64}

    particles::Vector{Particle}

    start_cell::Tuple{Int,Int}
    goal_cell::Tuple{Int,Int}
    goal_pixel::Vec2
    goal_pull::Float64
    total_goal_reaches::Int
    tick::Int
end

# JSON3 用
StructTypes.StructType(::Type{Particle}) = StructTypes.Mutable()

# ------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------

norm2(x, y) = sqrt(x*x + y*y)

function limit_vec(vx, vy, maxv)
    n = norm2(vx, vy)
    if n > maxv && n > 1e-9
        s = maxv / n
        return vx*s, vy*s
    end
    return vx, vy
end

function normalize_vec(vx, vy)
    n = norm2(vx, vy)
    if n < 1e-9
        θ = 2π * rand()
        return cos(θ), sin(θ)
    end
    return vx/n, vy/n
end

function inbounds(st::EngineState, cx::Int, cy::Int)
    1 <= cx <= st.cols && 1 <= cy <= st.rows
end

function cell_of(st::EngineState, x, y)
    cs = st.params.cell_size
    cx = clamp(floor(Int, x / cs) + 1, 1, st.cols)
    cy = clamp(floor(Int, y / cs) + 1, 1, st.rows)
    return cx, cy
end

function is_wall_at(st::EngineState, x, y)
    cx, cy = cell_of(st, x, y)
    return st.grid[cy, cx]
end

function rand_open_cell_near(st::EngineState, center::Tuple{Int,Int}; radius=2)
    sx, sy = center
    for _ in 1:300
        cx = clamp(sx + rand(-radius:radius), 1, st.cols)
        cy = clamp(sy + rand(-radius:radius), 1, st.rows)
        if !st.grid[cy, cx]
            return cx, cy
        end
    end
    return sx, sy
end

function cell_center(st::EngineState, cx, cy)
    cs = st.params.cell_size
    return ((cx - 0.5) * cs, (cy - 0.5) * cs)
end

# ------------------------------------------------------------
# 初期化
# ------------------------------------------------------------

function make_random_grid(cols, rows; wall_rate=0.18)
    grid = falses(rows, cols)
    for y in 1:rows, x in 1:cols
        edge = x == 1 || y == 1 || x == cols || y == rows
        grid[y,x] = edge || rand() < wall_rate
    end
    return grid
end

function clear_around!(grid, cx, cy, r)
    rows, cols = size(grid)
    for y in max(1,cy-r):min(rows,cy+r), x in max(1,cx-r):min(cols,cx+r)
        grid[y,x] = false
    end
end

function init_engine(; seed=42, params=default_params())
    Random.seed!(seed)
    cols = params.width ÷ params.cell_size
    rows = params.height ÷ params.cell_size

    start_cell = (2, 2)
    goal_cell = (cols - 1, rows - 1)

    grid = make_random_grid(cols, rows)
    clear_around!(grid, start_cell[1], start_cell[2], 2)
    clear_around!(grid, goal_cell[1], goal_cell[2], 2)

    flow = zeros(Float64, rows, cols, 2)
    heat_vec = zeros(Float64, rows, cols, 2)
    heat = zeros(Float64, rows, cols)

    dummy = EngineState(params, cols, rows, grid, flow, heat_vec, heat,
        Particle[], start_cell, goal_cell, (0.0,0.0),
        params.goal_pull_init, 0, 0)

    gx, gy = cell_center(dummy, goal_cell[1], goal_cell[2])
    dummy.goal_pixel = (gx, gy)

    particles = Particle[]
    for _ in 1:params.num_particles
        cx, cy = rand_open_cell_near(dummy, start_cell)
        px, py = cell_center(dummy, cx, cy)
        θ = 2π * rand()
        push!(particles, Particle(px, py, cos(θ)*0.8, sin(θ)*0.8, 0.0, 0, 0, 40))
    end
    dummy.particles = particles
    return dummy
end

# ------------------------------------------------------------
# 外部入力
# JS から渡す最小入力
# ------------------------------------------------------------

Base.@kwdef mutable struct EngineInput
    goal_x::Float64 = 760.0
    goal_y::Float64 = 560.0
    mouse_goal::Bool = false
    remap::Bool = false
end

function apply_input!(st::EngineState, input::EngineInput)
    st.goal_pixel = (input.goal_x, input.goal_y)
    st.goal_cell = cell_of(st, input.goal_x, input.goal_y)
end

# ------------------------------------------------------------
# 熱拡散・散逸
# ------------------------------------------------------------

function decay_heat!(st::EngineState)
    p = st.params
    st.heat_vec .*= p.heat_decay

    diff = zeros(size(st.heat_vec))

    for y in 1:st.rows, x in 1:st.cols
        st.grid[y,x] && continue
        hx = st.heat_vec[y,x,1]
        hy = st.heat_vec[y,x,2]
        h = norm2(hx, hy)
        h < 0.05 && continue

        ux, uy = normalize_vec(hx, hy)
        radius = min(floor(Int, h / 2.5) + 1, 5)
        spread = h * p.heat_diffuse

        cells = Tuple{Int,Int,Float64}[]
        totalw = 0.0
        for dy in -radius:radius, dx in -radius:radius
            dx == 0 && dy == 0 && continue
            dist = sqrt(dx*dx + dy*dy)
            dist > radius && continue
            nx, ny = x + dx, y + dy
            if inbounds(st, nx, ny) && !st.grid[ny,nx]
                w = 1.0 / (dist*dist + 0.5)
                push!(cells, (nx, ny, w))
                totalw += w
            end
        end

        totalw <= 0 && continue
        for (nx, ny, w) in cells
            amount = spread * (w / totalw)
            diff[ny,nx,1] += ux * amount
            diff[ny,nx,2] += uy * amount
        end
        diff[y,x,1] -= ux * spread
        diff[y,x,2] -= uy * spread
    end

    st.heat_vec .+= diff

    for y in 1:st.rows, x in 1:st.cols
        h = norm2(st.heat_vec[y,x,1], st.heat_vec[y,x,2])
        st.heat[y,x] = max(0.0, h)
    end
end

function decay_flow!(st::EngineState)
    p = st.params
    for y in 1:st.rows, x in 1:st.cols
        fx = st.flow[y,x,1]
        fy = st.flow[y,x,2]
        m = norm2(fx, fy)
        m < 0.001 && continue
        fx *= p.flow_decay
        fy *= p.flow_decay
        m2 = norm2(fx, fy)
        if m2 < p.flow_min
            ux, uy = normalize_vec(fx, fy)
            fx, fy = ux*p.flow_min, uy*p.flow_min
        end
        st.flow[y,x,1] = fx
        st.flow[y,x,2] = fy
    end
end

# ------------------------------------------------------------
# 局所ワープ M(flow) による速度変形
# ------------------------------------------------------------

function warp_velocity(vx, vy, fx, fy, heat, phase)
    fmag = norm2(fx, fy)
    fmag < 0.01 && return vx, vy

    # heat が高いほど既存Mの拘束が弱まる。
    # phase が高いほど気体化して flow の拘束を受けにくくなる。
    wc = clamp(fmag / (1 + heat * 0.12), 0.0, 1.0) * (1 - phase)

    vθ = atan(vy, vx)
    fθ = atan(fy, fx)
    ad = fθ - vθ
    while ad > π; ad -= 2π; end
    while ad < -π; ad += 2π; end

    newθ = vθ + ad * 0.55 * wc
    speed = norm2(vx, vy) * (1 + fmag * 0.25 * wc)
    return cos(newθ)*speed, sin(newθ)*speed
end

# ------------------------------------------------------------
# 粒子更新
# ------------------------------------------------------------

function respawn!(st::EngineState, pt::Particle)
    cx, cy = rand_open_cell_near(st, st.start_cell)
    pt.x, pt.y = cell_center(st, cx, cy)
    θ = 2π * rand()
    pt.vx = cos(θ) * st.params.max_speed * 0.4
    pt.vy = sin(θ) * st.params.max_speed * 0.4
    pt.stuck = 0
    pt.immunity = 40
end

function update_particle!(st::EngineState, pt::Particle)
    p = st.params
    cx, cy = cell_of(st, pt.x, pt.y)

    if st.grid[cy,cx]
        respawn!(st, pt)
        return
    end

    local_heat = st.heat[cy,cx]
    phase = clamp(local_heat / p.phase_gas_threshold, 0.0, 1.0)
    pt.heat_phase = phase

    # 局所跳躍 / death
    if pt.immunity > 0
        pt.immunity -= 1
    elseif local_heat > p.death_threshold
        excess = local_heat - p.death_threshold
        if rand() < 1 - exp(-excess * p.death_rate)
            ux, uy = normalize_vec(pt.vx, pt.vy)
            burst = p.death_threshold * 2.5
            st.heat_vec[cy,cx,1] += ux * burst
            st.heat_vec[cy,cx,2] += uy * burst
            pt.deaths += 1
            respawn!(st, pt)
            return
        end
    end

    gx, gy = st.goal_pixel
    dx = gx - pt.x
    dy = gy - pt.y
    dist = max(norm2(dx, dy), p.cell_size * 1.5)
    gdx, gdy = normalize_vec(dx, dy)

    grav = st.goal_pull * (p.goal_gravity_ref^2) / (dist^2)
    grav = min(grav, p.max_speed * 0.45)
    grav *= 1 / (1 + local_heat * 0.18)
    grav *= 1 / (1 + pt.stuck * 0.25)

    Fx = gdx * grav
    Fy = gdy * grav

    # ξ：熱相が高いほど揺らぎ増大
    noise = p.noise_scale * ((1 - phase) + p.phase_noise_scale * phase)
    θ = 2π * rand()
    Fx += cos(θ) * noise
    Fy += sin(θ) * noise

    # 状態更新 S(t+1) = S + F + ξ
    pt.vx += Fx
    pt.vy += Fy

    maxspd = p.max_speed * ((1 - phase) + p.phase_speed_scale * phase)
    pt.vx, pt.vy = limit_vec(pt.vx, pt.vy, maxspd)

    # M(flow) による局所ワープ
    fx = st.flow[cy,cx,1] * ((1 - phase) + p.phase_warp_scale * phase)
    fy = st.flow[cy,cx,2] * ((1 - phase) + p.phase_warp_scale * phase)
    pt.vx, pt.vy = warp_velocity(pt.vx, pt.vy, fx, fy, local_heat, phase)
    pt.vx, pt.vy = limit_vec(pt.vx, pt.vy, maxspd)

    oldx, oldy = pt.x, pt.y
    nx = pt.x + pt.vx
    ny = pt.y + pt.vy

    moved = false
    avx, avy = 0.0, 0.0

    if !is_wall_at(st, nx, ny)
        pt.x, pt.y = nx, ny
        avx, avy = pt.vx, pt.vy
        moved = true
        pt.stuck = 0
    elseif !is_wall_at(st, nx, pt.y)
        pt.x = nx
        pt.vy *= 0.3
        avx, avy = pt.vx, 0.0
        moved = true
        pt.stuck = 0
    elseif !is_wall_at(st, pt.x, ny)
        pt.y = ny
        pt.vx *= 0.3
        avx, avy = 0.0, pt.vy
        moved = true
        pt.stuck = 0
    else
        pt.stuck += 1
        θ = 2π * rand()
        pt.vx = cos(θ) * p.max_speed * 0.7
        pt.vy = sin(θ) * p.max_speed * 0.7
    end

    # 誤差 E = intended velocity - actual velocity
    evx = pt.vx - avx
    evy = pt.vy - avy
    err = max(0.0, norm2(evx, evy) - 0.30)
    heat_gain = err^2 * 0.8
    gain_scale = 1 + local_heat * p.heat_exp

    if !moved
        ux, uy = normalize_vec(pt.vx, pt.vy)
        st.heat_vec[cy,cx,1] += ux * (p.heat_wall_gain + heat_gain) * gain_scale
        st.heat_vec[cy,cx,2] += uy * (p.heat_wall_gain + heat_gain) * gain_scale
    else
        moved_dist = norm2(pt.x - oldx, pt.y - oldy)
        if moved_dist < 0.3
            ux, uy = normalize_vec(pt.vx, pt.vy)
            st.heat_vec[cy,cx,1] += ux * p.heat_stuck_gain * gain_scale
            st.heat_vec[cy,cx,2] += uy * p.heat_stuck_gain * gain_scale
        end

        if heat_gain > 0.001
            ux, uy = normalize_vec(evx, evy)
            st.heat_vec[cy,cx,1] += ux * heat_gain * gain_scale
            st.heat_vec[cy,cx,2] += uy * heat_gain * gain_scale
        end

        # flowField = 成功した移動の記憶、局所Mとして学習
        speed = norm2(pt.vx, pt.vy)
        if speed > 0.05
            ux, uy = normalize_vec(pt.vx, pt.vy)
            learn = p.flow_learn * ((1 - phase) + p.phase_learn_scale * phase)
            st.flow[cy,cx,1] = (1 - learn) * st.flow[cy,cx,1] + learn * ux * speed
            st.flow[cy,cx,2] = (1 - learn) * st.flow[cy,cx,2] + learn * uy * speed
        end
    end

    st.heat[cy,cx] = norm2(st.heat_vec[cy,cx,1], st.heat_vec[cy,cx,2])

    # ゴール到達
    if norm2(pt.x - gx, pt.y - gy) < p.cell_size * 1.2
        st.goal_pull = min(st.goal_pull + p.goal_pull_gain, p.goal_pull_max)
        st.total_goal_reaches += 1
        respawn!(st, pt)
    end
end

# ------------------------------------------------------------
# 1ステップ更新
# ------------------------------------------------------------

function step!(st::EngineState, input::EngineInput=EngineInput())
    apply_input!(st, input)

    decay_heat!(st)
    decay_flow!(st)

    for pt in st.particles
        update_particle!(st, pt)
    end

    st.goal_pull = max(st.goal_pull * st.params.goal_pull_decay, st.params.goal_pull_init)
    st.tick += 1
    return st
end

# ------------------------------------------------------------
# JSへ返す軽量スナップショット
# ------------------------------------------------------------

function snapshot(st::EngineState)
    particles = [Dict(
        "x" => pt.x,
        "y" => pt.y,
        "vx" => pt.vx,
        "vy" => pt.vy,
        "phase" => pt.heat_phase,
        "stuck" => pt.stuck,
        "deaths" => pt.deaths
    ) for pt in st.particles]

    Dict(
        "tick" => st.tick,
        "cols" => st.cols,
        "rows" => st.rows,
        "cellSize" => st.params.cell_size,
        "particles" => particles,
        "heat" => st.heat,
        "heatVec" => st.heat_vec,
        "flow" => st.flow,
        "goalPull" => st.goal_pull,
        "totalGoalReaches" => st.total_goal_reaches
    )
end

# ------------------------------------------------------------
# APIサーバ最小例
# ------------------------------------------------------------

function serve!(st::EngineState; host="127.0.0.1", port=8088)
    @eval begin
        using HTTP
    end

    HTTP.serve(host, port) do req
        if req.method == "OPTIONS"
            return HTTP.Response(200, [
                "Access-Control-Allow-Origin" => "*",
                "Access-Control-Allow-Headers" => "Content-Type",
                "Access-Control-Allow-Methods" => "POST, OPTIONS"
            ], "")
        end

        if req.target == "/step" && req.method == "POST"
            body = String(req.body)
            data = isempty(body) ? Dict() : JSON3.read(body, Dict)

            input = EngineInput(
                goal_x = Float64(get(data, "goal_x", st.goal_pixel[1])),
                goal_y = Float64(get(data, "goal_y", st.goal_pixel[2])),
                mouse_goal = Bool(get(data, "mouse_goal", false)),
                remap = Bool(get(data, "remap", false))
            )

            step!(st, input)
            return HTTP.Response(200, [
                "Content-Type" => "application/json",
                "Access-Control-Allow-Origin" => "*"
            ], JSON3.write(snapshot(st)))
        end

        return HTTP.Response(404, "not found")
    end
end

end # module RDFEngine

# ------------------------------------------------------------
# run_server.jl 例
# ------------------------------------------------------------
# include("RDFEngine.jl")
# using .RDFEngine
# st = RDFEngine.init_engine(seed=42)
# RDFEngine.serve!(st; port=8088)

# ------------------------------------------------------------
# JS側の最小接続イメージ
# ------------------------------------------------------------
# async function pullJuliaStep() {
#   const res = await fetch("http://127.0.0.1:8088/step", {
#     method: "POST",
#     headers: {"Content-Type": "application/json"},
#     body: JSON.stringify({
#       goal_x: mouseGoalMode ? mouseGX : (goalCell.x + 0.5) * cellSize,
#       goal_y: mouseGoalMode ? mouseGY : (goalCell.y + 0.5) * cellSize,
#       mouse_goal: mouseGoalMode
#     })
#   });
#   const world = await res.json();
#   // world.particles, world.heat, world.heatVec, world.flow を描画に使う
# }
