# RDF Julia エンジン設計図

> 更新メモ：本設計図では、短期実装を **対角近似版M** として扱う。行列版M、H_vec、ドーパミンによるF変調、p5.js連携通信層は中期拡張として分離する。

## 目的

RDFエンジンを、意味判断から切り離した計算基盤として設計する。

本設計では、Julia は **物理層・構造層・神経力学層の計算カーネル** を担当し、p5.js などの上位環境は **意味付け・判断・描画・行動制御** を担当する。

---

## 基本方針

```text
Julia = 計算・構造更新・神経力学的変調
p5.js = 判断・制御・描画・意味付け
```

Julia は「逃げる」「食べる」「死ぬ」「跳躍する」といった判断を行わない。

Julia は以下を返すだけにする。

```text
S_new       更新後状態
M_new       更新後構造
H_new       更新後熱
E           誤差
ξ           揺らぎ
leap_value  跳躍条件との差分
```

判断は外部層が行う。

---

## 全体階層

```text
[外部環境 / p5.js]
  - 描画
  - 入力
  - 行動制御
  - 食べる / 逃げる / 死ぬ / 跳躍演出
  - 意味付け

        ↑ / ↓

[Julia RDF Engine]
  ③ 神経力学層
  ② ネットワーク処理層
  ① 計算層
```

`RDFParams` は研究者・開発者用の低次パラメータ群として扱う。
実運用や個体設計では、少数の上位ラベルを持つ `AgentProfile` から `RDFParams`、`NeuroState`、`theta` へ展開する。

```julia
profile = AgentProfile(stability=0.8, curiosity=0.3, caution=0.7)
params, neuro, theta = make_profile(profile)
```

---

# ① 計算層 / Physics Core

## 役割

最下層の純粋計算層。
意味を持たず、状態更新だけを行う。

担当：

```text
S更新
E計算
H更新
ξ生成
```

## 基本式

```text
S(t+1) = M S(t) + F(t) + ξ(t)
```

または短期実装では、要素積版として：

```text
S_new = M .* S + F + ξ
```

これは **対角近似** である。

```text
M .* S ≒ Diagonal(M) * S
```

この近似では、整合方向 V と強度 Λ の分離は扱わない。
中期以降で `M = VΛV⁻¹` を扱う行列版へ拡張する。

## Julia例

```julia
Base.@kwdef struct RDFParams
    # Physics core
    T::Float64 = 0.01
    alpha::Float64 = 0.01
    K::Float64 = 1.0
    eps::Float64 = 1e-6
    theta_min::Float64 = 0.05
    theta_max::Float64 = 5.0

    # Structure network
    eta::Float64 = 0.01
    decay::Float64 = 0.001
    M_min::Float64 = 0.05
    M_max::Float64 = 5.0
    H_opt::Float64 = 1.0
    H_width::Float64 = 0.5

    # Neuro modulation weights
    serotonin_theta_gain::Float64 = 1.0
    norad_theta_gain::Float64 = 0.5
    dopamine_theta_gain::Float64 = 0.3
end

Base.@kwdef struct AgentProfile
    speed::Float64 = 0.5
    stability::Float64 = 0.5
    sensitivity::Float64 = 0.5
    curiosity::Float64 = 0.5
    caution::Float64 = 0.5
    bonding::Float64 = 0.5
end

function noise_time_scale(params::RDFParams)
    return sqrt(params.T)
end

function core_step(S, M, F, H, params::RDFParams; rng=Random.default_rng())
    E = F .- M .* S
    err = norm(E)

    H = H + params.T * (err^2 - params.alpha * H)

    # 対角近似版 ξ
    # D[ξ_i] ≈ K / |M_i|
    # Mが強い方向ほど揺らぎが乗りにくい、という近似。
    ξ = noise_time_scale(params) .* (params.K ./ (abs.(M) .+ params.eps)) .* randn(rng, length(S))

    # RDF的な連続更新
    S_new = S .+ params.T .* (-M .* S .+ F .+ ξ)

    return S_new, H, E, ξ
end
```

---

# ② ネットワーク処理層 / Structure Network

## 役割

M、つまり整合慣性・構造ネットワークを扱う。

担当：

```text
M更新
整合成功方向の強化
構造的拘束の調整
```

## 方針

M は普段は安定的に扱う。
ただし、誤差を処理できた方向では成長する。
大きな誤差は「成長しない」のではなく、処理難度が高い状態として扱う。
可塑性が開いている熱状態で統合できたなら、大きな誤差ほど大きな構造更新の材料になる。

```text
低熱 → 流れるだけで構造化しにくい
微誤差 → 微成長
大誤差 → 処理は難しいが、統合できたなら大成長
高熱 → 処理困難として成長抑制、または崩壊側へ
```

## Julia例

```julia
function heat_gate(H, params::RDFParams)
    return exp(-((H - params.H_opt)^2) / (2 * params.H_width^2))
end

function error_difficulty(E)
    return log1p(norm(E))
end

function update_M(M, S, E, H, params::RDFParams)
    difficulty = error_difficulty(E)
    plasticity = heat_gate(H, params)

    s = S / (norm(S) + params.eps)

    # 対角近似版Mでは、外積ではなく各方向の二乗を使う。
    # 行列版Mでは ΔM = s * s' を使う。
    ΔM = s .* s

    M_new = (1 - params.decay) .* M .+ params.eta * difficulty * plasticity .* ΔM

    return clamp.(M_new, params.M_min, params.M_max)
end
```

---

# ③ 神経力学層 / Neuro Dynamics

## 役割

内部状態によって、閾値・感度・探索性を変動させる。

担当：

```text
θ調整
熱感度調整
揺らぎ強度調整
学習ゲート調整
```

## 神経力学的パラメータ例

物理計算・熱・揺らぎ・構造更新の基礎設定は `RDFParams` にまとめる。

```text
T       時間刻み / 更新率
alpha   熱散逸率
K       揺らぎ保存量・個体/系パラメータ
eps     値0を扱うための数値的逃がし
theta_min thetaの下限安全装置
theta_max thetaの上限安全装置
eta       Mの学習率・構造化速度
decay     Mの自然減衰率
M_min     完全自由化を防ぐ最低整合慣性
M_max     硬直・絶対化を防ぐ最大整合慣性
H_opt     最適可塑性熱
H_width   可塑性が開いている熱範囲
*_theta_gain  神経状態がthetaに効く係数
```

外部操作では、低次パラメータを直接触る代わりに `AgentProfile` を使う。

```text
speed        更新速度
stability    熱散逸・セロトニン・高theta・低etaへ展開
sensitivity  揺らぎ幅・可塑性幅へ展開
curiosity    dopamine・低theta・高Kへ展開
caution      norad・高theta・危機反応へ展開
bonding      oxytocinへ展開
```

神経力学層は、その時点の神経状態を `NeuroState` として扱う。

```text
serotonin   誤差耐性・安定性
norad       危機検出・即応性
dopamine    探索性・誤差解決への引力
oxytocin    結合・自己拡張的重み付け
```

短期実装では dopamine を theta 低下として近似する。
中期実装では dopamine を `F` の重み変換へ分離する。

```text
短期：dopamine → theta低下
中期：dopamine → Fの符号・重み変換
```

## Julia例

```julia
Base.@kwdef struct NeuroState
    serotonin::Float64 = 1.0
    norad::Float64 = 0.0
    dopamine::Float64 = 0.0
    oxytocin::Float64 = 0.0
end

function neuro_mod(H, theta, neuro::NeuroState, params::RDFParams)
    theta_mod = theta

    theta_mod *= params.serotonin_theta_gain * neuro.serotonin
    theta_mod -= params.norad_theta_gain * neuro.norad
    theta_mod -= params.dopamine_theta_gain * neuro.dopamine

    return clamp(theta_mod, params.theta_min, params.theta_max)
end
```

---

# 統合ステップ

## 方針

Julia側では判断しない。

`H > theta` の if 判定も原則として外部に渡す。

Juliaは `leap_value = H_new - theta_mod` を返すだけにする。

## Julia例

```julia
function rdf_step(S, M, F, H, theta, params::RDFParams, neuro::NeuroState; rng=Random.default_rng())
    S_new, H_new, E, ξ = core_step(S, M, F, H, params; rng=rng)

    theta_mod = neuro_mod(H_new, theta, neuro, params)

    M_new = update_M(M, S_new, E, H_new, params)

    leap_value = H_new - theta_mod

    return S_new, M_new, H_new, E, ξ, leap_value
end

# 再現性を確認する場合は、外からrngを注入する。
rng = MersenneTwister(42)
profile = AgentProfile(stability=0.8, curiosity=0.3, caution=0.7)
params, neuro, theta = make_profile(profile)
result = rdf_step(S, M, F, H, theta, params, neuro; rng=rng)
```

---

# p5.js 側の役割

p5.js 側は、Juliaが返した値を使って判断する。

```javascript
const result = rdfStep(S, M, F, H, theta, params);

S = result.S_new;
M = result.M_new;
H = result.H_new;

if (result.leap_value > 0) {
  leap();
}
```

p5.js 側が担当するもの：

```text
草を食べる
肉食が追う
死亡判定
跳躍演出
速度制限
描画
環境との接触判定
```

---

# RDF的意味整理

## Julia側

```text
意味を持たない物理計算層
```

Juliaは「何が起きたか」を知らない。
ただ数値を更新する。

## p5.js側

```text
意味を貼る観測・制御層
```

p5.jsが、距離・接触・捕食・摂食・死亡などの意味を割り当てる。

---

# 設計上の重要ルール

## 1. Julia は判断しない

```text
NG: if H > theta then leap!
OK: leap_value = H - theta を返す
```

## 2. Julia は意味語を知らない

```text
NG: eat, die, escape, chase
OK: S, M, F, H, E, ξ
```

## 3. M は通常安定、成功時に微成長

```text
低熱 → 構造化しにくい
微誤差処理 → M微成長
大誤差処理 → M大成長
高熱 → 処理困難として成長抑制
```

## 4. 0と1の扱い

```text
構造0：許可する
値0：必要ならεで逃がす
1：近似層では許可、理論層では極限扱い
```

## 5. NAMとBase Modelを分ける

```text
NAM：安定領域の低負荷近似。0/1使用可。
Base Model：揺らぎ・熱・跳躍を含む。0/1は極限扱い。
```

---

# 今後の拡張

## 短期

```text
- rdf_step の最小実装
- 対角近似版Mで安定動作確認
- rng注入による再現性確保
- p5.js から呼び出す形式の整理
- 草食・肉食の計算部分だけJulia化
```

## 中期

```text
- Mを行列版へ拡張
- H_error と H_boredom を分離
- HをH_vecへ拡張
- 神経力学パラメータを個体ごとに持たせる
- dopamineをF変調へ分離
- p5.js ↔ Julia の通信層を決める
```

行列版Mに備えた型階層案：

```julia
abstract type IntegrationInertia end

struct DiagonalM <: IntegrationInertia
    diag::Vector{Float64}
end

struct FullM <: IntegrationInertia
    V::Matrix{Float64}
    Λ::Vector{Float64}
end
```

この構造により、Juliaの多重ディスパッチで `core_step` を切り替えられる。

## 長期

```text
- LLMを外部構造圧 F として接続
- RDFエージェントの自己学習
- 多モーダル環境との統合
- Julia側のライブラリ化
```

---

# 一文圧縮

RDF Julia エンジンは、意味判断を外部に委ね、S・M・F・H・ξ の更新だけを行う物理計算カーネルとして設計する。
