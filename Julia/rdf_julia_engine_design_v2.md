# RDF Julia エンジン設計図 v2.0

*RDF概念的数理モデル v2.2 準拠 / 0から再設計*

---

## 設計方針

```text
Julia = EFP解釈・慣性誤差計算・M更新・神経力学変調の計算カーネル
p5.js = 意味付け・判断・描画・行動制御
```

v2.2の基本構造に素直に従う。

```text
F(t)  = interp(M, EFP)        // MがEFPを解釈
E(t)  = F(t+Δ) - M·F(t)      // 慣性誤差
dM/dt = f(M, E, ξ)            // Mは誤差と揺らぎで更新される
```

**Sは存在しない。Mが主役である。**

---

## 全体階層

```text
[外部環境 / p5.js]
  - 描画
  - 行動制御（食べる・逃げる・死ぬ・跳躍演出）
  - EFP生成（環境から流入する作用束の構成）
  - 意味付け・判断

        ↑ / ↓

[Julia RDF Engine]
  ③ 神経力学層　NeuroState によるθ・感度変調
  ② 構造更新層　M更新・整合慣性の管理
  ① 計算核　　　EFP解釈・慣性誤差・熱・揺らぎ
```

Juliaは判断しない。`leap_value = H - θ_mod` を返すだけ。判断はp5.js側が行う。

---

## 返り値の設計

```text
Juliaが返すもの：

F_new       MがEFPを解釈した作用解釈F
M_new       更新後の整合慣性
H_new       更新後の熱
E           慣性誤差（F_new - M·F_prev）
ξ           揺らぎ
leap_value  跳躍条件との差分（H - θ_mod）
```

p5.jsはF_newを位置・速度・状態の代わりとして使う。

---

# ① 計算核 / Physics Core

## 役割

```text
EFP → F = interp(M, EFP)
E   = F_new - M·F_prev
H   更新
ξ   生成
```

## 対角近似版の解釈

短期実装では対角近似を使う。

```text
interp(M, EFP) ≈ M .* EFP

意味：Mの各次元がEFPの各次元を重み付けして解釈する
MがリッチなほどEFPから多くを読み取れる
MがフラットならEFPはほぼ素通り
```

慣性投影：

```text
M·F_prev ≈ M .* F_prev

意味：現在のFからMが慣性として次に流れ込む値
```

慣性誤差：

```text
E = F_new - M .* F_prev

意味：慣性投影と実際に来たFの差
これが熱の源泉であり、M更新の駆動力
```

## Julia実装

```julia
Base.@kwdef struct RDFParams
    # 基本時間・熱
    dt::Float64       = 0.01    # 時間刻み
    alpha::Float64    = 0.01    # 熱散逸率
    K::Float64        = 1.0     # 関係保存量（揺らぎ保存定数）
    eps::Float64      = 1e-6    # ゼロ除算防止

    # 跳躍閾値
    theta_min::Float64 = 0.05
    theta_max::Float64 = 5.0

    # M更新
    eta::Float64      = 0.01    # M学習率
    decay::Float64    = 0.001   # M自然減衰率
    M_min::Float64    = 0.05    # M下限（完全自由化防止）
    M_max::Float64    = 5.0     # M上限（絶対化防止）
    H_opt::Float64    = 1.0     # 最適可塑性熱
    H_width::Float64  = 0.5     # 可塑性が開いている熱範囲

    # 神経力学ゲイン
    serotonin_theta_gain::Float64  = 1.0
    norad_theta_gain::Float64      = 0.5
    dopamine_theta_gain::Float64   = 0.3
end

"""
    interp_diag(M, EFP)

対角近似版の解釈関数。
MがEFPを重み付けして作用解釈Fを生成する。

中期以降でinterp(M, EFP) = M * EFP（行列版）へ拡張する。
"""
function interp_diag(M, EFP)
    return M .* EFP
end

"""
    compute_inertial_error(M, F_new, F_prev)

慣性誤差 E = F_new - M·F_prev を計算する。

F_new  : 今ステップの作用解釈F（interp(M, EFP_new)）
F_prev : 前ステップの作用解釈F
M·F_prev : 慣性投影（Mの慣性がF_prevに乗ったまま次へ流れた値）
"""
function compute_inertial_error(M, F_new, F_prev)
    inertial_projection = M .* F_prev
    return F_new .- inertial_projection
end

"""
    generate_xi(M, params; rng)

揺らぎ生成。関係保存則 ||M|| · D[ξ] = K より。
Mが強い次元ほど揺らぎが小さい。
"""
function generate_xi(M, params::RDFParams; rng=Random.default_rng())
    scale = sqrt(params.dt) .* (params.K ./ (abs.(M) .+ params.eps))
    return scale .* randn(rng, length(M))
end

"""
    core_step(EFP_new, F_prev, M, H, params; rng)

計算核の1ステップ。

入力：
  EFP_new  今ステップのEFP（未解釈の作用束）
  F_prev   前ステップの作用解釈F（慣性投影の基点）
  M        現在の整合慣性
  H        現在の熱

出力：
  F_new    今ステップの作用解釈F
  H_new    更新後の熱
  E        慣性誤差
  ξ        揺らぎ
"""
function core_step(EFP_new, F_prev, M, H, params::RDFParams; rng=Random.default_rng())
    # EFPをMで解釈 → 作用解釈F
    F_new = interp_diag(M, EFP_new)

    # 慣性誤差：慣性投影と実際のFの差
    E = compute_inertial_error(M, F_new, F_prev)

    # 熱更新：慣性誤差の二乗ノルムが熱源
    err_sq = norm(E)^2
    H_new = H + params.dt * (err_sq - params.alpha * H)

    # 揺らぎ生成
    ξ = generate_xi(M, params; rng=rng)

    return F_new, H_new, E, ξ
end
```

---

# ② 構造更新層 / Structure Update

## 役割

```text
dM/dt = f(M, E, ξ)

慣性誤差Eと揺らぎξによってMを更新する。
誤差を処理できた方向でMが強化される。
```

## 更新の方針

```text
低熱   → 可塑性が低い・流れるだけで構造化しにくい
適正熱 → 可塑性が最大・誤差を処理してMが成長
高熱   → 可塑性が低下・処理困難として成長抑制または崩壊
```

## Julia実装

```julia
"""
    heat_gate(H, params)

熱による可塑性ゲート。
H_optで最大、離れるほど可塑性低下。
ガウス型で実装。
"""
function heat_gate(H, params::RDFParams)
    return exp(-((H - params.H_opt)^2) / (2 * params.H_width^2))
end

"""
    update_M(M, F, E, H, ξ, params)

M更新。v2.2基本式 dM/dt = f(M, E, ξ) の対角近似実装。

Fの方向（正規化）に沿って、
慣性誤差Eの難度と熱可塑性の積でMを成長させる。
揺らぎξが根源的な自由度として関与する。
"""
function update_M(M, F, E, H, ξ, params::RDFParams)
    # Fの方向（どの次元で整合が起きているか）
    f_norm = F ./ (norm(F) .+ params.eps)

    # 誤差の難度：大きな誤差ほど更新の材料になる
    difficulty = log1p(norm(E))

    # 熱可塑性ゲート
    plasticity = heat_gate(H, params)

    # 対角近似版：F方向の二乗で成長
    # 行列版では ΔM = f_norm * f_norm' を使う
    ΔM_det = f_norm .^ 2

    # ξによる確率的自由度（根源的揺らぎとしての寄与）
    ΔM_stoch = params.K .* ξ ./ (abs.(M) .+ params.eps)

    # M更新
    M_new = (1 - params.decay) .* M .+
            params.eta * difficulty * plasticity .* ΔM_det .+
            params.dt .* ΔM_stoch

    return clamp.(M_new, params.M_min, params.M_max)
end
```

---

# ③ 神経力学層 / Neuro Dynamics

## 役割

```text
内部神経状態によってθ・感度・探索性を変調する。
```

## 神経物質の意味（v2.2対応）

```text
ドーパミン　：誤差を求め誤差解決時に快。θを下げ跳躍しやすくする。
セロトニン　：誤差耐性・安定性。θを上げ熱を散逸しやすくする。
ノルアドレナリン：生存脅威の検出・即応。θを下げ緊急跳躍を促す。
オキシトシン：自己拡張的重み付け。EFPの自他境界を動的に変える（中期実装）。
```

```text
短期実装：dopamine → θ低下として近似
中期実装：dopamine → EFP/Fの重み変換（引力への符号反転）へ分離
```

## Julia実装

```julia
Base.@kwdef struct NeuroState
    serotonin::Float64  = 1.0   # 誤差耐性・安定性
    norad::Float64      = 0.0   # 危機検出・即応性
    dopamine::Float64   = 0.0   # 探索性・誤差解決引力
    oxytocin::Float64   = 0.0   # 自他境界変調（中期実装）
end

"""
    neuro_mod(H, theta, neuro, params)

神経状態によるθ変調。

セロトニン高 → θ上昇（安定・跳躍しにくい）
ノルアドレナリン高 → θ低下（即応・緊急跳躍しやすい）
ドーパミン高 → θ低下（探索・誤差解決への引力）
"""
function neuro_mod(H, theta, neuro::NeuroState, params::RDFParams)
    theta_mod = theta
    theta_mod *= params.serotonin_theta_gain * neuro.serotonin
    theta_mod -= params.norad_theta_gain    * neuro.norad
    theta_mod -= params.dopamine_theta_gain * neuro.dopamine
    return clamp(theta_mod, params.theta_min, params.theta_max)
end
```

---

# 統合ステップ

## Julia実装

```julia
"""
    rdf_step(EFP_new, F_prev, M, H, theta, params, neuro; rng)

RDFエンジンの1ステップ。

入力：
  EFP_new  今ステップのEFP（p5.jsが環境から構成して渡す）
  F_prev   前ステップの作用解釈F（慣性投影の基点として保持）
  M        現在の整合慣性
  H        現在の熱
  theta    基底跳躍閾値
  params   パラメータ群
  neuro    神経状態

出力：
  F_new       作用解釈F（p5.jsが位置・速度等の代わりに使う）
  M_new       更新後の整合慣性
  H_new       更新後の熱
  E           慣性誤差
  ξ           揺らぎ
  leap_value  跳躍条件との差分（正なら跳躍トリガー）
"""
function rdf_step(EFP_new, F_prev, M, H, theta,
                  params::RDFParams, neuro::NeuroState;
                  rng=Random.default_rng())

    # ① 計算核：EFP解釈・慣性誤差・熱・揺らぎ
    F_new, H_new, E, ξ = core_step(EFP_new, F_prev, M, H, params; rng=rng)

    # ② 神経力学：θ変調
    theta_mod = neuro_mod(H_new, theta, neuro, params)

    # ③ 構造更新：dM/dt = f(M, E, ξ)
    M_new = update_M(M, F_new, E, H_new, ξ, params)

    # 跳躍判定値（判断はp5.js側）
    leap_value = H_new - theta_mod

    return F_new, M_new, H_new, E, ξ, leap_value
end
```

---

# AgentProfile（上位設計）

低次パラメータを直接触る代わりに上位ラベルから展開する。

```julia
Base.@kwdef struct AgentProfile
    speed::Float64       = 0.5  # 更新速度
    stability::Float64   = 0.5  # 安定性（セロトニン・高θ・低eta）
    sensitivity::Float64 = 0.5  # 感度（揺らぎ幅・可塑性幅）
    curiosity::Float64   = 0.5  # 探索性（ドーパミン・低θ・高K）
    caution::Float64     = 0.5  # 警戒（ノルアドレナリン・高θ）
    bonding::Float64     = 0.5  # 結合（オキシトシン・中期実装）
end

function make_profile(p::AgentProfile)
    params = RDFParams(
        dt      = 0.005 + 0.02 * p.speed,
        alpha   = 0.005 + 0.02 * p.stability,
        K       = 0.5   + 1.5  * p.curiosity,
        eta     = 0.005 + 0.02 * (1 - p.stability),
        H_width = 0.3   + 0.5  * p.sensitivity,
    )
    neuro = NeuroState(
        serotonin = 0.5 + p.stability,
        norad     = p.caution,
        dopamine  = p.curiosity,
        oxytocin  = p.bonding,
    )
    theta = 0.5 + 2.0 * p.stability
    return params, neuro, theta
end
```

---

# p5.js側の役割

```javascript
// 初期化
let F_prev = new Array(dim).fill(0.0);

// メインループ
function update() {
    // EFP構成：環境からの未解釈作用束をp5.js側で組み立てる
    const EFP = buildEFP({
        food_dist: distToFood,
        predator_dist: distToPredator,
        terrain: localTerrain,
        // ...
    });

    // Julia呼び出し
    const result = rdfStep(EFP, F_prev, M, H, theta, params, neuro);

    // 状態更新
    F_prev = result.F_new;   // 次ステップの慣性投影基点として保持
    M      = result.M_new;
    H      = result.H_new;

    // 跳躍判定（意味付けはp5.js側）
    if (result.leap_value > 0) {
        leap();
    }

    // F_newを位置・速度・行動の基点として使う
    applyF(result.F_new);
}
```

p5.js側が担当するもの：

```text
EFP構成（環境作用束の組み立て）
F_prevの保持（慣性投影の基点）
跳躍判定・演出
行動制御（食べる・逃げる・死ぬ）
描画
意味付け
```

---

# 設計上の重要ルール

## 1. Julia は判断しない

```text
NG: if H > theta then leap!
OK: leap_value = H - theta_mod を返す
```

## 2. Julia は意味語を知らない

```text
NG: eat, die, escape, chase
OK: EFP, F, M, H, E, ξ
```

## 3. SはFに変わった

```text
旧：S が主役・位置・速度・状態
新：F_new が主役・MがEFPを解釈した結果
    p5.js が F_new を位置・速度等に変換して使う
```

## 4. 慣性投影がEの定義

```text
E = F_new - M·F_prev

M·F_prev = 慣性投影（Mが前回のFに乗ったまま流れた値）
F_new    = 実際に来たF
E        = その差 = 慣性誤差
```

## 5. NAMとBase Modelを分ける

```text
NAM（対角近似）：安定領域の低負荷近似。短期実装。
Base Model　　：行列版M・H_vec・散逸行列A。中期以降。
```

---

# 今後の拡張

## 短期

```text
- rdf_step の最小実装・動作確認
- 対角近似版で生態系シミュレーション
- p5.js ↔ Julia 通信層の整備
- F_prevの保持とEFP構成の設計
```

## 中期

```text
- MをFullM（V, Λ）へ拡張
- HをH_vecへ拡張（Mの未拘束自由度の振動δM(t)として）
- 散逸行列Aの導入（Mの固有構造に対応した方向別散逸）
- dopamineをEFP/Fの重み変換へ分離
- oxytocin → EFPの自他境界変調
```

```julia
# 中期の型階層案
abstract type IntegrationInertia end

struct DiagonalM <: IntegrationInertia
    diag::Vector{Float64}
end

struct FullM <: IntegrationInertia
    V::Matrix{Float64}
    Λ::Vector{Float64}
end

# 多重ディスパッチでcore_stepを切り替える
function interp(M::DiagonalM, EFP)
    return M.diag .* EFP
end

function interp(M::FullM, EFP)
    return M.V * (M.Λ .* (M.V' * EFP))
end
```

## 長期

```text
- LLMを外部EFPソースとして接続
- RDFエージェントの自己学習
- 多エージェント間のSILNネットワーク
- Julia側のライブラリ化
```

---

# 一文圧縮

RDF Julia エンジンは、EFPをMで解釈しFを生成し、慣性投影との誤差EでMを更新する計算カーネルとして設計する。意味付けと判断はすべてp5.js側に委ねる。
