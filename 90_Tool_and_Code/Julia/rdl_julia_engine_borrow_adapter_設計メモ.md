# RDL Julia Engine Borrow Adapter 設計メモ

## 位置づけ

この文書は、RDL Julia エンジンを現行 RDL Core に追従させつつ、外部理論・既存実装を借用して拡張するための設計メモである。

基本方針は以下。

```text
Core は最小式だけを持つ
借用品は Adapter として差し替える
```

RDL の中核は `EFP_est / F / E / H / ξ / B / M_B` を扱う軽量カーネルとし、PID・カルマンフィルタ・状態空間モデル・NN・StateGraph 的モード遷移などは、用途に応じて外部 Adapter として接続する。

---

## 1. 現行RDLへの追従ポイント

### 1.1 `M` ではなく `M_B`

旧設計では `M` としていたが、現行 RDL では独立した `Mそのもの` は存在しない。

```text
M_B = 境界 B における整合慣性
```

したがって Julia 側の構造体・関数名も可能な範囲で `M_B` に寄せる。

例：

```julia
struct RDLState
    F_prev::Vector{Float64}
    M_B::AbstractMatrix{Float64}
    H::Float64
    phase::Symbol
end
```

---

### 1.2 `B` を明示する

`M_B` は境界 `B` とセットでしか定義できない。

そのため、計算状態とは別に `RDLContext` を持つ。

```julia
Base.@kwdef struct RDLContext
    spatial_scale::Symbol = :individual
    temporal_scale::Symbol = :step
    evaluation_axis::Symbol = :survival
    purpose::Symbol = :simulation
    observer_id::Symbol = :default
end
```

この `B` は厳密な実体ではなく、計算上の境界設定ラベルである。

---

### 1.3 `EFP` は直接観測ではなく `EFP_est`

現行 RDL では EFP は直接観測できない。
観測者が持つのは `F` であり、EFP は操作上の推定概念である。

実装上は、外部入力を次のように扱う。

```text
raw_sensor_flux  : p5.js / World 側から来る未整形入力
EFP_est          : センサー入力から構成された推定素流圧
F                : M_B が EFP_est を解釈した作用解釈
```

---

## 2. 最小Core

Julia Core が直接扱うものは以下に絞る。

```text
B      : 境界
M_B    : 境界Bにおける整合慣性
EFP_est: 推定素流圧
F      : 作用解釈
E      : 慣性誤差
H      : 熱・誤差蓄積
ξ      : 揺らぎ
phase  : M_act / M_lat / M_Δ / M_B'
κ      : 自己修正可能性
```

基本ダイナミクス：

```text
F(t)    = interp(M_B, EFP_est)
E(t)    = F(t+Δ) - M_B · F(t)
dM_B/dt = f(M_B, E, ξ)
H       = accumulate(E)
```

跳躍条件：

```text
H > θ_eff + ξ_floor
```

相状態：

```text
M_act  : 通常稼働中
M_lat  : 潜在・待機中
M_Δ    : 再編中
M_B'   : 再編後の安定状態
```

---

## 3. 借用Adapter一覧

### 3.1 StateSpaceAdapter

状態空間モデルを借用する。

```text
x(t+1) = A x(t) + B u(t)
```

RDL対応：

```text
x       → F の多変数表現
A       → M_B の行列表現
u       → 外部制御入力 / 作用介入
外乱    → ξ
```

用途：

- `M_B` の行列化
- 物理・生態・NPC状態の線形近似
- FullM 実装への移行

実装候補：

```julia
abstract type AbstractMModel end

struct DiagonalM <: AbstractMModel
    weights::Vector{Float64}
end

struct FullM <: AbstractMModel
    matrix::Matrix{Float64}
end

struct StateSpaceM <: AbstractMModel
    A::Matrix{Float64}
    B::Matrix{Float64}
end
```

---

### 3.2 PIDAdapter

PID制御を借用する。

RDL対応：

```text
P項 → 現在の誤差 E
I項 → 累積熱 H
D項 → 誤差変化率 dE/dt
```

用途：

- 空腹・疲労・危険などの行動制御
- 目標状態への収束
- H の暴走抑制

NPC例：

```text
空腹Eが大きい       → 食物探索を強める
疲労Hが溜まる       → 休息行動へ寄せる
危険EのdE/dtが大きい → 即時逃走へ切替
```

PID は Core ではなく、行動制御層に置く。

---

### 3.3 KalmanAdapter

カルマンフィルタを借用する。

RDL対応：

```text
予測値       → M_B · F(t)
観測値       → ξで汚れたF
更新後推定値 → 補正されたF
```

用途：

- センサー不確実性の処理
- 遮蔽物・見失い・音だけの検知
- NPCの対象追跡
- EFP_est の安定化

例：

```text
見えた餌
前回いた餌
音だけした敵
匂いだけ残る対象
```

これらを統合して、現在の `F` を推定する。

---

### 3.4 ModeGraphAdapter

StateGraph / ハイブリッドオートマトン的な状態遷移を借用する。

RDL対応：

```text
State             → M_B^ij ネットワークの現在断面
Node              → M_act の局所断面
Conditional Edge  → 跳躍条件
State Transition  → M_act / M_lat / M_Δ / M_B'
```

用途：

- 通常行動
- 探索行動
- 逃走行動
- 休眠行動
- 群れ行動
- 繁殖行動
- 異常時再編

設計例：

```text
act_node
  ↓ H <= θ_eff
continue
  ↓ H > θ_eff + ξ
reorg_node / M_Δ
  ↓ 再編成功
post_jump / M_B'
```

---

### 3.5 NNAdapter

ニューラルネットワークを借用する。

RDL対応：

```text
重み行列 W      → M_B^ij
順伝播          → interp(M_B, EFP_est)
損失            → E
誤差逆伝播      → dM_B/dt
ドロップアウト  → ξ注入
正則化          → M_B過剰固定の抑制
```

用途：

- `interp()` の非線形化
- `update_M_B()` の学習化
- センサー入力からFへの複雑な写像
- 個体差の学習

導入順序：

```text
Step 1: DiagonalM
Step 2: FullM
Step 3: NN_interp
Step 4: NN_update_M_B
```

最初からNN化しない。
まずは軽量Coreを動かしてから、必要部分だけNNに置換する。

---

### 3.6 BoidsAdapter

群知能・Boidsを借用する。

RDL対応：

```text
分離      → 近すぎる他個体への反発EFP
整列      → 周囲のM_Bとの同期
結合      → 群れ中心への引力F
```

用途：

- 群れ
- 鳥・魚・虫
- NPC集団の自然な移動
- 社会的同期

これは個体Coreではなく、集団層に置く。

---

### 3.7 ReactionDiffusionAdapter

反応拡散系を借用する。

用途：

- 匂い
- 栄養
- 毒
- 熱
- 魔力
- 汚染
- 感染

RDL対応：

```text
場の濃度勾配 → EFP_est
拡散          → ξを含む作用伝播
反応          → M_B間相互作用
```

生態系・仮想世界ではかなり重要。

---

### 3.8 EvolutionAdapter

進化アルゴリズムを借用する。

RDL対応：

```text
個体M_B       → 遺伝・学習された整合慣性
突然変異      → ξ注入
淘汰          → 生存バイアス
世代交代      → M_Bの残存・再編
```

用途：

- 生態系
- 世代交代
- 行動パターンの残存
- NPC種族差

---

### 3.9 GraphAdapter

グラフ理論を借用する。

用途：

- SILNネットワーク
- M_B接続
- 社会構造
- 生態系食物網
- 概念ネットワーク

RDL対応：

```text
Node   → SILN / M_B
Edge   → C_ij / W_ij
Weight → 接続強度
Flow   → EFP / F の伝播
```

---

## 4. 推奨実装順序

### Phase 0：最小確認

```text
DiagonalM
EFP_est → F
E → H
ξ生成
leap_value
```

目的：RDL式が動くことを確認する。

---

### Phase 1：実用的な個体制御

```text
PIDAdapter
KalmanAdapter
StateSpaceAdapter
```

目的：NPC個体が、曖昧な観測を処理しながら目的行動できるようにする。

---

### Phase 2：相状態遷移

```text
ModeGraphAdapter
M_act / M_lat / M_Δ / M_B'
κ
```

目的：通常行動・逃走・探索・休眠・再編を切り替える。

---

### Phase 3：環境場

```text
ReactionDiffusionAdapter
CellularAutomataAdapter
```

目的：匂い・栄養・毒・熱・植生などの環境場を作る。

---

### Phase 4：集団・生態系

```text
BoidsAdapter
GraphAdapter
EvolutionAdapter
```

目的：群れ、食物網、世代交代、社会構造を扱う。

---

### Phase 5：学習化

```text
NNAdapter
```

目的：interp と M_B 更新の一部を学習可能にする。

---

## 5. 推奨アーキテクチャ

```text
[p5.js / World]
  - 描画
  - センサー入力
  - 地形・環境場
  - 行動実行
  - 意味付け補助

        ↓ raw_sensor_flux

[Julia RDL Core]
  - B
  - M_B
  - EFP_est
  - F
  - E
  - H
  - ξ
  - κ
  - phase

        ↓ / ↑

[Borrow Adapters]
  - StateSpaceAdapter
  - PIDAdapter
  - KalmanAdapter
  - ModeGraphAdapter
  - NNAdapter
  - BoidsAdapter
  - ReactionDiffusionAdapter
  - EvolutionAdapter
  - GraphAdapter
```

---

## 6. 最初に入れるべき3つ

最初の実装では、以下の3つに絞る。

```text
1. StateSpaceAdapter
2. PIDAdapter
3. KalmanAdapter
```

これで最低限、以下が実現できる。

```text
M_Bの行列化
E/H制御
ξ込み観測
```

その後に、ModeGraph、Boids、ReactionDiffusion、NNを順番に足す。

---

## 7. 結論

RDL Julia Engine は、借用前提で設計した方がよい。

ただし、Coreに全部を入れない。

```text
Core = RDL最小ダイナミクス
Adapter = 外部理論の借用
World = 実行環境・描画・意味付け
```

この分離を守れば、軽量実装から仮想世界シミュレーター、生態系NPC、Minecraft連携まで拡張しやすい。

次の設計名候補：

```text
RDL Julia Engine v0.3 Borrow Adapter版
```

