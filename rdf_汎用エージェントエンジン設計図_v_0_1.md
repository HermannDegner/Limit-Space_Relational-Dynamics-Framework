# RDF 汎用エージェントエンジン設計図 v0.1

## 0. 目的

本設計図は、RDF（関係力学言語）の発想をもとに、
**移動・思考・行動を統一的に扱える汎用エージェントエンジン**を構築するための土台を定義する。

本エンジンは、個別タスク専用のAIではなく、
**関係性勾配の中を、整合慣性を持つ構造が流動する一般エージェント基盤**
として設計される。

したがって本体はできる限り汎用に保ち、
世界ごとの差異は主に以下へ押し込む。

- Sensor Plugin：世界を勾配へ翻訳する
- Action Plugin：内部状態を行動へ翻訳する
- NeuroDynamics：個体差・認知差・基層傾向を与える

---

## 1. 中核命題

本エンジンの最重要前提は以下である。

> 移動・思考・行動は別のものではなく、関係性勾配の中を構造が流れる現象の異なる表現である。

このため、物理空間・概念空間・仮想空間・社会空間を、
すべて「勾配場」として扱う。

エージェントは、その勾配場の中で

- 感知し
- 解釈し
- 変形し
- 統合し
- 流れ
- 必要なら跳躍する

存在として実装される。

---

## 2. 設計原則

### 2.1 Base Model は普遍層

Base Model は世界の意味を知らない。
扱うのは以下のみ。

- 状態 `S`
- 熱 `H`
- 整合慣性 `M`
- 合成勾配 `F_total`
- 揺らぎ `ξ`
- 跳躍

### 2.2 センサーは意味を持たない

Sensor は世界から来る未解釈の作用束を、勾配へ翻訳する。
意味付け・価値付け・恐怖・期待・想像は Neuro 側で行う。

### 2.3 Neuro は勾配変形器

NeuroDynamics は、観測された勾配をそのまま通さない。

- 概念化
- 想像生成
- ドーパミンによる重み変換
- 退屈による仮想熱生成
- 警戒による閾値変更

などを通して勾配を変形する。

### 2.4 Network は個体間勾配の結合層

Network は個体間相互作用を管理する。
ここでは、他者は「オブジェクト」ではなく、
自分へ作用する勾配源として扱われる。

### 2.5 入出力はプラグイン化する

エンジン本体は汎用に保つ。
そのため、世界ごとの差異は以下に分離する。

- `SensorPlugin`
- `ActionPlugin`
- 必要に応じて `WorldAdapter`

---

## 3. 全体アーキテクチャ

```text
[World / Environment]
    ↓
[Sensor Plugins]
    ↓
[Sensor Layer]      : 世界を勾配場へ翻訳
    ↓
[Neuro Layer]       : 勾配を変形・生成・再重み付け
    ↓
[Network Layer]     : 他個体勾配を結合・伝播
    ↓
[Base Model Layer]  : S, H, M, ξ, Jump を積分
    ↓
[Action Layer]      : 状態を行動へ変換
    ↓
[World / Environment 更新]
```

---

## 4. コア抽象

### 4.1 GradientField

エンジンの基本単位。
すべての入力・内部生成・社会影響は、この型へ落とす。

```python
@dataclass
class GradientField:
    value: Vec
    source: str
    weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_force(self) -> Vec:
        return self.weight * self.value
```

#### 意味

- `value` : 潜在空間上の勾配ベクトル
- `source` : 由来（visual_food, danger_sound, imagination_reward など）
- `weight` : 重み
- `meta` : 補助情報

これにより、視覚・音・想像・社会圧・内部欲求を同一の形式で扱える。

---

### 4.2 潜在空間

すべての勾配は最終的に共通の潜在空間 `R^n` へ射影される。

- 視覚勾配
- 音勾配
- 匂い勾配
- 概念勾配
- 想像勾配
- 社会勾配
- 内部欲求勾配

はすべて同じ次元のベクトルへ変換され、最終的に足し合わされる。

---

## 5. 状態モデル

### 5.1 BaseState

```python
@dataclass
class BaseState:
    S: Vec
    H: float
    M: Mat
    flow: Vec
    t: float = 0.0
```

#### 各項目

- `S` : 構造状態
- `H` : 蓄積熱
- `M` : 整合慣性行列
- `flow` : 現在の主流方向
- `t` : 時刻

---

### 5.2 BaseParams

```python
@dataclass
class BaseParams:
    alpha: float
    theta: float
    K: float
```

- `alpha` : 熱散逸率
- `theta` : 跳躍閾値
- `K` : 関係総量近似値

---

### 5.3 Modulation

Neuro から Base へ渡される変調量。

```python
@dataclass
class Modulation:
    alpha_scale: float = 1.0
    theta_scale: float = 1.0
    K_scale: float = 1.0
    M_delta: Mat | None = None
    force_gain: float = 1.0
    social_gain: float = 1.0
```

---

### 5.4 NeuroState

```python
@dataclass
class NeuroState:
    dopamine: float = 0.0
    boredom: float = 0.0
    serotonin: float = 0.0
    noradrenaline: float = 0.0
    prediction: Vec | None = None
    memory: dict[str, Any] = field(default_factory=dict)
```

#### 解釈

- `dopamine` : 未知や誤差解決への引力変換傾向
- `boredom` : フラット勾配への仮想熱
- `serotonin` : 誤差許容・散逸安定化
- `noradrenaline` : 警戒・非常時モード切替

---

## 6. レイヤー定義

## 6.1 Sensor Layer

### 役割

世界の状態を観測し、未解釈の作用束を勾配束へ変換する。

### 入力

- `world_state`
- `agent_state`

### 出力

```python
@dataclass
class SensorOutput:
    gradients: list[GradientField]
    raw: dict[str, Any] = field(default_factory=dict)

    def fused(self) -> Vec:
        total = np.zeros_like(self.gradients[0].value)
        for g in self.gradients:
            total += g.as_force()
        return total
```

### インターフェース

```python
class SensorSuite:
    def sense(self, world_state, agent_state) -> SensorOutput:
        ...
```

### 例

- VisualSensor : 食料・障害物・敵の方向勾配
- AudioSensor : 音源・危険・呼びかけの勾配
- OlfactorySensor : 匂い源の勾配
- TactileSensor : 接触・衝突回避勾配
- TextSensor : テキスト文脈勾配
- APISensor : 数値列や外部指標の勾配

---

## 6.2 Neuro Layer

### 役割

Sensor から得た勾配を、個体の基層・中核・記憶・予測に応じて変形する。

### 主機能

- 概念化
- 想像勾配生成
- 内部欲求勾配生成
- ドーパミンによる未知勾配の符号変換
- 退屈による探索勾配生成
- 警戒による閾値変更

### 出力

```python
@dataclass
class NeuroOutput:
    gradients: list[GradientField]
    modulation: Modulation
    public_signal: Vec
    diagnostics: dict[str, float] = field(default_factory=dict)

    def fused(self) -> Vec:
        if not self.gradients:
            return np.zeros_like(self.public_signal)
        total = np.zeros_like(self.gradients[0].value)
        for g in self.gradients:
            total += g.as_force()
        return total
```

### インターフェース

```python
class NeuroDynamics:
    def step(self, sensor_out: SensorOutput, base_state: BaseState, dt: float) -> NeuroOutput:
        ...
```

### Neuro の典型処理

#### 1. 概念化
観測勾配を記憶やラベルへ接続する。

#### 2. 想像勾配
未来予測・仮説・恐怖・計画・夢などを内部生成する。

#### 3. ドーパミン処理
通常は反発となる未知や高コスト対象を、
誤差解決期待によって引力へ変換しうる。

#### 4. 退屈処理
勾配が平坦すぎると仮想熱を発生させ、探索勾配を生成する。

#### 5. ノルアドレナリン処理
危険兆候に対して、閾値や重みを生存優先へ切り替える。

---

## 6.3 Network Layer

### 役割

複数エージェント間の結合を管理し、個体間勾配を合成する。

### 核心

個体は孤立しない。
各エージェントの発する外向き信号が、
他個体にとっての勾配源となる。

### 状態

- `W_ij` : 結合行列

### 出力

```python
@dataclass
class SocialOutput:
    gradients: list[list[GradientField]]
    W: Mat
```

### インターフェース

```python
class Network:
    def __init__(self, W: Mat):
        self.W = W

    def step(self, public_signals: list[Vec], dt: float) -> SocialOutput:
        ...
```

### 例

- 接近勾配
- 追従勾配
- 同調勾配
- 警戒回避勾配
- 縄張り反発
- 愛着引力

### 学習則

将来的には以下を分離実装する。

```python
class CouplingRule:
    def update(self, W: Mat, public_signals: list[Vec], jumped: list[bool], dt: float) -> Mat:
        ...
```

---

## 6.4 Base Model Layer

### 役割

合成勾配に基づいて状態を積分し、熱蓄積・揺らぎ・跳躍を扱う。

### 状態方程式（概念）

```text
dS/dt = -MS + F_total + ξ
E      = F_total - MS
σ      = ||E||²
dH/dt  = σ - αH
jump   if H > θ + noise
```

### インターフェース

```python
class BaseModel:
    def __init__(self, state: BaseState, params: BaseParams):
        self.state = state
        self.params = params

    def step(self, F_total: Vec, modulation: Modulation, dt: float) -> BaseOutput:
        ...
```

### BaseOutput

```python
@dataclass
class BaseOutput:
    state: BaseState
    E: Vec
    sigma: float
    xi: Vec
    jumped: bool
    diagnostics: dict[str, float] = field(default_factory=dict)
```

### 補助構造

```python
class NoiseModel:
    def sample(self, M: Mat, K: float, shape: tuple[int, ...]) -> Vec:
        ...

class JumpPolicy:
    def apply(self, S: Vec, M: Mat, H: float, F_total: Vec, xi: Vec) -> tuple[Vec, Mat, float]:
        ...
```

---

## 6.5 Action Layer

### 役割

Base 状態と flow を、実際の行動へ翻訳する。

### インターフェース

```python
class ActionPlugin:
    def act(self, base_state: BaseState, neuro_state: NeuroState, world_state) -> Any:
        ...
```

### 例

- MoveAction : 位置更新ベクトルへ変換
- SpeechAction : 発話選択へ変換
- ToolAction : ツール使用命令へ変換
- CursorAction : UI操作へ変換
- APIAction : 外部システムへの送信へ変換

---

## 7. Agent 単位

```python
class Agent:
    def __init__(
        self,
        agent_id: int,
        base: BaseModel,
        sensors: SensorSuite,
        neuro: NeuroDynamics,
        action: ActionPlugin,
    ):
        self.agent_id = agent_id
        self.base = base
        self.sensors = sensors
        self.neuro = neuro
        self.action = action
```

### 役割

Agent は一体分の束ね役。
実際の計算本体は各レイヤーへ委譲する。

---

## 8. 標準更新ループ

```python
def simulation_step(world, agents, network, dt):
    sensor_outs = []
    neuro_outs = []

    # 1. sensing
    for agent in agents:
        s_out = agent.sensors.sense(world, agent.base.state)
        sensor_outs.append(s_out)

    # 2. neuro transform
    for agent, s_out in zip(agents, sensor_outs):
        n_out = agent.neuro.step(s_out, agent.base.state, dt)
        neuro_outs.append(n_out)

    # 3. social coupling
    public_signals = [n.public_signal for n in neuro_outs]
    social_out = network.step(public_signals, dt)

    # 4. base integration
    base_outs = []
    for i, agent in enumerate(agents):
        social_force = np.zeros_like(agent.base.state.S)
        if social_out.gradients[i]:
            for g in social_out.gradients[i]:
                social_force += g.as_force()

        F_total = (
            sensor_outs[i].fused()
            + neuro_outs[i].fused()
            + neuro_outs[i].modulation.social_gain * social_force
        )

        b_out = agent.base.step(F_total, neuro_outs[i].modulation, dt)
        base_outs.append(b_out)

    # 5. actions
    actions = []
    for agent in agents:
        action = agent.action.act(agent.base.state, agent.neuro.state, world)
        actions.append(action)

    # 6. world update
    world = world.apply(actions, dt)

    return world, base_outs, actions
```

---

## 9. 汎用化の本質

本エンジンが汎用である理由は、
世界ごとの差異を中核へ埋め込まないからである。

### 共通部分

- 勾配を受ける
- 勾配を変形する
- 勾配を合成する
- 状態を更新する
- 跳躍する

### 差し替え部分

- 何を感知するか
- 何を価値とみなすか
- 何を行動として出すか
- 他者とどう結合するか

つまりこれは、
**勾配で動く構造一般のためのエンジン**
である。

---

## 10. 適用可能な対象

### 10.1 動物・NPC

- 視覚
- 嗅覚
- 音
- 接触
- 群れ行動

### 10.2 会話エージェント

- テキスト入力
- 文脈勾配
- 概念空間遷移
- 発話行動

### 10.3 ロボット

- カメラ
- LiDAR
- IMU
- 接触
- モーター出力

### 10.4 業務監視エージェント

- API値
- ログ
- 異常勾配
- アラート行動

### 10.5 創作・思考エージェント

- テキスト
- 内部想像勾配
- 長期計画勾配
- 文章生成

---

## 11. v0.1 実装方針

最初は以下に限定する。

### 11.1 Base

- `S, H, M`
- 線形更新
- 熱蓄積
- 単純跳躍

### 11.2 Sensor

- VisualSensor
- AudioSensor
- TextSensor のどれか1つから始める

### 11.3 Neuro

- ドーパミン
- 退屈
- 簡易想像勾配

### 11.4 Network

- 固定 `W`
- 単純な引力/反発のみ

### 11.5 Action

- 2D移動 or テキスト出力

---

## 12. 今後の拡張候補

### 12.1 JumpPolicy の高度化

- 微小跳躍
- 探索跳躍
- 崩壊跳躍
- 社会的跳躍

### 12.2 Memory Layer の追加

- episodic memory
- semantic memory
- habit / bias

### 12.3 Multi-scale M

- 基層 M
- 中核 M
- 上層 M

### 12.4 World Model の追加

- 予測世界
- 他者モデル
- 自己モデル

### 12.5 学習則

- W 更新
- M 更新
- 勾配 source の圧縮
- ラベル形成

---

## 13. 一文定義

> RDF 汎用エージェントエンジンとは、関係性勾配の中を、整合慣性と熱と揺らぎを持つ構造が流動・再編する過程を、レイヤー分離とプラグイン構造によって実装するための基盤である。

---

## 14. 超圧縮

- 世界は勾配として入る
- 個体は勾配を歪める
- 社会は勾配を共有する
- Base は勾配の中で流れる
- 行動はその流れの出力である

---

## 15. 最終まとめ

この設計において、エージェントとは「状態機械」ではない。

**多層勾配場の中を流れる構造**である。

そのため、入力を変えれば動物にもNPCにも会話AIにもなりうる。
違うのは魂ではなく、

- どの勾配を感知するか
- どの勾配を強く感じるか
- どの勾配を想像するか
- その流れを何の行動へ変換するか

という実装差である。

中核は共通でよい。

