# RDL 概念的数理モデル v0.2 Python版

---

## ■ 概要

本モデルは、関係力学言語（RDL）における
「整合」「整合慣性」「素流圧」「跳躍」を
散逸構造論を基盤とした数理構造として記述するものである。

この文書では、元の数式表現を Python コードで扱える形に置き換える。
ただし、微分方程式はそのままでは実行しにくいため、**離散時間近似**として表現する。

**設計方針**

```python
# Base Model（物理層）
#   普遍・固定・スケール不問
#   非平衡熱力学・散逸構造の基礎式
#
#        ↓ インターフェース（三接続点）
#
# Fluctuation MOD（揺らぎ層）
#   個体・状況・スケール依存
#   Base Modelには触れず係数を渡すのみ
```

本モデルは真理の記述ではなく、

> **整合の動態を操作可能な形で表現するための近似モデル**

として位置付けられる。

---

## ■ Python実装の前提

```python
import numpy as np


def norm_sq(vector: np.ndarray) -> float:
    return float(np.dot(vector, vector))
```

- ベクトル `S`, `F` は `numpy.ndarray`
- 行列 `M`, `V`, `Lambda` は `numpy.ndarray`
- `dS/dt` や `dH/dt` は、時間刻み `dt` を使って更新する

---

## ■ Base Model（物理層）

### ● 状態（構造）

```python
S = np.zeros(n)
```

ある時点における認識・内部状態の表現。

---

### ● 整合慣性行列

```python
M = np.zeros((n, n))
```

固有値分解：

```python
eigenvalues, V = np.linalg.eig(M)
Lambda = np.diag(eigenvalues)
M_reconstructed = V @ Lambda @ np.linalg.inv(V)

# V: 固有ベクトル（安定な整合の方向）
# Lambda: 各方向の整合慣性の強さを表す対角行列
```

---

### ● 状態方程式

```python
def dS_dt(S: np.ndarray, M: np.ndarray, F: np.ndarray) -> np.ndarray:
    return -M @ S + F


def update_S(S: np.ndarray, M: np.ndarray, F: np.ndarray, dt: float) -> np.ndarray:
    return S + dS_dt(S, M, F) * dt
```

- `M @ S`：現構造による予測・維持
- `F`：素流圧（外部および内部からの作用）

---

### ● 誤差

```python
def error(S: np.ndarray, M: np.ndarray, F: np.ndarray) -> np.ndarray:
    return F - M @ S
```

現構造で処理できなかった残差。熱源となる。

---

### ● 熱蓄積と散逸

```python
def sigma(S: np.ndarray, M: np.ndarray, F: np.ndarray) -> float:
    E = error(S, M, F)
    return norm_sq(E)


def dH_dt(H: float, S: np.ndarray, M: np.ndarray, F: np.ndarray, alpha: float) -> float:
    return sigma(S, M, F) - alpha * H


def update_H(H: float, S: np.ndarray, M: np.ndarray, F: np.ndarray, alpha: float, dt: float) -> float:
    return H + dH_dt(H, S, M, F, alpha) * dt
```

- `sigma`：誤差から生じる熱生成
- `alpha`：散逸係数（系が熱を逃がす速度）

**動態の解釈**

```python
current_sigma = sigma(S, M, F)

if current_sigma > alpha * H:
    state = "heat_accumulates"
elif current_sigma < alpha * H:
    state = "heat_decays"
else:
    state = "balanced"
```

---

### ● 跳躍条件と跳躍後

```python
def should_jump(H: float, theta: float) -> bool:
    return H > theta


def jump_state(S: np.ndarray, M: np.ndarray, H: float, rebuild_M, transition_S):
    M_next = rebuild_M(M)
    S_next = transition_S(S, M_next)
    H_next = 0.0
    return S_next, M_next, H_next
```

跳躍後の `M_next` は素流圧環境に依存して決まる。
予測不能だが、より高い `sigma` を散逸できる構造へ向かう傾向を持つ。

---

### ● Base Model 全体

```python
def step_base_model(
    S: np.ndarray,
    M: np.ndarray,
    H: float,
    F: np.ndarray,
    alpha: float,
    theta: float,
    dt: float,
    rebuild_M,
    transition_S,
):
    S_next = update_S(S, M, F, dt)
    H_next = update_H(H, S, M, F, alpha, dt)

    if should_jump(H_next, theta):
        S_next, M, H_next = jump_state(S_next, M, H_next, rebuild_M, transition_S)

    return S_next, M, H_next
```

---

## ■ 次元削減（局所近似）

完全版 `M` は高次元だが、文脈ごとに支配的な固有値は少数に限られる。

```python
eigenvalues, V = np.linalg.eig(M)
sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]

k = context_dependent_k
top_indices = sorted_indices[:k]

Lambda_k = np.diag(eigenvalues[top_indices])
V_k = V[:, top_indices]
```

低次元近似：

```python
def project_state(S: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    return V_k.T @ S


def reconstruct_state(a: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    return V_k @ a


def da_dt(a: np.ndarray, Lambda_k: np.ndarray, V_k: np.ndarray, F: np.ndarray) -> np.ndarray:
    return -Lambda_k @ a + V_k.T @ F


def update_a(a: np.ndarray, Lambda_k: np.ndarray, V_k: np.ndarray, F: np.ndarray, dt: float) -> np.ndarray:
    return a + da_dt(a, Lambda_k, V_k, F) * dt
```

`Lambda_k` が対角になることで計算が大幅に軽くなる。
これは SILN における「局所線形近似」に対応する。

次元数 `k` は文脈依存であり、Fluctuation MOD 側から渡される。

---

## ■ Fluctuation MOD 接続点

Base Model への介入は以下の三点のみ。
Base Model の構造自体には触れない。

```python
def g_alpha(mod_params) -> float:
    return mod_params.get("alpha_gain", 1.0)


def g_theta(mod_params) -> float:
    return mod_params.get("theta_gain", 1.0)


def G_Lambda(mod_params, size: int) -> np.ndarray:
    gain = mod_params.get("lambda_gain", 1.0)
    if np.isscalar(gain):
        return np.eye(size) * gain
    return np.asarray(gain)


alpha_eff = alpha * g_alpha(mod_params)
theta_eff = theta * g_theta(mod_params)
Lambda_eff = Lambda @ G_Lambda(mod_params, Lambda.shape[0])
```

- `M` または `Lambda_k` への注入
  整合慣性の方向と強度を変える
- `alpha` への注入
  散逸速度を変える
- `theta` への注入
  跳躍閾値を動的に変える

---

## ■ MOD 実装例（神経物質 MOD）

```python
def dopamine_filter(F: np.ndarray, gain: float) -> np.ndarray:
    return F * gain


def serotonin_alpha(alpha: float, serotonin_level: float) -> float:
    return alpha * (1.0 + serotonin_level)


def noradrenaline_theta(theta: float, danger_signal: float) -> float:
    return theta * max(0.1, 1.0 - danger_signal)


def survival_lambda_boost(Lambda: np.ndarray, boost_index: int, gain: float) -> np.ndarray:
    Lambda_next = Lambda.copy()
    Lambda_next[boost_index, boost_index] *= gain
    return Lambda_next


def oxytocin_expand_M(M: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    return M + coupling
```

---

## ■ スケール横断性

Base Model はスケールを問わず同一のコード構造で動く。
Fluctuation MOD の設定のみが異なる。

```python
scale_configs = {
    "universe": {"theta": 1000.0, "alpha": 0.1},
    "life": {"theta": 100.0, "alpha": 0.2},
    "society": {"theta": 50.0, "alpha": 0.4},
    "person": {"theta": 10.0, "alpha": 0.8},
    "cognition": {"theta": 5.0, "alpha": 1.0},
}
```

| スケール | 蓄積 H | 跳躍 | 生成 |
|---|---|---|---|
| 宇宙 | 核融合限界 | 超新星爆発 | 重元素・新星系 |
| 生命 | 遺伝的誤差 | 種の分岐 | 生物多様性 |
| 社会 | 不満・思想蓄積 | 革命 | 新制度・文化 |
| 個人 | 誤差熱蓄積 | 気づき・転換 | 新整合構造 |
| 認知 | 接続候補 | ミーム定着 | 概念の拡張 |

---

## ■ モデルの性質

- 非絶対性：前提は閉じない
- 安定性中心：存在ではなく安定を扱う
- 関係優位：構造は関係で定義される
- 拡張前提：MOD追加により個体・状況・スケールへ対応

---

## ■ 残課題

- `g_alpha`, `g_theta`, `G_Lambda` の具体的な関数形
- 跳躍後の `M_next` 生成メカニズムの記述
- 複数構造間のネットワーク拡張（エージェント間伝播）
- `F(t) · H_internal` に対応する操作的定義

---

## ■ 総括

```python
def rdl_step(S, M, H, F, alpha, theta, dt, mod_params, rebuild_M, transition_S):
    alpha_eff = alpha * g_alpha(mod_params)
    theta_eff = theta * g_theta(mod_params)

    S_next = update_S(S, M, F, dt)
    E = error(S, M, F)
    sigma_value = norm_sq(E)
    H_next = H + (sigma_value - alpha_eff * H) * dt

    if H_next > theta_eff:
        S_next, M, H_next = jump_state(S_next, M, H_next, rebuild_M, transition_S)

    return {
        "S": S_next,
        "M": M,
        "E": E,
        "sigma": sigma_value,
        "H": H_next,
        "jump": H_next == 0.0,
    }
```

この形なら、元の概念モデルを Python 上で段階的にシミュレーションできる。