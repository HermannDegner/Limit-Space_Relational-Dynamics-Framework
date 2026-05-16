"""
rdl_model.py

RDL (Relational Dynamics Framework) の軽量Python実装
- 構造 A
- 入力 x
- 予測との差 e
- 重み付き熱 H
- 微修正による学習
- 整合慣性 I
- 跳躍
- 感情出力

依存:
    pip install numpy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


Array = np.ndarray


# ============================================================
# パラメータ定義
# ============================================================

@dataclass
class HumanParams:
    """
    人間用アタッチ層の基層パラメータ
    d: ドーパミン的探索傾向
    s: セロトニン的誤差許容度
    o: オキシトシン的自己境界拡張度
    n: ノルアドレナリン的危機増幅度
    c_obs: 意識的観測・内省強度
    g: 関係的安全性・信頼基盤
    """
    d: float = 0.5
    s: float = 0.5
    o: float = 0.5
    n: float = 0.5
    c_obs: float = 0.5
    g: float = 0.5


@dataclass
class WeightConfig:
    """
    重み行列 W の各成分を作るための係数
    w_i = alpha_i + beta_i*n - gamma_i*s + delta_i*o*r_i
    """
    alpha: Array
    beta: Array
    gamma: Array
    delta: Array


@dataclass
class RDLConfig:
    """
    モデル全体のハイパーパラメータ
    """
    dim: int
    eta0: float = 0.03
    rho: float = 0.95
    kappa: float = 0.4
    lambda_d: float = 0.5
    mu_c: float = 0.5

    theta0: float = 2.5
    theta_a: float = 1.0
    theta_b: float = 1.0
    theta_c: float = 1.0

    leap_scale: float = 3.0
    leap_noise_scale: float = 0.01

    min_weight: float = 0.01


@dataclass
class EmotionState:
    anger: float = 0.0
    fear: float = 0.0
    care: float = 0.0
    curiosity: float = 0.0


@dataclass
class StepInfo:
    x_t: Array
    y_hat_t: Array
    y_t: Array
    e_t: Array
    w_t: Array
    H_t: float
    H_next: float
    eta_t: float
    theta_t: float
    jumped: bool
    inertia_before: float
    inertia_after: float
    emotions: EmotionState


# ============================================================
# 本体
# ============================================================

@dataclass
class RDLModel:
    """
    RDL軽量モデル

    A: 構造行列
    I: 整合慣性
    """
    config: RDLConfig
    weights: WeightConfig
    A: Array = field(init=False)
    I: float = field(default=0.0)

    def __post_init__(self) -> None:
        dim = self.config.dim
        self.A = np.eye(dim) + 0.05 * np.random.randn(dim, dim)

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        dim = self.config.dim
        for name, arr in {
            "alpha": self.weights.alpha,
            "beta": self.weights.beta,
            "gamma": self.weights.gamma,
            "delta": self.weights.delta,
        }.items():
            if arr.shape != (dim,):
                raise ValueError(f"{name}.shape must be ({dim},), got {arr.shape}")

    # --------------------------------------------------------
    # 基本計算
    # --------------------------------------------------------

    def predict_output(self, x_t: Array) -> Array:
        """
        y_t = A_t x_t
        """
        self._check_vec(x_t, "x_t")
        return self.A @ x_t

    def predict_expected(self, x_t: Array, expected_A: Optional[Array] = None) -> Array:
        """
        予測出力 y_hat_t
        外部から expected_A を入れなければ現在構造をそのまま使う。
        実務では別予測器に差し替えてもよい。
        """
        self._check_vec(x_t, "x_t")
        if expected_A is None:
            expected_A = self.A
        if expected_A.shape != self.A.shape:
            raise ValueError("expected_A shape mismatch")
        return expected_A @ x_t

    def error(self, y_t: Array, y_hat_t: Array) -> Array:
        """
        e_t = y_t - y_hat_t
        """
        self._check_vec(y_t, "y_t")
        self._check_vec(y_hat_t, "y_hat_t")
        return y_t - y_hat_t

    def build_weight_vector(self, params: HumanParams, r_t: Array) -> Array:
        """
        w_i = alpha_i + beta_i*n - gamma_i*s + delta_i*o*r_i
        """
        self._check_vec(r_t, "r_t")

        w = (
            self.weights.alpha
            + self.weights.beta * params.n
            - self.weights.gamma * params.s
            + self.weights.delta * params.o * r_t
        )
        return np.clip(w, self.config.min_weight, None)

    def heat(self, e_t: Array, w_t: Array) -> float:
        """
        H_t = || W_t e_t ||_2
        対角行列 W_t を要素積で表現
        """
        self._check_vec(e_t, "e_t")
        self._check_vec(w_t, "w_t")
        return float(np.linalg.norm(w_t * e_t, ord=2))

    def learning_rate(self, params: HumanParams) -> float:
        """
        eta_t = eta0 * (1 + lambda*d) * (1 + mu*c_obs)
        """
        return (
            self.config.eta0
            * (1.0 + self.config.lambda_d * params.d)
            * (1.0 + self.config.mu_c * params.c_obs)
        )

    def jump_threshold(self, params: HumanParams) -> float:
        """
        Theta_t = theta0 + a*s - b*n + c*g
        """
        return (
            self.config.theta0
            + self.config.theta_a * params.s
            - self.config.theta_b * params.n
            + self.config.theta_c * params.g
        )

    # --------------------------------------------------------
    # 更新
    # --------------------------------------------------------

    def small_update(self, x_t: Array, e_t: Array, w_t: Array, eta_t: float) -> Array:
        """
        微修正:
        A_{t+1} = A_t - eta_t * (W e) x^T
        """
        weighted_e = w_t * e_t
        delta_A = -eta_t * np.outer(weighted_e, x_t)
        return self.A + delta_A

    def leap_update(self, x_t: Array, e_t: Array, w_t: Array, eta_t: float) -> Array:
        """
        跳躍:
        微修正では吸収できない大きな熱に対する大規模更新
        """
        weighted_e = w_t * e_t
        delta_A = -self.config.leap_scale * eta_t * np.outer(weighted_e, x_t)
        noise = self.config.leap_noise_scale * np.random.randn(*self.A.shape)
        return self.A + delta_A + noise

    def update_inertia(self, H_t: float, H_next: float) -> float:
        """
        I_{t+1} = rho * I_t + kappa * max(0, H_t - H_next)
        """
        reduction = max(0.0, H_t - H_next)
        return self.config.rho * self.I + self.config.kappa * reduction

    # --------------------------------------------------------
    # 感情出力
    # --------------------------------------------------------

    def emotions(
        self,
        H_t: float,
        params: HumanParams,
        r_t: Array,
        sigma_t: float = 0.5,
    ) -> EmotionState:
        """
        感情出力の軽量版
        anger ~ H (1-r) n
        fear  ~ H (1-r) (1-sigma) n
        care  ~ H r o
        curiosity ~ H d (1-n)   # 実装上の追加
        """
        r_mean = float(np.mean(r_t))
        anger = H_t * (1.0 - r_mean) * params.n
        fear = H_t * (1.0 - r_mean) * (1.0 - sigma_t) * params.n
        care = H_t * r_mean * params.o
        curiosity = H_t * params.d * (1.0 - params.n)

        return EmotionState(
            anger=float(anger),
            fear=float(fear),
            care=float(care),
            curiosity=float(curiosity),
        )

    # --------------------------------------------------------
    # 1ステップ進める
    # --------------------------------------------------------

    def step(
        self,
        x_t: Array,
        y_hat_t: Array,
        params: HumanParams,
        r_t: Array,
        sigma_t: float = 0.5,
    ) -> StepInfo:
        """
        1ステップ進行
        """
        self._check_vec(x_t, "x_t")
        self._check_vec(y_hat_t, "y_hat_t")
        self._check_vec(r_t, "r_t")

        inertia_before = self.I

        # 1. 現在の出力
        y_t = self.predict_output(x_t)

        # 2. 誤差
        e_t = self.error(y_t, y_hat_t)

        # 3. 重み
        w_t = self.build_weight_vector(params, r_t)

        # 4. 熱
        H_t = self.heat(e_t, w_t)

        # 5. 学習率と跳躍閾値
        eta_t = self.learning_rate(params)
        theta_t = self.jump_threshold(params)

        # 6. 更新
        if H_t >= theta_t:
            A_next = self.leap_update(x_t, e_t, w_t, eta_t)
            jumped = True
        else:
            A_next = self.small_update(x_t, e_t, w_t, eta_t)
            jumped = False

        # 7. 更新後の熱を評価
        A_prev = self.A.copy()
        self.A = A_next
        y_next = self.predict_output(x_t)
        e_next = self.error(y_next, y_hat_t)
        H_next = self.heat(e_next, w_t)

        # 8. 整合慣性更新
        self.I = self.update_inertia(H_t, H_next)

        # 9. 感情出力
        emotion_state = self.emotions(H_t, params, r_t, sigma_t=sigma_t)

        return StepInfo(
            x_t=x_t.copy(),
            y_hat_t=y_hat_t.copy(),
            y_t=(A_prev @ x_t),
            e_t=e_t.copy(),
            w_t=w_t.copy(),
            H_t=H_t,
            H_next=H_next,
            eta_t=eta_t,
            theta_t=theta_t,
            jumped=jumped,
            inertia_before=inertia_before,
            inertia_after=self.I,
            emotions=emotion_state,
        )

    # --------------------------------------------------------
    # 補助
    # --------------------------------------------------------

    def run(
        self,
        inputs: Array,
        expected_outputs: Array,
        params: HumanParams,
        relations: Array,
        sigma_t: float = 0.5,
    ) -> list[StepInfo]:
        """
        複数ステップ実行

        inputs.shape = (T, dim)
        expected_outputs.shape = (T, dim)
        relations.shape = (T, dim)
        """
        if inputs.ndim != 2 or inputs.shape[1] != self.config.dim:
            raise ValueError("inputs must have shape (T, dim)")
        if expected_outputs.shape != inputs.shape:
            raise ValueError("expected_outputs must match inputs shape")
        if relations.shape != inputs.shape:
            raise ValueError("relations must match inputs shape")

        history = []
        for t in range(inputs.shape[0]):
            info = self.step(
                x_t=inputs[t],
                y_hat_t=expected_outputs[t],
                params=params,
                r_t=relations[t],
                sigma_t=sigma_t,
            )
            history.append(info)
        return history

    def snapshot(self) -> Dict[str, Any]:
        return {
            "A": self.A.copy(),
            "I": self.I,
            "config": self.config,
        }

    def _check_vec(self, x: Array, name: str) -> None:
        if x.shape != (self.config.dim,):
            raise ValueError(f"{name}.shape must be ({self.config.dim},), got {x.shape}")


# ============================================================
# 使用例
# ============================================================

def demo() -> None:
    np.random.seed(7)

    dim = 4

    config = RDLConfig(
        dim=dim,
        eta0=0.04,
        rho=0.93,
        kappa=0.5,
        theta0=1.8,
        theta_a=0.8,
        theta_b=1.0,
        theta_c=0.7,
        leap_scale=2.8,
        leap_noise_scale=0.02,
    )

    weights = WeightConfig(
        alpha=np.array([1.0, 1.0, 1.0, 1.0]),
        beta=np.array([0.8, 1.0, 0.9, 1.1]),
        gamma=np.array([0.7, 0.7, 0.7, 0.7]),
        delta=np.array([0.5, 0.5, 0.5, 0.5]),
    )

    model = RDLModel(config=config, weights=weights)

    params = HumanParams(
        d=0.6,
        s=0.4,
        o=0.7,
        n=0.5,
        c_obs=0.8,
        g=0.6,
    )

    # 3ステップだけ試す
    inputs = np.array([
        [1.0,  0.2, -0.1, 0.6],
        [0.8, -0.5,  0.3, 0.2],
        [1.2,  0.1, -0.4, 0.9],
    ], dtype=float)

    # 想定出力（外部から与える）
    expected_outputs = np.array([
        [0.9,  0.2, -0.1, 0.5],
        [0.7, -0.3,  0.2, 0.1],
        [1.0,  0.0, -0.2, 0.8],
    ], dtype=float)

    # 各成分への関係強度
    relations = np.array([
        [0.2, 0.9, 0.4, 0.8],
        [0.1, 0.7, 0.3, 0.5],
        [0.3, 0.8, 0.2, 0.9],
    ], dtype=float)

    history = model.run(
        inputs=inputs,
        expected_outputs=expected_outputs,
        params=params,
        relations=relations,
        sigma_t=0.45,
    )

    for i, step in enumerate(history):
        print("=" * 60)
        print(f"step: {i}")
        print("y_t       :", np.round(step.y_t, 4))
        print("y_hat_t   :", np.round(step.y_hat_t, 4))
        print("e_t       :", np.round(step.e_t, 4))
        print("w_t       :", np.round(step.w_t, 4))
        print("H_t       :", round(step.H_t, 6))
        print("H_next    :", round(step.H_next, 6))
        print("eta_t     :", round(step.eta_t, 6))
        print("theta_t   :", round(step.theta_t, 6))
        print("jumped    :", step.jumped)
        print("I(before) :", round(step.inertia_before, 6))
        print("I(after)  :", round(step.inertia_after, 6))
        print("emotions  :", step.emotions)

    print("=" * 60)
    print("Final inertia:", round(model.I, 6))
    print("Final A:")
    print(np.round(model.A, 4))


if __name__ == "__main__":
    demo()