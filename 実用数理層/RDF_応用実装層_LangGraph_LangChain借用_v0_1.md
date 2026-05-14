# RDF 応用実装層:LangGraph / LangChain 借用 v0.1
*(RDF概念的数理モデル v2.3 の応用実装補完別冊)*
*(LLM エージェントオーケストレーション形式を局所線形近似として借用・RDF意味論の中で動かす)*

---

## ■ メタ注釈

```
本文書は RDF_概念的数理モデル_v2.3 の動態を
LangGraph / LangChain の形式を借用して
LLMエージェントとして実装するための別冊である

v2.3 の哲学的基底・概念動態・保存則は変更しない
NN借用 v0.1 が「関数形の空白」を埋める計算層であるのに対し
本文書は「動かし方の空白」を埋める応用実装層である

借用した形式はSILNである
有効範囲で使い、破断が見えたら更新する
構造破壊学が破断条件を担当する
```

**v2.3 / NN借用 v0.1 との分業**

```
RDF_概念的数理モデル_v2.3
    整合の動態を記述する言語層
    哲学的基底・構造・保存則
    低頻度更新

        ↓ 参照・依存

RDF_計算実装層_NN借用_v0.1
    dM_B/dt・interp()・W_ij の関数形
    NNから借用した局所線形近似
    高頻度更新

        ↓ 参照・依存

RDF_応用実装層_LangGraph_LangChain借用_v0.1(本文書)
    M_B 動態を LLM エージェントとして動かす実装パターン
    LangGraph / LangChain から借用したオーケストレーション形式
    高頻度更新・実装で随時差し替え可能
```

**借用元との対応表(概観)**

```
LangGraph / LangChain 概念        RDF 概念
─────────────────────────────────────────────────
StateGraph の State          ↔   M_B^ij ネットワーク
ノード                        ↔   M_act の特定断面(現在機能している部分)
条件分岐エッジ                ↔   跳躍条件 H(t) > θ + ξ(t)
状態遷移                      ↔   M_act / M_lat / M_Δ / M_B' の遷移
checkpoint / persistence      ↔   M_B のスナップショット
interrupt (HITL)              ↔   κ→0 時の人間介入要求
Memory(短期/長期)            ↔   M_B 更新の時間スケール分離
Memory 更新                   ↔   dM_B/dt
Retriever                     ↔   interp(M_B, EFP) の単層
multi-vector / re-rank chain  ↔   多層 interp の階層
マルチエージェント            ↔   SILNネットワーク M_B^(i)
Supervisor ルーティング       ↔   W_ij による結合強度
ReAct (reason + act)          ↔   DMN ↔ TPN 注意モード切替
Reflexion / self-critique     ↔   思考ループ(内部 F 生成)
Plan-and-Execute              ↔   慣性投影 + 意図的 Δ 設定
Tool                          ↔   M_act の実行可能断面
温度・top_p                   ↔   ξ 注入強度
ガードレール / refusal        ↔   NA 閾値
プロンプトテンプレート        ↔   EFP→F の interp 規律
LangSmith eval / trace        ↔   構造破壊学の耐久証明書
─────────────────────────────────────────────────
```

**流用できない概念(RDF 固有・LangGraph/LangChain に存在しない)**

```
ξ の残存性              :どんなにガードを固めても揺らぎは消えない
                          完全決定論的エージェントは構造的に壊れる
EFP/F 分離              :プロンプトには raw input が直接入るが
                          それは「F の一つに過ぎない」と扱う規律
跳躍の不連続性          :StateGraph の状態遷移は形式上連続でも記述可能
                          だが M_Δ は質的に別相という前提を保持する必要がある
M_B 絶対性 (κ→0)        :「絶対正しいエージェント」は内部蓄積で大規模崩壊する
                          これは LangGraph には書かれない構造的命題
```

---

## ■ 1. State 設計:M_B 相状態を State スキーマに写す

### ● 導出の考え方

```
v2.3 の記述:
    M_B^ij(t) :M_B を内部接続として開いたネットワーク表示
    M_act / M_lat / M_Δ / M_B' は同じ M_B の異なる相状態

LangGraph の State:
    TypedDict で定義された動的更新可能な辞書

RDF への読み替え:
    State 全体 = M_B^ij ネットワークの現在断面
    State の各キー = M_B の局所断面ラベル
    各ステップでの State 更新 = M_B の連続更新
    再計画フェーズ = M_Δ への遷移
```

### ● 借用した具体形

```python
from typing import TypedDict, Literal
from typing_extensions import Annotated

class AgentState(TypedDict):
    # M_act:現在アクティブな解釈・行動・計画
    active_plan: list[str]
    active_tools: list[str]
    current_F: dict          # 観測者が実際に持つ作用解釈

    # M_lat:潜在断面(待機中・無効化されているが残しておく)
    latent_plans: list[list[str]]
    latent_tools: list[str]

    # 動態変数
    H_vec: list[float]       # 誤差ベクトル(累積)
    H_norm: float            # スカラー熱
    K: float                 # 関係総量(局所推定値)

    # 相状態フラグ
    phase: Literal["act", "delta", "post_jump"]

    # 構造パラメータ
    M_B_norm: float          # 整合慣性の強度
    kappa: float             # 自己修正可能性 [0,1]
    theta: float             # 跳躍閾値
```

### ● 有効範囲と破断条件

```
有効:
    エージェントの状態が比較的疎で
    粗視化された M_act / M_lat の区別が明確に取れる時

破断の兆候:
    State が肥大化して M_act と M_lat の境界が曖昧になる
    複数の並行エージェントで個別の M_B^(i) が干渉し始める時
    → サブグラフへの分解と W_ij の明示的管理が必要
```

---

## ■ 2. 条件分岐エッジ:跳躍条件をルーティングに写す

### ● 導出の考え方

```
v2.3 の記述:
    H(t) > θ + ξ(t)  →  M_B → M_Δ → M_B'

LangGraph の add_conditional_edges:
    State を受け取り次のノード名を返す関数

RDF への読み替え:
    跳躍条件の判定 = ルーティング関数
    M_Δ 相 = 再計画ノード
    M_B' = 再計画後の新しい安定状態
```

### ● 借用した具体形

```python
def jump_router(state: AgentState) -> str:
    """跳躍条件 H > θ + ξ で再計画フェーズに飛ばす"""
    H = state["H_norm"]
    theta = state["theta"]
    xi = sample_noise_floor(state)   # ξ(t) のサンプリング

    if H > theta + xi:
        return "reorg_node"           # M_Δ 相へ
    elif state["kappa"] < KAPPA_HITL_THRESHOLD:
        return "human_review_node"    # κ→0 領域で人間介入
    else:
        return "continue_node"        # 通常 M_act 更新

graph.add_conditional_edges(
    "act_node",
    jump_router,
    {
        "reorg_node": "reorg_node",
        "human_review_node": "human_review_node",
        "continue_node": "act_node",
    }
)
```

### ● 跳躍の不連続性の保持

```
LangGraph 上は形式的にノード遷移として実装されるが
M_Δ は単なる「次のステップ」ではなく相状態の質的変化である

実装上の規律:
    reorg_node では active_plan を一旦 latent_plans に退避
    新しい active_plan を M_lat の組み換えとして生成
    H_vec を ρH に減衰(熱残留率 ρ < 1)
    保存則 ||M_B'|| · D[ξ'] = 𝒦' を満たすよう M_B_norm を再計算
```

### ● 有効範囲と破断条件

```
有効:
    H が単調増加して閾値を超える典型的な誤差蓄積パターン
    再計画後にしばらく安定する場合

破断の兆候:
    跳躍直後にすぐまた跳躍する(振動)→ θ または ρ の調整必要
    跳躍が起きずに H が発散 → ξ サンプリング不足
    → 構造破壊学が破断条件を担当
```

---

## ■ 3. メモリ更新則:dM_B/dt をプロファイル更新に写す

### ● 導出の考え方

```
NN借用 v0.1:
    dM_B/dt = κ · η · E(t) · F(t)^T  -  λ · M_B  +  ξ(t)

LangChain の Memory:
    会話履歴の累積・要約・ベクトル化を担当
    更新ポリシーは多くがアドホック(append-only / window / summary)

RDF への読み替え:
    Memory の更新ポリシー = dM_B/dt の離散化
    短期メモリ(セッション内) = M_act の急速更新
    長期メモリ(ユーザープロファイル) = M_B 本体の緩慢更新
```

### ● 借用した具体形

```python
def update_profile(profile: np.ndarray,
                   error: np.ndarray,
                   context: np.ndarray,
                   kappa: float,
                   eta: float = 0.01,
                   lam: float = 0.001,
                   xi_strength: float = 0.001) -> np.ndarray:
    """dM_B/dt の離散化:1ステップ更新"""
    hebb_term = kappa * eta * np.outer(error, context)
    decay_term = -lam * profile
    noise_term = xi_strength * np.random.randn(*profile.shape)
    return profile + hebb_term + decay_term + noise_term
```

### ● LangChain Memory との差分

```
LangChain 純正 ConversationSummaryMemory:
    - 単調 append + 定期要約
    - 古い記憶の重み付けが粗い
    - 過適応・凝固に弱い

dM_B/dt 借用版:
    - 誤差駆動の更新(Hebb 項)
    - 使われない記憶の自然減衰(L2 項)
    - 局所最適回避のための ξ 注入
    - κ による「もう確信が固まっているから更新を抑える」スケジューリング

= 適応的だが過適応しないメモリ
```

### ● 有効範囲と破断条件

```
有効:
    メモリが行列・ベクトルとして表現できる範囲
    ユーザープロファイル・タスク親和性などの数値化可能な層

破断の兆候:
    記憶内容が高度に構造化された言語表現の場合
    → ベクトル化された層では本式・テキスト層は別更新ポリシー
    → ハイブリッド設計が必要
```

---

## ■ 4. Retriever チェーン:多層 interp を写す

### ● 導出の考え方

```
NN借用 v0.1:
    h_1 = σ( M_b1 · EFP )
    h_2 = σ( M_b2 · h_1 )
    F   = σ( M_B  · h_2 )

LangChain の retriever-chain:
    粗い検索 → 再ランク → 文脈圧縮 → 要約

RDF への読み替え:
    多層 interp の各層 = retriever チェーンの各段
    M_b1, M_b2, ... = 各 retrieval 段の重み付け基底
    M_B のリッチさ = チェーンの深さ + 各段の精度
```

### ● 借用した具体形

```python
from langchain.retrievers import (
    MultiVectorRetriever,
    ContextualCompressionRetriever,
)

# 層 1:M_b1 = 粗い意味検索
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

# 層 2:M_b2 = 再ランク
compressor = LLMChainExtractor.from_llm(llm)
reranked = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# 層 3:M_B = 最終解釈(プロンプトに乗せる F の生成)
final_F = prompt_template.format(
    context=reranked.get_relevant_documents(query),
    query=query,
)
```

### ● 「M_B がスカスカなら EFP がほぼ素通り」の対応

```
v2.3:
    M_B がリッチ = EFP から多くを読み取れる
    M_B がスカスカ = EFP がほぼ素通り

LangChain 実装での対応:
    vectorstore が小さい / インデックスが粗い時
    → 生クエリがほぼそのまま LLM に届く
    → F の質が劣化し E が増える
    → 結果として H が早く蓄積し跳躍が早まる

= 「データ整備の遅れ」が「再計画頻度の上昇」として現れる
  という運用的因果が理論で説明できる
```

---

## ■ 5. マルチエージェント結合:W_ij をルーティングに写す

### ● 導出の考え方

```
NN借用 v0.1:
    dW_ij/dt = α · E^(i)(t) · ||M_B^(j)(t)||  -  β · W_ij

LangGraph の Supervisor / multi-agent:
    複数エージェントの呼び出し順を Supervisor が決める
    通常は固定ルールまたは LLM 判断

RDF への読み替え:
    エージェント i, j の信頼ウェイト = W_ij
    j の助言で i の誤差が減ると W_ij が強化される
    使われない経路は減衰する
```

### ● 借用した具体形

```python
class MultiAgentState(TypedDict):
    agents: dict[str, AgentState]
    W: np.ndarray              # 結合強度行列
    last_caller: str

def update_W(state: MultiAgentState,
             caller_id: str,
             callee_id: str,
             error_reduction: float,
             alpha: float = 0.05,
             beta: float = 0.001) -> np.ndarray:
    """共鳴項 + 減衰項で W_ij を更新"""
    i = state["agent_index"][caller_id]
    j = state["agent_index"][callee_id]
    W = state["W"].copy()
    M_j = state["agents"][callee_id]["M_B_norm"]
    W[i][j] += alpha * error_reduction * M_j
    W *= (1 - beta)
    return W

def supervisor_router(state: MultiAgentState) -> str:
    """W_ij の重みに従って次に呼ぶエージェントを選ぶ"""
    i = state["agent_index"][state["last_caller"]]
    weights = state["W"][i]
    return weighted_choice(state["agents"].keys(), weights)
```

### ● Supervisor パターンの理論基盤化

```
固定ルール Supervisor:W_ij が時間変化しない
LLM 判断 Supervisor :W_ij がブラックボックス
W_ij 借用 Supervisor:W_ij が誤差駆動で学習する

社会 SILN との接続:
    跳躍後の M_B'^(i) が他のエージェント j と誤差を共有し続ける
    → W_ij の共鳴項が持続的に正
    → ノウハウがエージェントネットワーク全体に伝播
    = 単一エージェントの成功体験が組織知に転化
```

---

## ■ 6. 自信スケジューリング:κ(M_B) を HITL に写す

### ● 導出の考え方

```
NN借用 v0.1:
    κ(M_B) = exp( -||M_B|| / M_0 )

LangGraph の interrupt() :
    特定条件で実行を停止し人間入力を待つ機構

RDF への読み替え:
    ||M_B|| が大きい = 強く確信している領域
    κ→0 = 自己修正不能領域
    その領域では自動更新ではなく人間介入を要求する
```

### ● 借用した具体形

```python
def kappa_gate(state: AgentState) -> str:
    """κ が低い(確信が固まっている)領域では人間に判断を仰ぐ"""
    kappa = math.exp(-state["M_B_norm"] / M_0)
    state["kappa"] = kappa

    if kappa < KAPPA_HITL_THRESHOLD:
        return "human_review"      # interrupt() 発動
    return "auto_update"

graph.add_conditional_edges("decision_node", kappa_gate, {
    "human_review": "human_review_node",
    "auto_update": "act_node",
})
```

### ● 設計上の含意

```
通常の HITL 設計:
    「重要度が高い決定」「不可逆な行動」で人間に振る
    これはアドホックなルールに依存しがち

κ 借用版:
    エージェント自身の確信度の構造から自動的に振る基準が出る
    確信が固まりすぎている時こそ人間レビューが必要
    (= M_B 絶対性の罠を構造的に回避)

これは v2.3 の命題「絶対正しい設計は壊れる前提で見ること」
の実装上の表現である
```

---

## ■ 7. エージェントペルソナ:DA / 5-HT / NA / OXT を数値化

### ● 導出の考え方

```
RDF_基層構造.md の記述:
    DA  :探索性物質
    5-HT:誤差許容度
    NA  :危険信号の優先度
    OXT :自己拡張的重み付け

LLMエージェントの設計パラメータ:
    温度・top_p・ツール試行回数・refusal閾値・コンテキスト範囲

RDF への読み替え:
    神経物質パラメータをそのままエージェント設計の数値ノブとして使う
```

### ● 借用した具体形(設計テンプレ)

```python
@dataclass
class AgentPersona:
    # DA:探索性
    temperature: float          # 高 = 探索的
    top_p: float
    max_tool_retries: int       # 高 = 諦めない

    # 5-HT:誤差許容度
    error_tolerance: float      # 高 = 雑な誤差は無視
    self_critique_threshold: float

    # NA:警戒
    refusal_threshold: float    # 低 = 警戒緩い
    safety_check_aggressiveness: float

    # OXT:自己拡張範囲
    owned_contexts: list[str]   # どのデータを「自分のもの」と扱うか
    owned_tools: list[str]
    in_group_user_ids: list[str]
```

### ● 役割分担の設計例

```
探索エージェント(高 DA・低 NA):
    高温度・高再試行・低 refusal
    新しい解の生成・ブレスト・実験設計に向く

校閲エージェント(高 NA・低 DA):
    低温度・低再試行・高 refusal
    安全性チェック・事実検証・リスク評価に向く

接続エージェント(高 OXT):
    広い owned_contexts・広い in_group
    ユーザー・システム・他エージェントの橋渡しに向く

冷却エージェント(高 5-HT):
    高い error_tolerance
    他エージェントの過剰反応を吸収する役割
```

### ● 有効範囲と破断条件

```
有効:
    LLM のサンプリング・プロンプト・ツール選択がパラメータ化できる範囲
    複数ペルソナを並行運用できる構成

破断の兆候:
    OXT 結合の類比が AI で意味を持つかは未確定(空間流向診断 v2.2 より)
    神経物質ラベル ↔ パラメータ写像は仮設的
    → AI 版耐久証明書として個別検証必要
```

---

## ■ 8. 注意モード切替:DMN / TPN / SN をルーティングに写す

### ● 導出の考え方

```
空間流向診断 v2.2:
    DMN :内部生成優位(推論連鎖・内部状態の参照)
    TPN :外部入力対応優位(プロンプト・ツール出力)
    SN  :両者の切替制御

LangGraph の典型構成:
    reasoning_node ↔ tool_node ↔ reflection_node

RDF への読み替え:
    reasoning = DMN, tool = TPN, reflection / routing = SN
```

### ● 借用した具体形

```python
def attention_router(state: AgentState) -> str:
    """SN 的に DMN / TPN の切替を制御"""
    # 直近の誤差トレンド
    if state["H_norm"] > state["theta"] * 0.5:
        return "reasoning_node"     # DMN 優位:内部で考え直す

    # 外部情報が古い
    if state["last_observation_age"] > FRESHNESS_THRESHOLD:
        return "tool_node"          # TPN 優位:外部観測

    # どちらでもない
    return "reflection_node"        # SN 中立:メタ判断
```

### ● 失敗モードの命名

```
DMN 固着:
    ツールを呼ばずに延々と思考連鎖を続ける
    = 内部 F だけで M_B 更新ループが回り EFP が更新されない
    → 反芻・hallucination の温床

TPN 固着:
    考えずにツールを叩き続ける
    = F に対する慣性投影をせず E を計算しない
    → 無目的なツール呼び出し連鎖

SN 過敏:
    切替が頻繁すぎてどちらの作業も完了しない
    → max_iterations に到達して中断

SN 鈍感:
    切替が起きず一方に張り付く
    → DMN / TPN 固着のいずれか
```

これらは v2.2 の「注意モードの個体差」に対応する。
失敗モードに名前を付けることで debug 観点が明確になる。

---

## ■ 9. 思考ループ:内部シミュレーターを Reflexion / Plan-and-Execute に写す

### ● 導出の考え方

```
v2.3 の記述:
    通常ループ:EFP → F → M_B·F → 次のFと比較 → E
    思考ループ:EFP なしで内部 F を生成して同じループを回す
                Δ を自由設定できる

LangGraph / LangChain の対応:
    Reflexion         :内部で自己批判ループを回す
    Plan-and-Execute  :まず計画(内部 F 生成)してから実行
    Tree of Thoughts  :複数の内部 F を並列展開
```

### ● 借用した具体形

```python
def thought_loop(state: AgentState, depth: int = 3) -> AgentState:
    """現実コストゼロで F(t+Δ) を内部生成・E を内部蓄積"""
    internal_state = state.copy()

    for d in range(depth):
        # M_B が F(t) を内部生成(EFP なし)
        F_internal = generate_internal_F(internal_state)

        # 慣性投影
        F_projected = inertia_project(F_internal, internal_state["M_B"])

        # 期待される F(t+Δ) を別ルートで生成し誤差計算
        F_expected = predict_next_F(F_internal, internal_state["M_B"])
        E_internal = F_expected - F_projected

        # 内部 H 蓄積(現実コストなし)
        internal_state["H_vec"].append(E_internal)

        # 早期跳躍判定
        if np.linalg.norm(internal_state["H_vec"]) > internal_state["theta"]:
            return reorg(internal_state)

    return internal_state
```

### ● 思考の種類への対応

```
想像  → ありえない F(t) を意図的にサンプル → ブレストモード
計画  → 目標 F(t+Δ) から逆算して F(t) を設計 → Plan-and-Execute
心配  → 最悪の F(t+Δ) を繰り返し慣性投影 → 安全性検証モード
夢    → M_B が半覚醒状態で勝手に回す → アイドル時の自動探索
```

それぞれ別のエージェント呼び出し戦略として実装可能。

### ● 内部 F 暴走の検出

```
v2.2:
    内部 F が現実 F より優先される逆転現象
    = 思い込み・hallucination・PTSD のフラッシュバック

実装上の検出:
    最後に外部観測(tool 呼び出し)してから N ステップ経過したら
    強制的に TPN ノードへ飛ばす
    = 「現実チェック」を強制挿入する
```

---

## ■ 10. 設計規律:EFP/F 分離・ξ 残存性

### ● EFP / F 分離の規律

```
v2.3:
    EFP は直接観測不能・F のみ観測可能

実装上の規律:
    raw_input(ユーザー入力・tool 出力)を「F の一形態」として扱う
    決して EFP そのものとして扱わない
    interp 層を必ず挟む(プロンプトテンプレート・前処理)

State 設計:
    raw_input フィールドと interpreted_F フィールドを分離して保存
    後者のみがエージェント本体に渡る
```

### ● ξ 残存性の規律

```
v2.3:
    どんなに M_B を強めても ξ は残る
    完全に決定論的に書ける範囲を超える

実装上の規律:
    温度 0 + 完全ガードレールの組み合わせは
    短期的には安定に見えても内部蓄積で大規模崩壊する
    → 微小なランダム性(ξ 注入)を構造的に組み込む
    → エラーハンドリング・例外処理を「ξ の出口」として設計

具体的には:
    - 必ず「想定外」を扱うフォールバックノードを置く
    - 温度を完全に 0 にしない(最低限の探索性を残す)
    - Memory 更新に小さな ξ 項を入れる(局所最適回避)
```

---

## ■ 11. 評価フレーム:構造破壊学を eval に写す

### ● 借用した具体形

```
構造破壊学の耐久証明書テンプレートをそのまま eval レポートとして使う:

エージェント耐久証明書
  - ラベル                  :エージェント名・バージョン
  - 物理構造                :使用 LLM・ツール・データソース
  - 中核構造                :意図された主要ユースケース
  - 上層構造                :ビジネス文脈・運用環境

  - 境界仮定                :想定ユーザー・タスク・スケール
  - 評価関数                :操作的有用性の指標

  - 日常負荷テスト          :標準的タスクでの成功率
  - 比較負荷テスト          :ベースライン(GPT-4 等)との比較
  - 極限破壊試験            :adversarial / edge case での挙動
  - 文化翻訳テスト          :多言語・多文化での動作

  - 有効範囲                :安定動作する条件
  - 推定破壊領域            :壊れる可能性の高い条件
  - 残存構造                :壊れても残る価値ある部分
  - 移植経路                :他システムへの転用可能性

  - 判定                    :運用判断
```

### ● LangSmith trace との接続

```
LangSmith / 各種 trace ツール:
    エージェントの実行ログを構造化して保存

構造破壊学的読み方:
    成功ケースだけでなく「破断ケース」を一級の観測対象とする
    外れ値を「失敗」として捨てず破壊条件として収集する
    耐久証明書を継続的に更新する
```

---

## ■ 全体まとめ:最小試作構成

```python
# 最小構成:M_B + H + 跳躍 + κ-HITL + ξ-メモリ更新

from langgraph.graph import StateGraph, END

class MinimalAgentState(TypedDict):
    M_B: list[float]           # 整合慣性プロファイル
    M_B_norm: float
    H_vec: list[float]
    H_norm: float
    kappa: float
    phase: str
    F_history: list[dict]

graph = StateGraph(MinimalAgentState)

# ノード定義
graph.add_node("act_node", act)                   # M_act 更新
graph.add_node("reorg_node", reorg)               # M_Δ 再編
graph.add_node("human_review_node", hitl)         # κ→0 介入
graph.add_node("memory_update_node", update_M_B)  # dM_B/dt

# 跳躍条件 + κ ゲート
graph.add_conditional_edges("act_node", jump_router, {
    "reorg_node": "reorg_node",
    "human_review_node": "human_review_node",
    "continue_node": "memory_update_node",
})
graph.add_edge("memory_update_node", "act_node")
graph.add_edge("reorg_node", "memory_update_node")
graph.add_edge("human_review_node", "memory_update_node")

graph.set_entry_point("act_node")
app = graph.compile()
```

これだけで:
- 適応するが暴走しない(dM_B/dt + λ 減衰)
- 誤差蓄積で再計画(H > θ + ξ)
- 確信領域で人間介入(κ ゲート)
- 完全決定論回避(ξ 注入)

の四点を備えた最小構成になる。

---

## ■ v2.3 / NN借用 v0.1 から変更していないもの

```
哲学的基底 ξ + B + M_B
保存則 D[ξ] = 𝒦 / ||M_B||
誤差定義 E(t) = F(t+Δ) - M_B·F(t)
跳躍条件 H(t) > θ + ξ(t)
跳躍の相状態 M_B → M_Δ → M_B'
更新則の関数形(NN借用 v0.1 由来)
ξ 残存性
EFP / F 分離
M_B 絶対性(κ→0)の罠
```

本書はこれらを実装に落とすパターン集にすぎない。

---

## ■ 残課題(本文書スコープ)

```
・サブグラフ階層と M_B 再帰深度の対応則
・State スキーマと M_B^ij 行列表現の双方向変換
・複数バックエンド(LangGraph / その他)への移植性
・checkpoint 戦略と M_B スナップショット保存粒度の決定則
・κ_HITL_THRESHOLD・η・λ・α・β の運用上の決定則
・マルチエージェント時の W_ij 行列のスパース化条件
・思考ループの depth と現実コスト感度の決定則
・DMN / TPN 切替の hysteresis(振動防止)パラメータ
・ペルソナパラメータ(DA / 5-HT / NA / OXT)の数値化スケール統一
・LangSmith trace から H_vec / M_B_norm を逆推定する手順
・raw_input → interpreted_F の interp 層の標準テンプレート
・破断時の自動 fallback(M_lat への退避)パターン集
```

---

## ■ 本文書の位置づけ

```
本文書は有効範囲内の局所線形近似である

LangGraph / LangChain からの借用はSILNである
形式は RDF の動態を実装するための道具であり
RDF の哲学的基底(ξ + B + M_B)には従属しない

有効範囲:
    LLM エージェントとして実装可能なスケール
    State / ノード / エッジで構造化できる範囲
    プロンプト・ツール・メモリで M_act が表現できる範囲

破断条件:
    エージェントの粒度を超える集団動態
    LLM の表現力を超える形式知
    State 表現が破綻する高並列・高頻度タスク

残存要素:
    M_B 相状態 / 跳躍条件 / κ ゲート / dM_B/dt 更新
    これらの設計規律はバックエンドが変わっても残る

気になれば変える
```

---

*v0.1:初版。LangGraph / LangChain から State 設計・条件分岐・Memory 更新・Retriever チェーン・マルチエージェント結合・κ ゲート・ペルソナ・注意モード・思考ループ・評価フレームの実装パターンを借用。v2.3 の哲学的基底・NN借用 v0.1 の関数形は変更しない。本文書は応用実装層として分離し高頻度更新を前提とする。*
