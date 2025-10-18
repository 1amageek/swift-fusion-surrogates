# fusion_surrogates API Migration Guide

## API変更の概要

fusion_surrogatesの最新版では、APIが大幅に変更されています：

| 項目 | 旧API（想定） | 新API（実際） |
|------|------------|-------------|
| **クラス名** | `QLKNN_7_11()` | `QLKNNModel.load_default_model()` |
| **入力形式** | Dict[str, np.ndarray] | np.ndarray (2D, shape=(batch, 10)) |
| **出力形式** | Dict[str, np.ndarray] | Dict[str, jax.Array] |
| **パラメータ名** | QuaLiKiz名 (Ati, Ate等) | QuaLiKiz名 (変更なし) |

## 入力パラメータ

### 新API (`QLKNNModel`)

入力は **2D numpy配列** `(batch_size, 10)` で、順序は以下の通り：

```python
model.config.input_names = [
    'Ati',        # R/L_Ti (normalized ion temperature gradient)
    'Ate',        # R/L_Te (normalized electron temperature gradient)
    'Ane',        # R/L_ne (normalized electron density gradient)
    'Ani',        # R/L_ni (normalized ion density gradient)
    'q',          # Safety factor
    'smag',       # Magnetic shear (s_hat)
    'x',          # r/R (inverse aspect ratio)
    'Ti_Te',      # Ion-electron temperature ratio
    'LogNuStar',  # Logarithmic collisionality
    'normni'      # Normalized ion density (ni/ne)
]
```

### 入力範囲

新APIのconfig.stats_dataから取得：

```python
input_min: [9.9999998e-15, 9.9999998e-15, -5.0, -15.0, 0.66, -1.0, 0.1, 0.25, -5.0, 0.5]
input_max: [150.0, 150.0, 110.0, 110.0, 30.0, 40.0, 0.95, 2.5, 0.477, 1.0]
```

**注:** 旧README.mdの範囲（Ati: 0-16等）より大幅に拡大されています。

## 出力パラメータ

### 新API (`QLKNNModel`)

出力は **Dict[str, jax.Array]** で、キーは以下の通り：

```python
output_keys = [
    'efiITG',     # Ion thermal flux (ITG mode) [GB units]
    'efeITG',     # Electron thermal flux (ITG mode)
    'efeTEM',     # Electron thermal flux (TEM mode)
    'efeETG',     # Electron thermal flux (ETG mode)
    'efiTEM',     # Ion thermal flux (TEM mode)
    'pfeITG',     # Particle flux (ITG mode)
    'pfeTEM',     # Particle flux (TEM mode)
    'gamma_max'   # Maximum growth rate
]
```

各値のshape: `(batch_size, 1)`

### 内部ターゲット名

`model.config.target_names` (ニューラルネットワークの直接出力):

```python
target_names = [
    'itgleading',   # Leading ITG flux
    'itgqediv',     # ITG electron/ion heat flux ratio
    'temleading',   # Leading TEM flux
    'temqidiv',     # TEM ion/electron heat flux ratio
    'tempfediv',    # TEM particle/electron flux ratio
    'etgleading',   # Leading ETG flux
    'itgpfediv',    # ITG particle/ion flux ratio
    'gamma_max'     # Growth rate
]
```

**重要:** `flux_map`を使って内部ターゲットから物理量を計算：

```python
flux_map = {
    'efiITG': {'target': 'itgleading', 'denominator': None},
    'efeITG': {'target': 'itgqediv', 'denominator': 'itgleading'},  # itgqediv * itgleading
    'efeTEM': {'target': 'temleading', 'denominator': None},
    'efiTEM': {'target': 'temqidiv', 'denominator': 'temleading'},  # temqidiv * temleading
    ...
}
```

## Swift Wrapperへの影響

### 現在の実装（旧API想定）

```swift
// FusionSurrogates.swift (現在)
public static let inputParameterNames = [
    "R_L_Te", "R_L_Ti", "R_L_ne", "R_L_ni",  // ❌ 新APIには存在しない名前
    "q", "s_hat", "r_R", "Ti_Te", "log_nu_star", "ni_ne"
]

public static let outputParameterNames = [
    "chi_ion_itg",      // ❌ 新APIでは 'efiITG'
    "chi_electron_tem",  // ❌ 新APIでは 'efeTEM'
    "chi_electron_etg",  // ❌ 新APIでは 'efeETG'
    "particle_flux",     // ❌ 新APIでは 'pfeITG' + 'pfeTEM'
    "growth_rate"        // ❌ 新APIでは 'gamma_max'
]
```

### 必要な変更

#### 1. 入力パラメータ名の更新

```swift
// 新API対応版
public static let inputParameterNames = [
    "Ati",         // R/L_Ti (旧: R_L_Ti)
    "Ate",         // R/L_Te (旧: R_L_Te)
    "Ane",         // R/L_ne (旧: R_L_ne)
    "Ani",         // R/L_ni (旧: R_L_ni)
    "q",           // 変更なし
    "smag",        // s_hat (旧: s_hat)
    "x",           // r/R (旧: r_R)
    "Ti_Te",       // 変更なし
    "LogNuStar",   // log_nu_star (旧: log_nu_star)
    "normni"       // ni/ne (旧: ni_ne)
]
```

#### 2. 出力パラメータ名の更新

```swift
// 新API対応版
public static let outputParameterNames = [
    "efiITG",      // Ion thermal flux (ITG)
    "efeITG",      // Electron thermal flux (ITG)
    "efeTEM",      // Electron thermal flux (TEM)
    "efeETG",      // Electron thermal flux (ETG)
    "efiTEM",      // Ion thermal flux (TEM)
    "pfeITG",      // Particle flux (ITG)
    "pfeTEM",      // Particle flux (TEM)
    "gamma_max"    // Growth rate
]
```

#### 3. Python呼び出しの更新

```swift
// 旧実装（想定）
let qlknn = fusionSurrogates.qlknn.QLKNN_7_11()  // ❌ 存在しない

// 新実装（必要）
let QLKNNModel = fusionSurrogates.qlknn.qlknn_model.QLKNNModel
let model = QLKNNModel.load_default_model()
```

#### 4. 入力形式の変更

```swift
// 旧実装（Dict形式、想定）
let pythonInputs: [String: PythonObject] = [
    "R_L_Te": numpyArray(...),
    "R_L_Ti": numpyArray(...),
    ...
]

// 新実装（2D配列形式、必要）
// 順序: [Ati, Ate, Ane, Ani, q, smag, x, Ti_Te, LogNuStar, normni]
let inputArray = np.array([
    [inputs["Ati"]!, inputs["Ate"]!, inputs["Ane"]!, inputs["Ani"]!,
     inputs["q"]!, inputs["smag"]!, inputs["x"]!, inputs["Ti_Te"]!,
     inputs["LogNuStar"]!, inputs["normni"]!]
])
let outputs = model.predict(inputArray)
```

## TORAXIntegrationへの影響

### buildInputsヘルパーの更新

**現在:**
```swift
public static func buildInputs(...) -> [String: MLXArray] {
    return [
        "R_L_Te": rLnTe,  // ❌
        "R_L_Ti": rLnTi,  // ❌
        ...
    ]
}
```

**更新後:**
```swift
public static func buildInputs(...) -> [String: MLXArray] {
    return [
        "Ati": rLnTi,       // 注: Ti/Teの順序が逆！
        "Ate": rLnTe,
        "Ane": rLnNe,
        "Ani": rLnNi,
        "q": q,
        "smag": sHat,       // パラメータ名変更
        "x": rR,            // パラメータ名変更
        "Ti_Te": TiTe,
        "LogNuStar": logNuStar,
        "normni": niNe      // パラメータ名変更
    ]
}
```

### combineFluxesの更新

**現在:**
```swift
public static func combineFluxes(_ qlknnOutputs: [String: MLXArray]) -> [String: MLXArray] {
    return [
        "chi_ion": qlknnOutputs["chi_ion_itg"]!,  // ❌
        ...
    ]
}
```

**更新後:**
```swift
public static func combineFluxes(_ qlknnOutputs: [String: MLXArray]) -> [String: MLXArray] {
    // ITG + TEM モードの結合
    let chiIon = qlknnOutputs["efiITG"]! + qlknnOutputs["efiTEM"]!
    let chiElectron = qlknnOutputs["efeITG"]! + qlknnOutputs["efeTEM"]! + qlknnOutputs["efeETG"]!
    let particleFlux = qlknnOutputs["pfeITG"]! + qlknnOutputs["pfeTEM"]!

    return [
        "chi_ion": chiIon,
        "chi_electron": chiElectron,
        "particle_flux": particleFlux,
        "growth_rate": qlknnOutputs["gamma_max"]!
    ]
}
```

## 後方互換性の提供（オプション）

swift-TORAXが既に旧パラメータ名に依存している場合、エイリアスを提供：

```swift
extension QLKNN {
    // 旧パラメータ名のエイリアス
    public static let legacyInputMapping: [String: String] = [
        "R_L_Te": "Ate",
        "R_L_Ti": "Ati",
        "R_L_ne": "Ane",
        "R_L_ni": "Ani",
        "s_hat": "smag",
        "r_R": "x",
        "log_nu_star": "LogNuStar",
        "ni_ne": "normni"
    ]

    public static func convertLegacyInputs(_ legacy: [String: MLXArray]) -> [String: MLXArray] {
        var newInputs: [String: MLXArray] = [:]
        for (oldKey, newKey) in legacyInputMapping {
            if let value = legacy[oldKey] {
                newInputs[newKey] = value
            }
        }
        // パラメータ名が変わらないものはそのままコピー
        for key in ["q", "Ti_Te"] {
            if let value = legacy[key] {
                newInputs[key] = value
            }
        }
        return newInputs
    }
}
```

## マイグレーション手順

1. ✅ **Python APIの検証** (完了)
   - `verify_python_api.py`で新APIの動作確認済み

2. ⏸️ **Swift Wrapperの更新**
   - `FusionSurrogates.swift` - 新API使用
   - `QLKNN+MLX.swift` - パラメータ名更新
   - `TORAXIntegration.swift` - ヘルパー関数更新

3. ⏸️ **テストの更新**
   - BasicAPITests - 新パラメータ名
   - 統合テストの再有効化

4. ⏸️ **ドキュメント更新**
   - README.md
   - TORAX_INTEGRATION.md
   - IMPLEMENTATION_NOTES.md
   - Examples.md

5. ⏸️ **swift-TORAXとの調整**
   - 既存コードが旧パラメータ名を使用している場合
   - エイリアス/変換関数の提供を検討

## 参考資料

- **新API検証スクリプト:** `verify_python_api.py`
- **モデルファイル:** `fusion_surrogates/fusion_surrogates/qlknn/models/qlknn_7_11.qlknn`
- **モデルREADME:** `fusion_surrogates/fusion_surrogates/qlknn/models/README.md`
- **Python実装:** `fusion_surrogates/fusion_surrogates/qlknn/qlknn_model.py`

## 注意事項

1. **入力範囲の拡大**: 新APIは旧想定より大幅に広い範囲をサポート (例: Ati 0-150 vs 旧 0-16)
2. **JAX依存**: 新APIの出力は`jax.Array`型（numpy互換）
3. **バッチ処理**: 入力は常に2D配列 `(batch_size, num_features)`
4. **正規化**: モデル内部で自動的に入力正規化・出力非正規化を実行

---

**更新日:** 2025-10-18
**fusion_surrogates version:** 最新（Git submodule）
**検証済みモデル:** qlknn_7_11_v1
