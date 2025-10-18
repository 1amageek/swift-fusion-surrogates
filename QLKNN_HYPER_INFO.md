# QLKNN-Hyper とは？

## 概要

**qlknn-hyper** は、QuaLiKiz Neural Network（QLKNN）の**モデルウェイト（学習済みパラメータ）**を提供するリポジトリです。

- **リポジトリ:** https://gitlab.com/qualikiz-group/qlknn-hyper
- **論文:** van de Plassche et al., 2020 (https://doi.org/10.1063/1.5134126)
- **ライセンス:** 無料で一般利用可能

## 位置づけ

```
┌─────────────────────────────────────────────────────────────┐
│ fusion_surrogates (Google DeepMind)                         │
│  - Pythonライブラリ                                          │
│  - モデルのインターフェース・API                             │
│  - 複数のサロゲートモデルをサポート                          │
│    └── QLKNN (QuaLiKiz Neural Network)                      │
│    └── TGLFNN (TGLF Neural Network)                         │
└─────────────────────────────────────────────────────────────┘
                           ↓ 使用
┌─────────────────────────────────────────────────────────────┐
│ qlknn-hyper (QuaLiKiz Group)                                │
│  - 学習済みニューラルネットワークの重み                       │
│  - JSONフォーマット（各物理量ごとに1ファイル）               │
│  - 22個のモデルファイル（各輸送係数・フラックス用）           │
└─────────────────────────────────────────────────────────────┘
                           ↓ ロード
┌─────────────────────────────────────────────────────────────┐
│ swift-fusion-surrogates (このプロジェクト)                  │
│  - fusion_surrogatesのSwiftラッパー                         │
│  - swift-TORAXとの統合                                       │
└─────────────────────────────────────────────────────────────┘
```

## qlknn-hyperの内容

### モデルファイル（22個のJSON）

各ファイルは1つの物理量を予測するニューラルネットワークの重みを含む：

| ファイル名 | 物理量 | 説明 |
|----------|--------|------|
| `efiitg_gb.json` | efi_ITG | Ion thermal flux (ITG mode) |
| `efeitg_gb_div_efiitg_gb.json` | efe_ITG / efi_ITG | Electron-ion heat flux ratio (ITG) |
| `efetem_gb.json` | efe_TEM | Electron thermal flux (TEM mode) |
| `efeetg_gb.json` | efe_ETG | Electron thermal flux (ETG mode) |
| `pfeitg_gb_div_efiitg_gb.json` | pfe_ITG / efi_ITG | Electron particle flux ratio (ITG) |
| `pfetem_gb_div_efetem_gb.json` | pfe_TEM / efe_TEM | Electron particle flux ratio (TEM) |
| `vti*_gb_div_efiitg_gb.json` | vti* / efi_ITG | Ion convection ratios |
| `vte*_gb_div_efetem_gb.json` | vte* / efe_TEM | Electron convection ratios |
| `gam_leq_gb.json` | γ | Growth rate |
| 他13ファイル | 各種フラックス比 | 輸送係数の派生量 |

### JSONファイル構造

```json
{
    "_metadata": {
        "epoch": 2725,
        "best_epoch": 2725,
        "rms_validation": 0.2766
    },
    "feature_names": [
        "Zeff",        // 有効電荷数
        "Ati",         // R/L_Ti (ion temperature gradient)
        "Ate",         // R/L_Te (electron temperature gradient)
        "An",          // R/L_n (density gradient)
        "q",           // 安全係数
        "smag",        // 磁気シア
        "x",           // r/R (逆アスペクト比)
        "Ti_Te",       // 温度比
        "logNustar"    // 対数衝突頻度
    ],
    "target_names": ["efeITG_GB_div_efiITG_GB"],
    "hidden_activation": "tanh",
    "output_activation": "none",
    "feature_min": [...],  // 入力の正規化範囲
    "feature_max": [...],
    "target_min": [...],   // 出力の正規化範囲
    "target_max": [...],
    "prescale_bias": [...],
    "prescale_weights": [...],
    "weights_layer_*": [...],  // ニューラルネットワークの重み
    "biases_layer_*": [...]    // バイアス項
}
```

## fusion_surrogatesとの関係

### fusion_surrogatesの役割

fusion_surrogatesは、qlknn-hyperのモデルファイルを：

1. **ロード**する
2. **正規化/非正規化**する
3. **推論**を実行する
4. **統一されたAPI**を提供する

### 使用例（Python）

```python
from fusion_surrogates.qlknn import qlknn_model

# fusion_surrogatesが内部でqlknn-hyperのJSONをロード
model = qlknn_model.QLKNNModel()

# 予測実行（qlknn-hyperの重みを使用）
inputs = {
    'Zeff': 1.0,
    'Ati': 5.0,    # R/L_Ti
    'Ate': 5.0,    # R/L_Te
    'An': 1.0,     # R/L_n
    'q': 2.0,
    'smag': 1.0,
    'x': 0.3,
    'Ti_Te': 1.0,
    'logNustar': -10.0
}

outputs = model.predict(inputs)
# outputs['efi_ITG'], outputs['efe_TEM'], etc.
```

## swift-fusion-surrogatesへの影響

### 現在の実装

swift-fusion-surrogatesは、fusion_surrogatesをラップしているため、**qlknn-hyperのモデルは自動的に利用可能**です：

```swift
import FusionSurrogates

let qlknn = try QLKNN(modelVersion: "7_11")

// qlknn-hyperのモデルファイルが
// fusion_surrogatesを通じて自動的に使用される
let outputs = try qlknn.predict(inputs)
```

### 入力パラメータのマッピング

| QLKNN入力名 | qlknn-hyper名 | 説明 |
|------------|--------------|------|
| `R_L_Te` | `Ate` | Normalized electron temp gradient |
| `R_L_Ti` | `Ati` | Normalized ion temp gradient |
| `R_L_ne` | `An` | Normalized density gradient |
| `q` | `q` | Safety factor |
| `s_hat` | `smag` | Magnetic shear |
| `r_R` | `x` | Inverse aspect ratio |
| `Ti_Te` | `Ti_Te` | Temperature ratio |
| `log_nu_star` | `logNustar` | Collisionality |

**注:** `Zeff`（有効電荷数）と`R_L_ni`は追加パラメータ

## まとめ

### qlknn-hyperは何か？

- ✅ **学習済みニューラルネットワークの重み**（22個のJSONファイル）
- ✅ QuaLiKiz乱流輸送シミュレーションのサロゲートモデル
- ✅ fusion_surrogatesが内部で使用する
- ✅ 研究論文で検証済み（2020年論文）

### どう使うか？

**直接使用はしない** - fusion_surrogatesを通じて間接的に使用：

```
swift-fusion-surrogates
    ↓ ラップ
fusion_surrogates (Python)
    ↓ ロード
qlknn-hyper (JSONモデル)
```

### 重要なポイント

1. **モデルファイルは別リポジトリ**: fusion_surrogatesはコードのみ、qlknn-hyperは重みのみ
2. **自動ダウンロード**: fusion_surrogatesが必要に応じてqlknn-hyperをダウンロード
3. **引用が必要**: 使用時は論文を引用する必要がある

### 参考文献

- van de Plassche, K. L., et al. "Fast modeling of turbulent transport in fusion plasmas using neural networks." Physics of Plasmas 27.2 (2020): 022310.
- DOI: https://doi.org/10.1063/1.5134126

### 関連リンク

- GitLab: https://gitlab.com/qualikiz-group/qlknn-hyper
- QuaLiKiz: http://qualikiz.com/
- fusion_surrogates: https://github.com/google-deepmind/fusion_surrogates

## swift-fusion-surrogatesでの確認

qlknn-hyperのモデルが正しくロードされているか確認：

```swift
import FusionSurrogates

// fusion_surrogatesがqlknn-hyperのJSONを自動的にロード
let qlknn = try QLKNN(modelVersion: "7_11")

// 入力パラメータ名を確認
print(QLKNN.inputParameterNames)
// ["R_L_Te", "R_L_Ti", "R_L_ne", "R_L_ni", "q", "s_hat", "r_R", "Ti_Te", "log_nu_star", "ni_ne"]

// 出力パラメータ名を確認
print(QLKNN.outputParameterNames)
// ["chi_ion_itg", "chi_electron_tem", "chi_electron_etg", "particle_flux", "growth_rate"]
```

これらのパラメータは、qlknn-hyperのJSONファイルから来ています。
