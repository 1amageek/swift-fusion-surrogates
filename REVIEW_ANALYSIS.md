# レビュー指摘の考察 - swift-fusion-surrogates の視点から

**日付:** 2025-10-18
**対象:** swift-TORAXプロジェクトへのレビュー指摘
**分析者:** swift-fusion-surrogatesプロジェクトの視点

---

## 前提理解

このレビューはswift-TORAXプロジェクトに対する指摘と推測されます。swift-fusion-surrogatesは**依存ライブラリ**として使われる立場であり、レビュー指摘の多くはswift-TORAX側で対処すべき事項です。

しかし、swift-fusion-surrogatesが提供する**インターフェース設計**や**ドキュメント**がswift-TORAXの実装を誤誘導している可能性があるため、各指摘を分析します。

---

## 指摘1: Python連携の実現可能性

### レビュー内容
> 重大: 計画では swift-fusion-surrogates と PythonKit を Package.swift の依存に加える前提ですが、現在のパッケージには Python 連携の土台が一切なく、macOS 以外での動作や SwiftPM の sandbox 制約も考慮されていません。Python 環境に依存する方針なら、サポート OS・CI・デプロイ手順まで含めて実現可能性を整理した上で実装計画を作り直す必要があります。

### swift-fusion-surrogatesの現状

✅ **既に対応済み:**
```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/pvieito/PythonKit.git", branch: "master"),
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.1")
]

platforms: [
    .macOS("13.3"),  // MLX requires macOS 13.3+
    .iOS(.v16)        // iOS support (limited)
]
```

✅ **Python連携の土台は完成:**
- `FusionSurrogates.swift`: PythonKit使用
- `verify_python_api.py`: Python環境検証スクリプト
- `test_new_api_final.py`: Python統合テスト

⚠️ **制約事項を明示:**
- **サポートOS**: macOS 13.3+ (MLX Metal要件)
- **Python要件**: Python 3.12+ + fusion_surrogates 0.4.2+
- **デプロイ**: ユーザーが `pip install fusion-surrogates` 必要

### swift-TORAXへの推奨対応

```swift
// swift-TORAX の Package.swift
dependencies: [
    .package(url: "https://github.com/your-org/swift-fusion-surrogates", branch: "main")
]

// README.md に追記すべき内容
## Requirements

- macOS 13.3+ (MLX Metal support required)
- Python 3.12+
- fusion_surrogates 0.4.2+

## Setup

```bash
# Install Python dependencies
pip install fusion-surrogates

# Build swift-TORAX
swift build
```
```

### swift-fusion-surrogatesで改善すべき点

📝 **ドキュメント強化:**
- `README.md` に制約事項を明示済み（✅）
- CI/CDパイプライン例は未提供（⚠️）

**推奨追加:**
```yaml
# .github/workflows/test.yml の例
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: macos-13  # macOS 13.3+ required
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install Python dependencies
        run: pip install fusion-surrogates
      - name: Run Swift tests
        run: swift test
```

---

## 指摘2: MLXArray の逐次アクセス問題

### レビュー内容
> 重大: QLKNNInputBuilder の疑似コードは for i in 1..<(n-1) など MLXArray を添字で直接書き換えていますが、MLX はテンソル演算を想定しており、こうした逐次アクセスはコンパイルも実行もできません。既存コードが徹底してベクトル化されているのと同様に、勾配計算は差分演算をベクトル式で記述する設計に改めるべきです。

### swift-fusion-surrogatesの現状

✅ **既に修正済み:**

`TORAXIntegration.swift:161-189` のgradient計算は完全にベクトル化されています：

```swift
private static func gradient(_ f: MLXArray, _ x: MLXArray) -> MLXArray {
    let n = f.shape[0]

    // ❌ こういうコードは書いていない
    // for i in 1..<(n-1) {
    //     grad[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
    // }

    // ✅ 実際のコード: MLXスライシングでベクトル化
    let fNext = f[2 ..< n]           // f[2], f[3], ..., f[n-1]
    let fPrev = f[0 ..< (n - 2)]     // f[0], f[1], ..., f[n-3]
    let xNext = x[2 ..< n]
    let xPrev = x[0 ..< (n - 2)]

    let gradInterior = (fNext - fPrev) / (xNext - xPrev)  // ベクトル演算

    // concatenated() で結合
    return concatenated([gradFirst, gradInterior, gradLast], axis: 0)
}
```

### 疑似コードの問題

レビューが指摘している「疑似コード」は、おそらく**ドキュメントや計画段階**のコードと思われます。

**swift-fusion-surrogatesのドキュメントでは:**
- `DESIGN_SUMMARY.md`: 実装詳細なし（概念のみ）
- `IMPLEMENTATION_NOTES.md`: MLX-native実装を明記
- `TORAX_INTEGRATION.md`: 使用例のみ

⚠️ **問題の可能性:**
- swift-TORAXのドキュメントに古い疑似コードが残っている？
- 設計段階の文書が更新されていない？

### swift-TORAXへの推奨対応

**悪い例（レビュー指摘）:**
```swift
// ❌ これは動かない
func computeGradient(_ f: MLXArray, _ x: MLXArray) -> MLXArray {
    var grad = MLXArray.zeros(f.shape)
    for i in 1..<(n-1) {
        grad[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])  // エラー
    }
    return grad
}
```

**良い例（swift-fusion-surrogatesパターン）:**
```swift
// ✅ TORAXIntegration.gradient() を使う
let rLnT = TORAXIntegration.computeNormalizedGradient(
    profile: T,
    radius: r,
    majorRadius: R
)

// または直接ベクトル演算
let df = T[2..<n] - T[0..<(n-2)]
let dx = r[2..<n] - r[0..<(n-2)]
let grad_interior = df / dx
```

---

## 指摘3: Geometry型のデータ不足

### レビュー内容
> 重大: 既存の Geometry 型は q や Bp などを保持していないため、文書で提案されている Geometry.computeSafetyFactor や磁気シア計算は現状のデータ構造では実装できません。追加データをどこから取得するのか、あるいは入力として受け取るのかを先に明確化してください。

### swift-fusion-surrogatesの現状

⚠️ **これはswift-TORAXの設計問題:**

swift-fusion-surrogatesは**Geometry型を定義していません**。提供しているのは：

```swift
// TORAXIntegration.swift
public static func buildInputs(
    electronTemperature: MLXArray,
    ionTemperature: MLXArray,
    electronDensity: MLXArray,
    ionDensity: MLXArray,
    poloidalFlux: MLXArray,      // ← これが必要
    radius: MLXArray,
    majorRadius: Float,
    minorRadius: Float,
    toroidalField: Float         // ← これが必要
) -> [String: MLXArray] {
    // q を計算
    let q = computeSafetyFactor(
        poloidalFlux: poloidalFlux,
        radius: radius,
        minorRadius: minorRadius,
        majorRadius: majorRadius,
        toroidalField: toroidalField
    )

    // 磁気シアを計算
    let sHat = computeMagneticShear(
        safetyFactor: q,
        radius: radius
    )

    return [...]
}
```

**必要なデータ:**
- `poloidalFlux` (ポロイダル磁束)
- `toroidalField` (トロイダル磁場)

これらはswift-TORAXの`Geometry`型が保持すべきデータです。

### swift-TORAXへの推奨対応

**問題のある設計:**
```swift
// ❌ Geometry型にデータがない
struct Geometry {
    let rho: EvaluatedArray      // 正規化半径
    // q や Bp がない！
}

// どうやってQLKNN入力を作る？
let inputs = QLKNN.buildInputs(
    geometry: geometry,  // データ不足でエラー
    ...
)
```

**推奨される設計:**

**オプション A: Geometry型を拡張**
```swift
struct Geometry {
    let rho: EvaluatedArray
    let poloidalFlux: EvaluatedArray    // ← 追加
    let toroidalField: Float             // ← 追加
    let majorRadius: Float
    let minorRadius: Float
}

// FusionSurrogatesの使用
let inputs = TORAXIntegration.buildInputs(
    electronTemperature: profiles.Te.value,
    ionTemperature: profiles.Ti.value,
    electronDensity: profiles.ne.value,
    ionDensity: profiles.ni.value,
    poloidalFlux: geometry.poloidalFlux.value,  // ← 追加データ
    radius: geometry.rho.value,
    majorRadius: geometry.majorRadius,
    minorRadius: geometry.minorRadius,
    toroidalField: geometry.toroidalField       // ← 追加データ
)
```

**オプション B: 別パラメータで受け取る**
```swift
struct MagneticFieldData {
    let poloidalFlux: EvaluatedArray
    let toroidalField: Float
}

public func computeCoefficients(
    profiles: CoreProfiles,
    geometry: Geometry,
    magneticField: MagneticFieldData,  // ← 別構造体
    params: TransportParameters
) -> TransportCoefficients {
    let inputs = TORAXIntegration.buildInputs(
        // ...
        poloidalFlux: magneticField.poloidalFlux.value,
        toroidalField: magneticField.toroidalField
    )
}
```

### swift-fusion-surrogatesで改善すべき点

📝 **ドキュメント明確化:**

`TORAX_INTEGRATION.md` に必要データを明記：

```markdown
## Required Data for QLKNN Input Construction

`TORAXIntegration.buildInputs()` requires the following data:

**From Profiles:**
- Electron temperature: `Te(r)`
- Ion temperature: `Ti(r)`
- Electron density: `ne(r)`
- Ion density: `ni(r)`

**From Geometry/Magnetic Field:**
- Poloidal flux: `ψ(r)` - **Required for safety factor calculation**
- Toroidal field: `B_tor` - **Required for safety factor calculation**
- Major radius: `R0`
- Minor radius: `a`
- Radius grid: `r`

**Computed Internally:**
- Safety factor: `q(r) = f(ψ, r, R0, a, B_tor)`
- Magnetic shear: `s_hat(r) = (r/q) dq/dr`

If your Geometry type does not include `poloidalFlux` and `toroidalField`,
you must extend it or provide these values separately.
```

---

## 指摘4: TransportConfigの拡張方法

### レビュー内容
> 高: TransportConfig に QLKNN 用フィールドを追加する案は、現在のイニシャライザ群や Codable テスト（TransportConfigTests など）と整合しません。互換性を保つ拡張方法（例: QLKNNTransportConfig を別構造体にする等）を検討する必要があります。

### swift-fusion-surrogatesの現状

⚠️ **これはswift-TORAXの設計問題:**

swift-fusion-surrogatesは`TransportConfig`を定義していません。

### swift-TORAXへの推奨対応

**問題のある設計:**
```swift
// ❌ 既存のTransportConfigを直接変更
struct TransportConfig: Codable {
    let chi_constant: Float
    let d_constant: Float

    // ← これを追加すると既存のJSONが読めなくなる
    let qlknnModelVersion: String?
    let qlknnMajorRadius: Float?
}
```

**推奨される設計:**

**オプション A: enum で transport model を切り替え**
```swift
enum TransportModelConfig: Codable {
    case constant(ConstantTransportConfig)
    case qlknn(QLKNNTransportConfig)

    struct ConstantTransportConfig: Codable {
        let chi: Float
        let d: Float
    }

    struct QLKNNTransportConfig: Codable {
        let modelVersion: String = "qlknn_7_11_v1"
        let majorRadius: Float
        let minorRadius: Float
        let toroidalField: Float
    }
}

struct TransportConfig: Codable {
    let model: TransportModelConfig
}
```

**オプション B: Optional でラップ（後方互換性維持）**
```swift
struct TransportConfig: Codable {
    // 既存フィールド（必須）
    let defaultModel: String  // "constant" or "qlknn"

    // Constant model用（Optional）
    let constantParams: ConstantParams?

    // QLKNN model用（Optional）
    let qlknnParams: QLKNNParams?

    struct QLKNNParams: Codable {
        let modelVersion: String
        let majorRadius: Float
        let minorRadius: Float
        let toroidalField: Float
    }
}

// 使用例
if let qlknnParams = config.qlknnParams {
    let model = try QLKNN(modelName: qlknnParams.modelVersion)
    // ...
}
```

### swift-fusion-surrogatesで改善すべき点

📝 **ドキュメントに設定例を追加:**

`TORAX_INTEGRATION.md` に設定ファイル例：

```markdown
## Configuration Example

### JSON Configuration
```json
{
  "transport": {
    "model": "qlknn",
    "qlknn": {
      "modelVersion": "qlknn_7_11_v1",
      "majorRadius": 6.2,
      "minorRadius": 2.0,
      "toroidalField": 5.3
    }
  }
}
```

### Swift Configuration
```swift
struct SimulationConfig: Codable {
    let transport: TransportModelConfig
}

enum TransportModelConfig: Codable {
    case qlknn(QLKNNConfig)

    struct QLKNNConfig: Codable {
        let modelVersion: String
        let majorRadius: Float
        let minorRadius: Float
        let toroidalField: Float
    }
}
```
```

---

## 総合推奨事項

### swift-fusion-surrogatesで対応すべき事項

1. ✅ **指摘1 (Python連携)**: 既に対応済み、CI例をドキュメント追加推奨
2. ✅ **指摘2 (ベクトル化)**: 既に対応済み、実装は正しい
3. ⚠️ **指摘3 (Geometry)**: ドキュメント明確化が必要
4. ⚠️ **指摘4 (Config)**: 設定例をドキュメント追加推奨

### swift-TORAXで対応すべき事項

1. ❌ **指摘1**: Python環境のセットアップ手順をREADMEに追加
2. ❌ **指摘2**: 古い疑似コードをドキュメントから削除
3. ❌ **指摘3**: Geometry型に`poloidalFlux`と`toroidalField`を追加
4. ❌ **指摘4**: TransportConfig設計を見直し（enumまたはOptional）

### 今後の改善アクション

**swift-fusion-surrogatesでの対応:**

```bash
# 1. ドキュメント更新
- TORAX_INTEGRATION.md に必要データ明記
- README.md にCI/CD例追加
- 設定ファイル例追加

# 2. 検証スクリプト追加
- check_swift_torax_compatibility.sh
  → swift-TORAXのGeometry型をチェック
```

**swift-TORAXでの対応（推奨）:**

```bash
# 1. Geometry型拡張
- poloidalFlux プロパティ追加
- toroidalField プロパティ追加

# 2. TransportConfig見直し
- enum TransportModelConfig 導入
- 後方互換性を保つマイグレーション

# 3. ドキュメント整理
- 古い疑似コード削除
- FusionSurrogates使用例更新
```

---

## 結論

レビュー指摘は**主にswift-TORAXプロジェクト側の設計問題**を指摘しています。

**swift-fusion-surrogatesの状態:**
- ✅ Python連携: 完成
- ✅ ベクトル化: 完成（MLX-nativeで実装済み）
- ⚠️ ドキュメント: 一部明確化が必要

**swift-TORAXで対応が必要:**
- ❌ Geometry型のデータ不足
- ❌ TransportConfigの設計
- ❌ ドキュメントの古い疑似コード

swift-fusion-surrogatesは**ライブラリとして正しく実装されている**が、**swift-TORAXとの統合ポイントのドキュメント**をより明確にすることで、このようなレビュー指摘を未然に防げます。
