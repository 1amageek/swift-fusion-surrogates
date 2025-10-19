# FusionSurrogates for Swift

A Swift wrapper for Google DeepMind's [fusion_surrogates](https://github.com/google-deepmind/fusion_surrogates) Python library, designed for integration with [swift-TORAX](https://github.com/google-deepmind/torax) - a Swift implementation of the TORAX tokamak plasma transport simulator.

[![Swift 6.0+](https://img.shields.io/badge/Swift-6.0+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/platform-macOS%2013.3+-lightgrey.svg)](https://www.apple.com/macos/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Overview

**FusionSurrogates** provides a Swift interface to neural network surrogate models for turbulent transport in fusion plasmas. It enables fast, GPU-accelerated transport coefficient predictions for tokamak simulations on Apple Silicon.

**⚠️ Development Status:** This project is in active development and not yet production-ready.

### What is TORAX?

[TORAX](https://torax.readthedocs.io/) (TOkamak Rapid Advanced eXecution) is Google DeepMind's differentiable tokamak core transport simulator, originally implemented in Python/JAX. **swift-TORAX** is a Swift reimplementation that uses Apple's MLX framework instead of JAX, optimized for Apple Silicon.

FusionSurrogates provides the transport model layer, enabling swift-TORAX to use fast neural network predictions instead of expensive first-principles simulations.

```
┌─────────────────────────────────────────────────────────┐
│  TORAX (Python/JAX) - Google DeepMind                   │
│  https://torax.readthedocs.io/                          │
└─────────────────────────────────────────────────────────┘
                         ↓ reimplemented in Swift
┌─────────────────────────────────────────────────────────┐
│  swift-TORAX (Swift/MLX) - Apple Silicon optimized      │
│  Tokamak transport simulator                            │
└─────────────────────────────────────────────────────────┘
                         ↓ uses
┌─────────────────────────────────────────────────────────┐
│  swift-fusion-surrogates (this package)                 │
│  Swift wrapper for fusion_surrogates                    │
└─────────────────────────────────────────────────────────┘
                         ↓ wraps
┌─────────────────────────────────────────────────────────┐
│  fusion_surrogates (Python) - Google DeepMind           │
│  Neural network surrogate models (QLKNN, TGLFNN)        │
└─────────────────────────────────────────────────────────┘
```

## Features

- ✅ **Swift-native API** - Type-safe interface using `MLXArray` (Float32 only)
- ✅ **GPU-accelerated** - MLX-native gradient computation (10-100× faster)
- ✅ **Automatic validation** - Shape checking, NaN/Inf detection, bounds verification
- ✅ **QLKNN support** - QuaLiKiz neural network for ITG/TEM/ETG turbulence
- ✅ **swift-TORAX integration** - Helpers for `EvaluatedArray` conversion
- ✅ **Upstream tracking** - Uses fusion_surrogates as git submodule for automatic updates
- ✅ **Float32 only** - No Double/Float64 in codebase; optimized for GPU efficiency

## Quick Start

### Prerequisites

- Swift 6.0 or later
- macOS 13.3 or later (for MLX support)
- Python 3.12+ with fusion_surrogates installed

### Installation

#### 1. Install Python dependencies

```bash
pip install fusion-surrogates
```

#### 2. Add to your Swift package

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/your-org/swift-fusion-surrogates", branch: "main")
]
```

#### 3. Import and use

```swift
import FusionSurrogates
import MLX

// Initialize QLKNN model
let qlknn = try QLKNN(modelName: "qlknn_7_11_v1")

// Prepare inputs (normalized gradients, safety factor, etc.)
let inputs: [String: MLXArray] = [
    "Ati": MLXArray([5.0, 5.5, 6.0], [3]),        // Ion temp gradient (R/L_Ti)
    "Ate": MLXArray([5.0, 5.5, 6.0], [3]),        // Electron temp gradient (R/L_Te)
    "Ane": MLXArray([1.0, 1.2, 1.4], [3]),        // Electron density gradient
    "Ani": MLXArray([1.0, 1.2, 1.4], [3]),        // Ion density gradient
    "q": MLXArray([2.0, 2.5, 3.0], [3]),          // Safety factor
    "smag": MLXArray([1.0, 1.2, 1.4], [3]),       // Magnetic shear
    "x": MLXArray([0.3, 0.35, 0.4], [3]),         // Inverse aspect ratio (r/R)
    "Ti_Te": MLXArray([1.0, 1.0, 1.0], [3]),      // Temperature ratio
    "LogNuStar": MLXArray([-10.0, -9.5, -9.0], [3]), // Collisionality
    "normni": MLXArray([1.0, 1.0, 1.0], [3])      // Density ratio (ni/ne)
]

// Run prediction
let outputs = try qlknn.predict(inputs)

// Access transport coefficients
let efiITG = outputs["efiITG"]!      // Ion heat flux (ITG mode)
let efeTEM = outputs["efeTEM"]!      // Electron heat flux (TEM mode)
let efeETG = outputs["efeETG"]!      // Electron heat flux (ETG mode)

// Combine fluxes for TORAX
let combined = TORAXIntegration.combineFluxes(outputs)
let chiIon = combined["chi_ion"]!           // Total ion heat diffusivity
let chiElectron = combined["chi_electron"]! // Total electron heat diffusivity
```

## Usage with swift-TORAX

> **⚠️ Required Data:** Integration with swift-TORAX requires **poloidal flux** (ψ) and **toroidal magnetic field** (B_tor) data. See [TORAX_INTEGRATION.md](TORAX_INTEGRATION.md) for details.

### Transport Model Integration

```swift
import FusionSurrogates
import MLX

public struct QLKNNTransportModel: TransportModel {
    private let qlknn: QLKNN
    private let majorRadius: Float
    private let minorRadius: Float
    private let toroidalField: Float

    public init(
        majorRadius: Float,
        minorRadius: Float,
        toroidalField: Float
    ) throws {
        self.qlknn = try QLKNN(modelName: "qlknn_7_11_v1")
        self.majorRadius = majorRadius
        self.minorRadius = minorRadius
        self.toroidalField = toroidalField
    }

    public func computeCoefficients(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: TransportParameters
    ) -> TransportCoefficients {
        // 1. Extract MLXArray from EvaluatedArray
        let Te = profiles.electronTemperature.value
        let Ti = profiles.ionTemperature.value
        let ne = profiles.electronDensity.value

        // 2. Build QLKNN inputs
        // ⚠️ REQUIRED: poloidalFlux (ψ) and toroidalField (B_tor)
        let inputs = QLKNN.buildInputs(
            electronTemperature: Te,
            ionTemperature: Ti,
            electronDensity: ne,
            ionDensity: ne,
            poloidalFlux: profiles.poloidalFlux.value,  // Required: ψ(r)
            radius: geometry.rho.value,
            majorRadius: majorRadius,
            minorRadius: minorRadius,
            toroidalField: toroidalField                // Required: B_tor
        )

        // 3. Predict transport coefficients
        let outputs = try! qlknn.predict(inputs)
        let combined = TORAXIntegration.combineFluxes(outputs)

        // 4. Convert to EvaluatedArray (Sendable for swift-TORAX)
        let evaluated = EvaluatedArray.evaluatingBatch([
            combined["chi_ion"]!,
            combined["chi_electron"]!,
            combined["particle_diffusivity"]!,
            combined["convection_velocity"]!
        ])

        return TransportCoefficients(
            chiIon: evaluated[0],
            chiElectron: evaluated[1],
            particleDiffusivity: evaluated[2],
            convectionVelocity: evaluated[3]
        )
    }
}
```

### EvaluatedArray Conversion

**Important:** FusionSurrogates returns `MLXArray`, not `EvaluatedArray`. Conversion is swift-TORAX's responsibility.

```swift
// ✅ Efficient: Batch evaluation
let evaluated = EvaluatedArray.evaluatingBatch([
    mlxArray1,
    mlxArray2,
    mlxArray3
])

// ❌ Inefficient: Multiple eval() calls
let eval1 = EvaluatedArray(mlxArray1)  // eval() called
let eval2 = EvaluatedArray(mlxArray2)  // eval() called
let eval3 = EvaluatedArray(mlxArray3)  // eval() called
```

See [`TORAX_INTEGRATION.md`](TORAX_INTEGRATION.md) for complete integration guide.

## API Overview

### QLKNN Input Parameters

| Parameter | Description | Valid Range | Physical Meaning |
|-----------|-------------|-------------|------------------|
| `Ati` | R/L_Ti | ≈0 to 150 | Normalized ion temperature gradient |
| `Ate` | R/L_Te | ≈0 to 150 | Normalized electron temperature gradient |
| `Ane` | R/L_ne | -5 to 110 | Normalized electron density gradient |
| `Ani` | R/L_ni | -15 to 110 | Normalized ion density gradient |
| `q` | Safety factor | 0.66 to 30 | Magnetic field line winding |
| `smag` | Magnetic shear (s_hat) | -1 to 40 | Rate of change of q |
| `x` | Inverse aspect ratio (r/R) | 0.1 to 0.95 | Radial position normalized |
| `Ti_Te` | Temperature ratio | 0.25 to 2.5 | Ion/electron temperature |
| `LogNuStar` | Logarithmic collisionality | -5 to 0.48 | Collision frequency |
| `normni` | Density ratio (ni/ne) | 0.5 to 1.0 | Ion/electron density |

**Note:** Ranges from QLKNN 7_11_v1 model `config.stats_data`. Values outside these ranges may produce unreliable predictions.

### QLKNN Output Parameters

| Parameter | Description | Physics | Mode |
|-----------|-------------|---------|------|
| `efiITG` | Ion thermal flux | GB units | ITG |
| `efeITG` | Electron thermal flux | GB units | ITG |
| `efeTEM` | Electron thermal flux | GB units | TEM |
| `efeETG` | Electron thermal flux | GB units | ETG |
| `efiTEM` | Ion thermal flux | GB units | TEM |
| `pfeITG` | Particle flux | GB units | ITG |
| `pfeTEM` | Particle flux | GB units | TEM |
| `gamma_max` | Maximum growth rate | 1/s | All |

**Combined outputs (via `TORAXIntegration.combineFluxes`):**
- `chi_ion` = `efiITG` + `efiTEM`
- `chi_electron` = `efeITG` + `efeTEM` + `efeETG`
- `particle_flux` = `pfeITG` + `pfeTEM`
- `growth_rate` = `gamma_max`

## GPU Acceleration

FusionSurrogates includes MLX-native gradient computation for 10-100× speedup:

```swift
// GPU-accelerated gradient using MLX slicing
let rLnT = TORAXIntegration.computeNormalizedGradient(
    profile: T,
    radius: r,
    majorRadius: R
)
// Uses centered differences for interior points
// Forward/backward differences at boundaries
// Fully vectorized on GPU
```

## Documentation

- **[TORAX_INTEGRATION.md](TORAX_INTEGRATION.md)** ⭐ - Complete swift-TORAX integration guide (essential reading)
- **[API_MIGRATION.md](API_MIGRATION.md)** - API reference and parameter mappings
- **[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)** - Technical details and design decisions
- **[TESTING.md](TESTING.md)** - Testing guide and manual verification
- **[CLAUDE.md](CLAUDE.md)** - Project guidance for AI assistants

## Architecture

### Type System

```swift
// FusionSurrogates uses standard MLX types
public func predict(_ inputs: [String: MLXArray]) throws -> [String: MLXArray]

// swift-TORAX uses EvaluatedArray (Sendable)
public struct EvaluatedArray: Sendable {
    let value: MLXArray  // Already evaluated
}

// Conversion happens at swift-TORAX layer
let mlxOutputs = try qlknn.predict(inputs)
let combined = TORAXIntegration.combineFluxes(mlxOutputs)
let evaluated = EvaluatedArray.evaluatingBatch(Array(combined.values))
```

### Design Principles

1. **Generic Wrapper** - Not swift-TORAX-specific, uses standard MLX types
2. **Upstream Tracking** - fusion_surrogates as submodule for automatic updates
3. **Type Safety** - Swift's type system prevents runtime errors
4. **GPU Acceleration** - MLX-native operations where possible
5. **Clear Separation** - FusionSurrogates → MLXArray, swift-TORAX → EvaluatedArray
6. **Float32 Only** - All numeric operations use Float32 exclusively
   - ✅ No `Double` or `Float64` anywhere in the codebase
   - ✅ All MLXArray operations use `Float.self` (32-bit)
   - ✅ Optimized for GPU memory bandwidth and cache efficiency

## Testing

```bash
# Run unit tests (no Python/MLX dependencies required)
swift test

# Expected: Test run with 4 tests in 2 suites passed
```

**Integration Tests:** Python/MLX integration tests require environment setup. See [TESTING.md](TESTING.md) for:
- Manual verification scripts (`verify_python_api.py`, `test_new_api_final.py`)
- MLX/Python environment configuration
- Known limitations and workarounds

## Performance

### GPU Acceleration

FusionSurrogates uses **MLX-native gradient computation** for significant performance improvements:

| Operation | Performance | Notes |
|-----------|-------------|-------|
| Gradient computation | **10-100× faster** | GPU (MLX) vs CPU (Swift arrays) |
| Array conversion | ~10-100 μs per array | MLXArray ↔ Python numpy |
| Prediction overhead | ~1-10 ms | Grid size 100-500 cells |
| Total overhead | **<1% of simulation** | PDE solver dominates runtime |

**Key Optimization:** Use `EvaluatedArray.evaluatingBatch()` instead of multiple individual `eval()` calls to enable GPU kernel fusion.

## Requirements

- **Swift:** 6.0 or later
- **Platform:** macOS 13.3 or later (MLX Metal support required)
- **Python:** 3.12 or later
- **Dependencies:**
  - [PythonKit](https://github.com/pvieito/PythonKit) - Python interop
  - [MLX-Swift](https://github.com/ml-explore/mlx-swift) - Array operations (0.29.1+)
  - [fusion_surrogates](https://github.com/google-deepmind/fusion_surrogates) - Python library (pip, v0.4.2+)

**Platform Constraints:**
- ⚠️ **macOS only** - MLX requires Metal (Apple Silicon or Intel with Metal support)
- ⚠️ **Python runtime required** - Not a pure Swift solution
- ⚠️ **No Linux/Windows support** - Due to MLX Metal dependency

## API Information

This package uses the `QLKNNModel` API from fusion_surrogates.

**Implementation:**
- Model loading: `QLKNNModel.load_default_model()`
- Input format: 2D numpy array (batch_size, 10 features)
- Output format: Dict[str, jax.Array]

See [API_MIGRATION.md](API_MIGRATION.md) for technical details on the Python API.

## Required Data for Integration

When integrating with swift-TORAX, `TORAXIntegration.buildInputs()` requires:

**From Profiles:**
- Electron temperature: `Te(r)`
- Ion temperature: `Ti(r)`
- Electron density: `ne(r)`
- Ion density: `ni(r)`

**From Geometry/Magnetic Configuration:**
- **Poloidal flux: `ψ(r)`** ⚠️ Required for safety factor calculation
- **Toroidal field: `B_tor`** ⚠️ Required for safety factor calculation
- Major radius: `R0`
- Minor radius: `a`
- Radius grid: `r`

**Important:** If your `Geometry` type does not include `poloidalFlux` and `toroidalField`, you must extend it or provide these values separately. See [REVIEW_ANALYSIS.md](REVIEW_ANALYSIS.md) for details.

## Related Projects

- **[TORAX](https://torax.readthedocs.io/)** - Original Python/JAX tokamak simulator (Google DeepMind)
- **[swift-TORAX](https://github.com/google-deepmind/torax)** - Swift/MLX reimplementation for Apple Silicon
- **[fusion_surrogates](https://github.com/google-deepmind/fusion_surrogates)** - Neural network surrogate models (Google DeepMind)
- **[MLX](https://github.com/ml-explore/mlx)** - Array framework for Apple Silicon
- **[QuaLiKiz](http://qualikiz.com/)** - Quasilinear gyrokinetic transport model

## Citation

If you use this package, please cite:

**QLKNN Model:**
```bibtex
@article{vanDePlassche2020,
  title={Fast modeling of turbulent transport in fusion plasmas using neural networks},
  author={van de Plassche, Karel L and others},
  journal={Physics of Plasmas},
  volume={27},
  number={2},
  pages={022310},
  year={2020},
  publisher={AIP Publishing},
  doi={10.1063/1.5134126}
}
```

**TORAX:**
```bibtex
@software{torax2024,
  title={TORAX: Tokamak transport simulation in JAX},
  author={Google DeepMind},
  year={2024},
  url={https://github.com/google-deepmind/torax}
}
```

## Contributing

Contributions welcome! Please:

1. Check existing documentation first
2. Follow Swift API design guidelines
3. Add tests for new features
4. Update documentation

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Google DeepMind for TORAX and fusion_surrogates
- QuaLiKiz Group for QLKNN model weights
- Apple for MLX framework
- Swift community for PythonKit

## Support

- **Documentation:** See [documentation files](.)
- **Issues:** GitHub Issues
- **Questions:** Check [TESTING.md](TESTING.md) and [TORAX_INTEGRATION.md](TORAX_INTEGRATION.md)

---

**Development Status:** 🚧 In active development

This project is currently under development. APIs and implementations may change.

**Current Implementation:**
- Uses fusion_surrogates `QLKNNModel` API
- Parameter names: Ati/Ate (normalized gradients)
- Input ranges: Expanded for QLKNN 7_11_v1 model
- Output: 8 parameters (ITG/TEM/ETG modes)
- Numeric precision: **Float32 exclusively** (no Double/Float64 in codebase)
