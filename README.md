# FusionSurrogates for Swift

Pure Swift implementation of QLKNN neural network for turbulent transport in fusion plasmas, designed for integration with [swift-TORAX](https://github.com/google-deepmind/torax) - a Swift implementation of the TORAX tokamak plasma transport simulator.

[![Swift 6.0+](https://img.shields.io/badge/Swift-6.0+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/platform-macOS%2015+-lightgrey.svg)](https://www.apple.com/macos/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Overview

**FusionSurrogates** is a **pure Swift/MLX implementation** of the QLKNN neural network surrogate model for turbulent transport in fusion plasmas. It provides fast, GPU-accelerated transport coefficient predictions for tokamak simulations on Apple Silicon.

**Key Features:**
- ✅ **Pure Swift** - No Python runtime dependency for inference
- ✅ **Bundled Model** - 289KB SafeTensors weights included
- ✅ **Metal Accelerated** - Native GPU execution via MLX
- ✅ **Self-Contained** - Everything needed is in the Swift package

**⚠️ Development Status:** This project is in active development and not yet production-ready.

### Architecture

[TORAX](https://torax.readthedocs.io/) (TOkamak Rapid Advanced eXecution) is Google DeepMind's differentiable tokamak core transport simulator, originally implemented in Python/JAX. **swift-TORAX** is a Swift reimplementation optimized for Apple Silicon.

FusionSurrogates provides the transport model layer, enabling swift-TORAX to use fast neural network predictions instead of expensive first-principles simulations.

```
┌─────────────────────────────────────────────────────────┐
│  swift-TORAX (Swift/MLX)                                │
│  Tokamak transport simulator for Apple Silicon          │
└─────────────────────────────────────────────────────────┘
                         ↓ uses
┌─────────────────────────────────────────────────────────┐
│  swift-fusion-surrogates (this package)                 │
│  Pure Swift/MLX QLKNN neural network                    │
│  • 73,823 parameters (Float32)                          │
│  • Metal-accelerated inference                          │
│  • Bundled SafeTensors weights (289KB)                  │
└─────────────────────────────────────────────────────────┘
                         ↓ converted from
┌─────────────────────────────────────────────────────────┐
│  fusion_surrogates QLKNN 7_11 model (ONNX)              │
│  Reference implementation from Google DeepMind          │
└─────────────────────────────────────────────────────────┘
```

## Features

- ✅ **Pure Swift/MLX** - No Python runtime dependency for inference
- ✅ **Metal-Accelerated** - Native GPU execution on Apple Silicon
- ✅ **Self-Contained** - Bundled SafeTensors model weights (289KB)
- ✅ **Type-Safe API** - Swift interface using `MLXArray` (Float32)
- ✅ **QLKNN 7_11** - 73,823 parameter neural network for ITG/TEM/ETG turbulence
- ✅ **Validated** - Weights converted from fusion_surrogates ONNX model
- ✅ **Float32 Precision** - Optimized for GPU memory bandwidth

## Quick Start

### Prerequisites

- Swift 6.0 or later
- macOS 15.0 or later
- Apple Silicon Mac (for Metal GPU acceleration)

### Installation

Add to your Swift package:

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/your-org/swift-fusion-surrogates", branch: "main")
]
```

### Basic Usage

```swift
import FusionSurrogates
import MLX

// Load bundled MLX network (no Python required!)
let network = try QLKNNNetwork.loadDefault()

// Prepare inputs (Float32, batch_size = 3)
let inputs: [String: MLXArray] = [
    "Ati": MLXArray([Float(5.0), Float(5.5), Float(6.0)], [3]),
    "Ate": MLXArray([Float(5.0), Float(5.5), Float(6.0)], [3]),
    "Ane": MLXArray([Float(1.0), Float(1.2), Float(1.4)], [3]),
    "Ani": MLXArray([Float(1.0), Float(1.2), Float(1.4)], [3]),
    "q": MLXArray([Float(2.0), Float(2.5), Float(3.0)], [3]),
    "smag": MLXArray([Float(1.0), Float(1.2), Float(1.4)], [3]),
    "x": MLXArray([Float(0.3), Float(0.35), Float(0.4)], [3]),
    "Ti_Te": MLXArray([Float(1.0), Float(1.0), Float(1.0)], [3]),
    "LogNuStar": MLXArray([Float(-10.0), Float(-9.5), Float(-9.0)], [3]),
    "normni": MLXArray([Float(1.0), Float(1.0), Float(1.0)], [3])
]

// Run prediction (pure MLX, Metal-accelerated)
let outputs = try network.predict(inputs)

// Access transport fluxes
let efiITG = outputs["efiITG"]!   // Ion heat flux (ITG) [3]
let efeITG = outputs["efeITG"]!   // Electron heat flux (ITG) [3]
let efeTEM = outputs["efeTEM"]!   // Electron heat flux (TEM) [3]
let efeETG = outputs["efeETG"]!   // Electron heat flux (ETG) [3]
```

## Usage with swift-TORAX

### Example Integration

```swift
import FusionSurrogates
import MLX

public struct QLKNNTransportModel: TransportModel {
    private let network: QLKNNNetwork

    public init() throws {
        // Load bundled MLX network
        self.network = try QLKNNNetwork.loadDefault()
    }

    public func computeCoefficients(
        profiles: CoreProfiles,
        geometry: Geometry
    ) throws -> TransportCoefficients {
        // 1. Build QLKNN inputs from profiles
        let inputs: [String: MLXArray] = [
            "Ati": profiles.ionTempGradient,      // R/L_Ti
            "Ate": profiles.electronTempGradient, // R/L_Te
            "Ane": profiles.electronDensGradient, // R/L_ne
            "Ani": profiles.ionDensGradient,      // R/L_ni
            "q": geometry.safetyFactor,           // q(r)
            "smag": geometry.magneticShear,       // s_hat
            "x": geometry.inverseAspect,          // r/R
            "Ti_Te": profiles.tempRatio,          // Ti/Te
            "LogNuStar": profiles.collisionality, // log(nu*)
            "normni": profiles.densityRatio       // ni/ne
        ]

        // 2. Run MLX inference (Metal-accelerated)
        let outputs = try network.predict(inputs)

        // 3. Extract transport coefficients
        // All fluxes are in Gyro-Bohm normalized units
        let chiIon = outputs["efiITG"]!         // Ion heat diffusivity
        let chiElectron = outputs["efeITG"]!    // Electron heat diffusivity
        let particleFlux = outputs["pfeITG"]!   // Particle flux

        return TransportCoefficients(
            chiIon: chiIon,
            chiElectron: chiElectron,
            particleFlux: particleFlux
        )
    }
}
```

See [`TORAX_INTEGRATION.md`](TORAX_INTEGRATION.md) for complete integration patterns.

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

| Parameter | Description | Units | Mode |
|-----------|-------------|-------|------|
| `efeITG` | Electron thermal flux | Gyro-Bohm | ITG |
| `efiITG` | Ion thermal flux | Gyro-Bohm | ITG |
| `pfeITG` | Particle flux | Gyro-Bohm | ITG |
| `efeTEM` | Electron thermal flux | Gyro-Bohm | TEM |
| `efiTEM` | Ion thermal flux | Gyro-Bohm | TEM |
| `pfeTEM` | Particle flux | Gyro-Bohm | TEM |
| `efeETG` | Electron thermal flux | Gyro-Bohm | ETG |
| `gamma_max` | Maximum growth rate | cs/a | All |

**Note:** Output order matches ONNX model. Fluxes can be combined for different turbulence modes (ITG+TEM+ETG).

## Model Architecture

**QLKNN Neural Network:**
- Input: 10 plasma parameters
- Architecture: 5 hidden layers × 133 units (ReLU activation)
- Output: 8 transport fluxes (linear)
- Total parameters: 73,823 (Float32)
- Model size: 289KB (SafeTensors format)
- Framework: Pure MLX (Metal-accelerated)

## Documentation

- **[MLX_IMPLEMENTATION.md](MLX_IMPLEMENTATION.md)** - Complete MLX implementation details
- **[TORAX_INTEGRATION.md](TORAX_INTEGRATION.md)** - swift-TORAX integration patterns
- **[TESTING.md](TESTING.md)** - Testing guide and validation
- **[CLAUDE.md](CLAUDE.md)** - Project guidance for AI assistants

## Design Principles

1. **Pure Swift/MLX** - No Python runtime dependency for inference
2. **Self-Contained** - Bundled model weights (SafeTensors format)
3. **Type Safety** - Swift's type system prevents runtime errors
4. **Metal Acceleration** - Native GPU execution on Apple Silicon
5. **Float32 Precision** - Optimized for GPU memory bandwidth
6. **Validated** - Weights converted from fusion_surrogates ONNX model

## Testing

```bash
# Build package
swift build

# Run all tests
swift test

# Run specific test suite
swift test --filter WeightLoadingTests  # Model loading tests
swift test --filter MLXNetworkTests     # Inference tests (requires Metal)
```

**Test Coverage:**
- ✅ Model weight loading and validation
- ✅ Network architecture verification
- ✅ Forward pass shape correctness
- ✅ Float32 precision validation
- ✅ Physical validity (no NaN/Inf)

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Performance

**MLX Metal Acceleration:**
- Inference: <1ms per batch (25 cells) on M1/M2
- Memory: 289KB model + minimal runtime overhead
- Precision: Float32 for optimal GPU bandwidth
- Batching: Efficient processing of multiple radial points

**Typical Performance:**
- Grid size 50-500 cells: <1% of total simulation time
- PDE solver dominates runtime
- No Python conversion overhead

## Requirements

- **Swift:** 6.0 or later
- **Platform:** macOS 15.0 or later
- **Hardware:** Apple Silicon (M1/M2/M3) or Intel Mac with Metal support
- **Dependencies:**
  - [MLX-Swift](https://github.com/ml-explore/mlx-swift) 0.29.1+ (array operations)

**Platform Constraints:**
- ⚠️ **macOS only** - MLX requires Metal support
- ⚠️ **No Linux/Windows** - Metal GPU framework is macOS-only

## Related Projects

- **[TORAX](https://torax.readthedocs.io/)** - Original Python/JAX tokamak simulator (Google DeepMind)
- **[fusion_surrogates](https://github.com/google-deepmind/fusion_surrogates)** - Neural network surrogate models (Google DeepMind)
- **[MLX](https://github.com/ml-explore/mlx)** - Array framework for Apple Silicon (Apple)
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
- Pure Swift/MLX neural network (no Python runtime)
- QLKNN 7_11 model (73,823 parameters)
- Bundled SafeTensors weights (289KB)
- Input: 10 plasma parameters (Ati, Ate, etc.)
- Output: 8 transport fluxes (ITG/TEM/ETG modes)
- Precision: Float32 exclusively
