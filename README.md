# FusionSurrogates for Swift

A Swift wrapper for Google DeepMind's [fusion_surrogates](https://github.com/google-deepmind/fusion_surrogates) Python library, designed for integration with [swift-TORAX](https://github.com/google-deepmind/torax) - a Swift implementation of the TORAX tokamak plasma transport simulator.

[![Swift 6.0+](https://img.shields.io/badge/Swift-6.0+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/platform-macOS%2013.3+-lightgrey.svg)](https://www.apple.com/macos/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Overview

**FusionSurrogates** provides a Swift interface to neural network surrogate models for turbulent transport in fusion plasmas. It enables fast, GPU-accelerated transport coefficient predictions for tokamak simulations on Apple Silicon.

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

- ✅ **Swift-native API** - Type-safe interface using `MLXArray`
- ✅ **GPU-accelerated** - MLX-native gradient computation (10-100× faster)
- ✅ **Automatic validation** - Shape checking, NaN/Inf detection, bounds verification
- ✅ **QLKNN support** - QuaLiKiz neural network for ITG/TEM/ETG turbulence
- ✅ **swift-TORAX integration** - Helpers for `EvaluatedArray` conversion
- ✅ **Upstream tracking** - Uses fusion_surrogates as git submodule for automatic updates

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
let qlknn = try QLKNN(modelVersion: "7_11")

// Prepare inputs (normalized gradients, safety factor, etc.)
let inputs: [String: MLXArray] = [
    "R_L_Te": MLXArray([5.0, 5.5, 6.0], [3]),    // Electron temp gradient
    "R_L_Ti": MLXArray([5.0, 5.5, 6.0], [3]),    // Ion temp gradient
    "R_L_ne": MLXArray([1.0, 1.2, 1.4], [3]),    // Density gradient
    "R_L_ni": MLXArray([1.0, 1.2, 1.4], [3]),    // Ion density gradient
    "q": MLXArray([2.0, 2.5, 3.0], [3]),         // Safety factor
    "s_hat": MLXArray([1.0, 1.2, 1.4], [3]),     // Magnetic shear
    "r_R": MLXArray([0.3, 0.35, 0.4], [3]),      // Inverse aspect ratio
    "Ti_Te": MLXArray([1.0, 1.0, 1.0], [3]),     // Temperature ratio
    "log_nu_star": MLXArray([-10.0, -9.5, -9.0], [3]), // Collisionality
    "ni_ne": MLXArray([1.0, 1.0, 1.0], [3])      // Density ratio
]

// Run prediction
let outputs = try qlknn.predict(inputs)

// Access transport coefficients
let chiIon = outputs["chi_ion_itg"]!           // Ion heat diffusivity
let chiElectron = outputs["chi_electron_tem"]! // Electron heat diffusivity
```

## Usage with swift-TORAX

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
        self.qlknn = try QLKNN(modelVersion: "7_11")
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
        let inputs = QLKNN.buildInputs(
            electronTemperature: Te,
            ionTemperature: Ti,
            electronDensity: ne,
            ionDensity: ne,
            poloidalFlux: profiles.poloidalFlux.value,
            radius: geometry.rho.value,
            majorRadius: majorRadius,
            minorRadius: minorRadius,
            toroidalField: toroidalField
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

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `R_L_Te` | Normalized electron temperature gradient | 0-16 |
| `R_L_Ti` | Normalized ion temperature gradient | 0-16 |
| `R_L_ne` | Normalized electron density gradient | -5 to 5 |
| `R_L_ni` | Normalized ion density gradient | -15 to 15 |
| `q` | Safety factor | 0.66-10 |
| `s_hat` | Magnetic shear | -1 to 4 |
| `r_R` | Inverse aspect ratio | 0.1-0.95 |
| `Ti_Te` | Ion-electron temperature ratio | 0.25-2.5 |
| `log_nu_star` | Logarithmic collisionality | -5 to 0 |
| `ni_ne` | Normalized density ratio | 0.7-1.0 |

### QLKNN Output Parameters

| Parameter | Description | Physics |
|-----------|-------------|---------|
| `chi_ion_itg` | Ion heat diffusivity (ITG) | m²/s |
| `chi_electron_tem` | Electron heat diffusivity (TEM) | m²/s |
| `chi_electron_etg` | Electron heat diffusivity (ETG) | m²/s |
| `particle_flux` | Particle flux | 1/s |
| `growth_rate` | Instability growth rate | 1/s |

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

- **[DESIGN_SUMMARY.md](DESIGN_SUMMARY.md)** - Architecture overview and design principles
- **[TORAX_INTEGRATION.md](TORAX_INTEGRATION.md)** - Complete swift-TORAX integration guide ⭐
- **[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)** - Technical details and known issues
- **[TESTING.md](TESTING.md)** - Testing guide and validation
- **[QLKNN_HYPER_INFO.md](QLKNN_HYPER_INFO.md)** - About the QLKNN model weights
- **[STATUS.md](STATUS.md)** - Current project status

## Architecture

### Type System

```swift
// FusionSurrogates uses standard MLX types
public func predict(_ inputs: [String: MLXArray]) -> [String: MLXArray]

// swift-TORAX uses EvaluatedArray (Sendable)
public struct EvaluatedArray: Sendable {
    let value: MLXArray  // Already evaluated
}

// Conversion happens at swift-TORAX layer
let mlxOutputs = qlknn.predict(inputs)
let evaluated = EvaluatedArray.evaluatingBatch(mlxOutputs.values)
```

### Design Principles

1. **Generic Wrapper** - Not swift-TORAX-specific, uses standard MLX types
2. **Upstream Tracking** - fusion_surrogates as submodule for automatic updates
3. **Type Safety** - Swift's type system prevents runtime errors
4. **GPU Acceleration** - MLX-native operations where possible
5. **Clear Separation** - FusionSurrogates → MLXArray, swift-TORAX → EvaluatedArray

## Testing

```bash
# Run all tests
swift test

# Run specific test suite
swift test --filter BasicAPITests

# Expected output:
# Test run with 4 tests in 2 suites passed after 0.001 seconds.
```

See [`TESTING.md`](TESTING.md) for details on manual integration testing.

## Performance

| Operation | Performance |
|-----------|-------------|
| Gradient computation | 10-100× faster (GPU vs CPU) |
| Array conversion | ~10-100 μs per array |
| Prediction overhead | ~1-10 ms (grid size 100-500) |
| Impact on simulation | <1% (PDE solver dominates) |

## Requirements

- **Swift:** 6.0 or later
- **Platform:** macOS 13.3 or later
- **Python:** 3.12 or later
- **Dependencies:**
  - [PythonKit](https://github.com/pvieito/PythonKit) - Python interop
  - [MLX-Swift](https://github.com/ml-explore/mlx-swift) - Array operations (0.29.1+)
  - [fusion_surrogates](https://github.com/google-deepmind/fusion_surrogates) - Python library (pip)

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

**Status:** ✅ Ready for swift-TORAX integration

**Version:** 1.0.0

**Tested with:**
- Swift 6.2
- MLX-Swift 0.29.1
- fusion_surrogates 0.4.2
- Python 3.12
