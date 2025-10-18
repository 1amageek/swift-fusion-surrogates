# swift-TORAX Integration Guide

This document describes how to integrate FusionSurrogates with swift-TORAX for turbulent transport modeling.

## Overview

FusionSurrogates provides a **PythonKit-based wrapper** for Google DeepMind's `fusion_surrogates` library, with seamless **MLX-Swift integration** for use in swift-TORAX.

### Key Features

- ✅ **PythonKit wrapper**: Uses the official Python fusion_surrogates library
- ✅ **Automatic upstream tracking**: New models and updates are immediately available
- ✅ **MLXArray conversion**: Seamless bidirectional conversion between MLX and numpy
- ✅ **TORAX-ready helpers**: Input preprocessing from CoreProfiles and Geometry
- ✅ **Zero model conversion**: No need to convert Python model weights to Swift

### Design Philosophy

This package intentionally uses **PythonKit as a wrapper** rather than implementing native Swift models. This design choice provides:

1. **Automatic updates**: When fusion_surrogates is updated, you get new features immediately
2. **Maintenance-free**: No need to manually port Python code or convert model weights
3. **Future-proof**: New models (TGLFNN, etc.) work automatically
4. **Validation**: Results match the reference Python implementation exactly

The trade-off is a small performance overhead for Python ⇔ Swift conversion, which is acceptable for transport modeling where the bottleneck is typically the PDE solver, not the surrogate evaluation.

## Architecture

```
swift-TORAX (EvaluatedArray based)
    ↓ Extract MLXArray
CoreProfiles.ionTemperature.value → MLXArray
    ↓
FusionSurrogates (this package)
    ├── MLXArray ⇔ PythonObject conversion
    ├── QLKNN wrapper with MLX API
    └── TORAX integration helpers
    ↓
fusion_surrogates (Python library)
    └── Pre-trained neural network models
    ↓
QLKNN prediction (PythonObject)
    ↓
Convert to MLXArray
    ↓
Wrap in EvaluatedArray → TransportCoefficients
```

## Type System Integration

### FusionSurrogates Return Types

**Important**: FusionSurrogates returns `MLXArray`, not `EvaluatedArray`.

```swift
// FusionSurrogates API
public func predict(_ inputs: [String: MLXArray]) -> [String: MLXArray]
//                                   ^^^^^^^^              ^^^^^^^^
//                                   MLXArray              MLXArray
```

This is intentional:
- **FusionSurrogates**: Generic wrapper, returns standard MLX types
- **swift-TORAX**: Owns the `EvaluatedArray` type system
- **Conversion**: swift-TORAX's responsibility

### EvaluatedArray Conversion Pattern

swift-TORAX uses `EvaluatedArray` to guarantee MLXArray evaluation before crossing actor boundaries or storing in `Sendable` structures.

**Data Flow:**
```swift
CoreProfiles (contains EvaluatedArray)
    → extract .value → MLXArray
    → FusionSurrogates.predict() → MLXArray
    → wrap in EvaluatedArray → TransportCoefficients
```

## Installation

### 1. Install Python fusion_surrogates

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install fusion_surrogates
pip install fusion_surrogates
```

### 2. Add FusionSurrogates to swift-TORAX

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/YOUR_USERNAME/FusionSurrogates.git", branch: "main"),
    // ... other dependencies
]

targets: [
    .target(
        name: "TORAX",
        dependencies: [
            "FusionSurrogates",
            // ... other dependencies
        ]
    ),
]
```

### 3. Set PYTHONPATH (if using submodule)

If you're using the fusion_surrogates submodule directly:

```bash
export PYTHONPATH=/path/to/FusionSurrogates/fusion_surrogates:$PYTHONPATH
```

## Usage

### Basic QLKNN Usage with MLXArrays

```swift
import MLX
import FusionSurrogates

// Initialize QLKNN model
let qlknn = try QLKNN(modelVersion: "7_11")

// Prepare inputs as MLXArrays
let inputs: [String: MLXArray] = [
    "R_L_Te": MLXArray([5.0, 5.1, 5.2]),
    "R_L_Ti": MLXArray([5.0, 5.0, 5.0]),
    "R_L_ne": MLXArray([2.0, 2.0, 2.0]),
    "R_L_ni": MLXArray([2.0, 2.0, 2.0]),
    "q": MLXArray([2.0, 2.1, 2.2]),
    "s_hat": MLXArray([1.0, 1.0, 1.0]),
    "r_R": MLXArray([0.3, 0.4, 0.5]),
    "Ti_Te": MLXArray([1.0, 1.0, 1.0]),
    "log_nu_star": MLXArray([-2.0, -2.0, -2.0]),
    "ni_ne": MLXArray([1.0, 1.0, 1.0])
]

// Predict transport coefficients
let outputs = try qlknn.predict(inputs)

// Extract results
let chiIon = outputs["chi_ion_itg"]
let chiElectron = outputs["chi_electron_tem"]
```

## EvaluatedArray Conversion Patterns

### Pattern 1: Individual Conversion

Convert each MLXArray output to EvaluatedArray separately:

```swift
import MLX
import FusionSurrogates

let qlknn = try QLKNN(modelVersion: "7_11")
let inputs: [String: MLXArray] = [...] // QLKNN inputs

// Get QLKNN predictions (returns MLXArray)
let qlknnOutputs = try qlknn.predict(inputs)  // [String: MLXArray]

// Combine fluxes
let combined = TORAXIntegration.combineFluxes(qlknnOutputs)

// Convert to EvaluatedArray individually
return TransportCoefficients(
    chiIon: EvaluatedArray(evaluating: combined["chi_ion"]!),
    chiElectron: EvaluatedArray(evaluating: combined["chi_electron"]!),
    particleDiffusivity: EvaluatedArray(evaluating: combined["particle_diffusivity"]!),
    convectionVelocity: EvaluatedArray(evaluating: combined["convection_velocity"]!)
)
```

### Pattern 2: Batch Conversion (Recommended for Performance)

Use `EvaluatedArray.evaluatingBatch()` for efficient multi-array evaluation:

```swift
let qlknnOutputs = try qlknn.predict(inputs)
let combined = TORAXIntegration.combineFluxes(qlknnOutputs)

// Batch evaluation (more efficient - single eval() call)
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
```

**Why batch conversion is better:**
- Single `eval()` call instead of 4 separate calls
- More efficient memory usage
- Better for GPU synchronization

### Integration with swift-TORAX TransportModel Protocol

Create a `QLKNNTransportModel.swift` in your TORAX project:

```swift
import Foundation
import MLX
import FusionSurrogates

public struct QLKNNTransportModel: TransportModel {
    public let name = "qlknn"

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

        // 1. Extract MLXArrays from EvaluatedArray (no eval needed - already evaluated)
        let Te = profiles.electronTemperature.value  // MLXArray
        let Ti = profiles.ionTemperature.value       // MLXArray
        let ne = profiles.electronDensity.value      // MLXArray
        let psi = profiles.poloidalFlux.value        // MLXArray

        // 2. Build QLKNN inputs (returns [String: MLXArray])
        let inputs = QLKNN.buildInputs(
            electronTemperature: Te,
            ionTemperature: Ti,
            electronDensity: ne,
            ionDensity: ne,  // Assume ni = ne for single-ion plasma
            poloidalFlux: psi,
            radius: geometry.rho.value,
            majorRadius: majorRadius,
            minorRadius: minorRadius,
            toroidalField: toroidalField
        )

        // 3. Run QLKNN prediction (returns [String: MLXArray])
        let qlknnOutputs = try! qlknn.predict(inputs)

        // 4. Combine fluxes into transport coefficients (returns [String: MLXArray])
        let combined = TORAXIntegration.combineFluxes(qlknnOutputs)

        // 5. Convert to EvaluatedArray using batch evaluation (efficient!)
        let evaluated = EvaluatedArray.evaluatingBatch([
            combined["chi_ion"]!,
            combined["chi_electron"]!,
            combined["particle_diffusivity"]!,
            combined["convection_velocity"]!
        ])

        // 6. Return TransportCoefficients (Sendable, contains EvaluatedArray)
        return TransportCoefficients(
            chiIon: evaluated[0],
            chiElectron: evaluated[1],
            particleDiffusivity: evaluated[2],
            convectionVelocity: evaluated[3]
        )
    }
}
```

**Key Points:**

1. **Extract MLXArray** from `EvaluatedArray.value` - no `eval()` needed (already evaluated)
2. **FusionSurrogates operations** work with `MLXArray` (lazy evaluation OK here)
3. **Batch conversion** to `EvaluatedArray` at the end (efficient single `eval()`)
4. **Return Sendable type** (`TransportCoefficients`) safe for actor boundaries

### Important Notes

**Why doesn't FusionSurrogates use EvaluatedArray?**

FusionSurrogates is designed as a **generic wrapper library** that:
- Wraps the Python `fusion_surrogates` library for upstream tracking
- Uses standard MLX types (`MLXArray`) for maximum compatibility
- Can be used in contexts beyond swift-TORAX
- Keeps the wrapper layer simple and maintainable

`EvaluatedArray` is a **swift-TORAX-specific type** that enforces `Sendable` conformance for actor-isolated code. The conversion responsibility naturally belongs in swift-TORAX, not in the generic wrapper.

**Performance Implications:**

```swift
// ❌ Inefficient: Multiple eval() calls
let chiIon = EvaluatedArray(combined["chi_ion"]!)      // eval() called
let chiElectron = EvaluatedArray(combined["chi_electron"]!)  // eval() called
let particleDiff = EvaluatedArray(combined["particle_diffusivity"]!)  // eval() called

// ✅ Efficient: Single batch eval()
let evaluated = EvaluatedArray.evaluatingBatch([
    combined["chi_ion"]!,
    combined["chi_electron"]!,
    combined["particle_diffusivity"]!
])  // Single eval() for all arrays
```

Batch evaluation can be **significantly faster** because:
1. MLX can optimize the computation graph across all arrays
2. GPU kernels are launched once for the entire batch
3. Memory transfers are batched

**When to Use Each Pattern:**

- **Batch conversion** (recommended): When you have multiple related outputs
- **Individual conversion**: Only when you need one or two arrays, or when arrays are computed at different times

**Debugging Evaluation Issues:**

If you encounter actor isolation errors:
```swift
// ❌ Error: MLXArray is not Sendable
// actor MyActor {
//     var data: MLXArray  // Won't compile
// }

// ✅ Solution: Use EvaluatedArray
actor MyActor {
    var data: EvaluatedArray  // Sendable
}
```

If you see unexpected lazy evaluation behavior:
```swift
// Check if array is evaluated
print(myArray.isEvaled)  // false = lazy, true = evaluated

// Force evaluation if needed
eval(myArray)
```

### Update TransportModelFactory

In `TransportModelFactory.swift`, replace the `notImplemented()` error:

```swift
case .qlknn:
    let majorRadius = config.parameters["major_radius"] as? Float ?? 6.2
    let minorRadius = config.parameters["minor_radius"] as? Float ?? 2.0
    let toroidalField = config.parameters["toroidal_field"] as? Float ?? 5.3

    return try QLKNNTransportModel(
        majorRadius: majorRadius,
        minorRadius: minorRadius,
        toroidalField: toroidalField
    )
```

### Configuration

Add to your TORAX configuration JSON:

```json
{
  "runtime": {
    "dynamic": {
      "transport": {
        "modelType": "qlknn",
        "parameters": {
          "major_radius": 6.2,
          "minor_radius": 2.0,
          "toroidal_field": 5.3
        }
      }
    }
  }
}
```

## API Reference

### MLXConversion

Utilities for converting between MLXArray and Python numpy arrays:

```swift
// MLXArray → numpy
let numpyArray = MLXConversion.toPython(mlxArray)
let numpyArray = mlxArray.pythonArray  // Extension method

// numpy → MLXArray
let mlxArray = MLXConversion.fromPython(numpyArray)
let mlxArray = MLXArray.from(pythonArray: numpyArray)  // Static method

// Batch conversion
let numpyDict = MLXConversion.batchToPython(mlxDict)
let mlxDict = MLXConversion.batchFromPython(pythonDict)
```

### QLKNN Extensions

```swift
// Predict with MLXArray inputs
let outputs = try qlknn.predict(inputs: [String: MLXArray])

// Predict with scalar inputs (broadcast to grid)
let outputs = try qlknn.predictScalar(inputs: [String: Double], nCells: 100)

// Predict with profile inputs
let outputs = try qlknn.predictProfiles(profiles: [String: MLXArray])

// Validate inputs
try QLKNN.validateInputs(inputs)
```

### TORAXIntegration Helpers

```swift
// Compute normalized gradients
let rLnTe = TORAXIntegration.computeNormalizedGradient(
    profile: temperature,
    radius: radius,
    majorRadius: 6.2
)

// Compute safety factor
let q = TORAXIntegration.computeSafetyFactor(
    poloidalFlux: psi,
    radius: radius,
    minorRadius: 2.0,
    majorRadius: 6.2,
    toroidalField: 5.3
)

// Compute magnetic shear
let sHat = TORAXIntegration.computeMagneticShear(
    safetyFactor: q,
    radius: radius
)

// Compute collisionality
let logNuStar = TORAXIntegration.computeCollisionality(
    density: ne,
    temperature: Te,
    majorRadius: 6.2,
    safetyFactor: q
)

// Combine QLKNN fluxes
let coeffs = TORAXIntegration.combineFluxes(qlknnOutputs)
```

### Complete Input Builder

```swift
// Build all QLKNN inputs from TORAX data
let inputs = QLKNN.buildInputs(
    electronTemperature: Te,
    ionTemperature: Ti,
    electronDensity: ne,
    ionDensity: ni,
    poloidalFlux: psi,
    radius: r,
    majorRadius: R,
    minorRadius: a,
    toroidalField: BT
)
```

## QLKNN Input Parameters

| Parameter | Symbol | Description | Unit |
|-----------|--------|-------------|------|
| `R_L_Te` | R/L_{Te} | Normalized electron temperature gradient | - |
| `R_L_Ti` | R/L_{Ti} | Normalized ion temperature gradient | - |
| `R_L_ne` | R/L_{ne} | Normalized electron density gradient | - |
| `R_L_ni` | R/L_{ni} | Normalized ion density gradient | - |
| `q` | q | Safety factor | - |
| `s_hat` | ŝ | Magnetic shear | - |
| `r_R` | r/R | Local inverse aspect ratio | - |
| `Ti_Te` | T_i/T_e | Ion-electron temperature ratio | - |
| `log_nu_star` | log(ν*) | Logarithmic normalized collisionality | - |
| `ni_ne` | n_i/n_e | Normalized density ratio | - |

## QLKNN Output Parameters

| Parameter | Description |
|-----------|-------------|
| `chi_ion_itg` | Ion heat diffusivity from ITG mode [m²/s] |
| `chi_electron_tem` | Electron heat diffusivity from TEM mode [m²/s] |
| `chi_electron_etg` | Electron heat diffusivity from ETG mode [m²/s] |
| `particle_flux` | Particle diffusivity [m²/s] |
| `growth_rate` | Maximum growth rate [1/s] |

## Performance Considerations

### Conversion Overhead

The PythonKit ⇔ MLX conversion has overhead (~10-100 μs per array). For typical TORAX simulations:

- **Grid size**: 100-500 cells
- **Transport evaluation**: ~1-10 times per timestep
- **Overhead**: <1% of total simulation time

The PDE solver (Newton-Raphson, Jacobian computation) dominates the runtime, so the conversion overhead is negligible.

### Optimization Tips

1. **Batch conversions**: Use `batchToPython()` and `batchFromPython()` for multiple arrays
2. **Minimize calls**: Cache QLKNN inputs if profiles haven't changed
3. **Profile-guided**: Profile your simulation to identify actual bottlenecks

## Troubleshooting

### Python Module Not Found

```bash
# Ensure fusion_surrogates is installed
pip install fusion_surrogates

# Or set PYTHONPATH to submodule
export PYTHONPATH=/path/to/FusionSurrogates/fusion_surrogates:$PYTHONPATH
```

### PythonKit Import Errors

```swift
// Specify Python library path explicitly
PythonLibrary.useLibrary(at: "/usr/local/bin/python3")

let qlknn = try QLKNN(modelVersion: "7_11")
```

### Shape Mismatches

```swift
// Ensure all inputs have the same shape
let nCells = profiles.electronTemperature.shape[0]
print("Grid size:", nCells)

// Validate input shapes
for (key, value) in inputs {
    assert(value.shape[0] == nCells, "Shape mismatch for \(key)")
}
```

## Future Enhancements

### Planned Features

- [ ] Gradient computation support (for sensitivity analysis)
- [ ] TGLFNN model integration
- [ ] Model caching and memoization
- [ ] Comprehensive finite difference schemes
- [ ] Benchmark suite comparing with Python implementation

### Migration Path to Native MLX (Optional)

If you later want to migrate to native MLX for maximum performance:

1. Export QLKNN model to ONNX format
2. Convert ONNX to MLX using `mlx.nn.Module`
3. Replace `QLKNNTransportModel` implementation
4. Keep the same API for backward compatibility

This package is designed to make that migration seamless when needed.

## References

- [fusion_surrogates GitHub](https://github.com/google-deepmind/fusion_surrogates)
- [swift-TORAX](https://github.com/YOUR_USERNAME/swift-TORAX)
- [PythonKit](https://github.com/pvieito/PythonKit)
- [MLX-Swift](https://github.com/ml-explore/mlx-swift)
- [QLKNN Paper (Van de Plassche et al. 2020)](https://doi.org/10.1063/1.5134126)

## License

This project follows the same licensing as fusion_surrogates:
- Code: Apache License 2.0
- Models: Creative Commons Attribution 4.0 International (CC-BY)
