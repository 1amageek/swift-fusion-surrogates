# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**swift-fusion-surrogates** is a Swift library providing QLKNN neural network inference for fusion plasma transport modeling, designed for integration with swift-TORAX (a Swift reimplementation of the TORAX tokamak plasma transport simulator for Apple Silicon).

**Key Architectural Principle:** This is a generic library that uses standard MLX types (`MLXArray`), NOT swift-TORAX-specific types like `EvaluatedArray`. Type conversion to `EvaluatedArray` is the responsibility of swift-TORAX, not this library.

## Build & Test Commands

```bash
# Build the package
swift build

# Run all tests
swift test

# Run specific test suite
swift test --filter MLXNetworkStructureTests

# Run MLX inference tests (requires Metal runtime)
swift test --filter MLXNetworkTests

# List all tests
swift test --list-tests

# Clean build artifacts
swift package clean
```

## Architecture Overview

### Layer Structure

```
┌─────────────────────────────────────────────────────────┐
│  swift-TORAX (consumer)                                 │
│  - Uses EvaluatedArray (Sendable, for actors)           │
│  - Responsible for EvaluatedArray conversion            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  FusionSurrogates (this library)                        │
│  - MLX-native neural network (QLKNNNetwork)             │
│  - Uses standard MLXArray (Float32)                     │
│  - Generic library, not TORAX-specific                  │
│  - Returns [String: MLXArray]                           │
│  - ⚡ Metal-accelerated inference                        │
└─────────────────────────────────────────────────────────┘
```

### Numeric Precision Policy

**All numeric types use Float32 (32-bit floating point) exclusively throughout the codebase:**

- **Public APIs**: Accept `Float` (not `Double`)
- **Internal calculations**: Use `Float` for MLX operations
- **Strict Policy**: `Double` and `Float64` are **completely absent** from the codebase
  - ✅ Verified: No `Double` or `Float64` in Sources/, Tests/, or scripts
  - ✅ All MLXArray operations use `Float.self`
- **Rationale**: GPU efficiency, memory bandwidth optimization, and consistency with MLX best practices

### Core Components (Sources/FusionSurrogates/)

#### Neural Network Implementation

1. **QLKNNNetwork.swift**
   - Pure MLX neural network implementation
   - 5 hidden layers (133 units) + output layer (8 outputs)
   - 73,823 parameters, Float32 precision
   - `loadDefault()`: Loads from bundled SafeTensors weights
   - `callAsFunction()`: Forward pass with ReLU activations
   - `predict()`: High-level API with dictionary inputs/outputs

2. **ModelLoader.swift**
   - Pure Swift SafeTensors loader
   - Uses MLX's native `loadArrays(url:)` function
   - Loads from `Bundle.module` automatically

3. **Resources/**
   - `qlknn_7_11_weights.safetensors` - Bundled model weights (289 KB)
   - `qlknn_7_11_metadata.json` - Model architecture metadata

#### High-Level API

4. **QLKNN+MLX.swift**
   - High-level prediction API
   - `predictMLX()`: Pure MLX prediction with QLKNNNetwork
   - Input/output parameter name definitions
   - Input validation: shape checking, NaN/Inf detection

#### Helper Components

5. **TORAXIntegration.swift**
   - Helper functions for swift-TORAX integration
   - `buildInputs()`: Constructs QLKNN inputs from physics quantities
   - `combineFluxes()`: Combines ITG/TEM/ETG mode outputs into total transport coefficients
   - **MLX-native gradient computation** (GPU-accelerated, 10-100× faster than CPU)

### Input Parameters

**10 input parameters:**
```swift
QLKNN.inputParameterNames = [
    "Ati",        // R/L_Ti (ion temperature gradient)
    "Ate",        // R/L_Te (electron temperature gradient)
    "Ane",        // R/L_ne (electron density gradient)
    "Ani",        // R/L_ni (ion density gradient)
    "q",          // Safety factor
    "smag",       // Magnetic shear
    "x",          // r/R (inverse aspect ratio)
    "Ti_Te",      // Ion-electron temperature ratio
    "LogNuStar",  // Logarithmic normalized collisionality
    "normni"      // Normalized ion density (ni/ne)
]
```

### Output Parameters (8)

**CRITICAL**: Order matches ONNX model output order exactly.

```swift
QLKNN.outputParameterNames = [
    "efeITG",     // Electron thermal flux (ITG mode) - Index 0
    "efiITG",     // Ion thermal flux (ITG mode) - Index 1
    "pfeITG",     // Particle flux (ITG mode) - Index 2
    "efeTEM",     // Electron thermal flux (TEM mode) - Index 3
    "efiTEM",     // Ion thermal flux (TEM mode) - Index 4
    "pfeTEM",     // Particle flux (TEM mode) - Index 5
    "efeETG",     // Electron thermal flux (ETG mode) - Index 6
    "gamma_max"   // Maximum growth rate - Index 7
]
```

**Note:** This order was verified against ONNX model and is critical for correct output assignment. See `Scripts/verify_output_order.py`.

## Usage Pattern

```swift
// 1. Load MLX network (Metal-accelerated)
let network = try QLKNNNetwork.loadDefault()

// 2. Prepare inputs
let inputs: [String: MLXArray] = [
    "Ati": MLXArray([...], [batch_size]),
    "Ate": MLXArray([...], [batch_size]),
    // ... all 10 parameters
]

// 3. Predict (Metal-accelerated, <1ms)
let mlxOutputs = try network.predict(inputs)
// Returns: ["efeITG": MLXArray, "efiITG": MLXArray, ...]

// 4. Combine mode-specific fluxes
let combined = TORAXIntegration.combineFluxes(mlxOutputs)
// Returns: ["chi_ion", "chi_electron", "particle_flux", "growth_rate"]

// 5. swift-TORAX converts to EvaluatedArray (NOT FusionSurrogates' job)
let evaluated = EvaluatedArray.evaluatingBatch([
    combined["chi_ion"]!,
    combined["chi_electron"]!,
    combined["particle_flux"]!
])
```

**Critical Design Decisions:**
- FusionSurrogates MUST NOT return `EvaluatedArray`. It uses standard MLX types for generic compatibility.
- All MLXArrays use Float32 (not Float64) for GPU efficiency and memory bandwidth optimization.

## Dependencies

- **MLX-Swift**: Array operations and neural network modules (≥0.29.1)
  - `MLX`: Core array operations
  - `MLXNN`: Neural network module system (Linear, Module, etc.)

**Platform:** macOS 13.3+ (for MLX Metal support)

## Model Information

- **Current Model:** qlknn_7_11
- **Architecture:** 5 hidden layers × 133 units + output layer (8 outputs)
- **Parameters:** 73,823 total
- **Input Ranges:** Significantly expanded (e.g., Ati: 0-150 vs old 0-16)
- **Model Format:** SafeTensors (bundled in package resources)
- **Model Source:** Converted from ONNX model in fusion_surrogates Python package

### Model Conversion Pipeline

The ONNX → SafeTensors conversion was performed once during development:

```bash
python3 Scripts/convert_onnx_to_safetensors.py
```

**Source**: `/Library/Frameworks/Python.framework/.../fusion_surrogates/qlknn/models/qlknn_7_11.onnx`
**Output**: `Sources/FusionSurrogates/Resources/qlknn_7_11_weights.safetensors` (289 KB)

Users of this library do **not** need to perform this conversion - weights are bundled in the package.

## Key Design Patterns

### 1. Batch Evaluation (Performance Critical)

```swift
// ✅ RECOMMENDED: Single eval() call
let evaluated = EvaluatedArray.evaluatingBatch([array1, array2, array3])

// ❌ AVOID: Multiple eval() calls
let eval1 = EvaluatedArray(array1)  // GPU kernel launch
let eval2 = EvaluatedArray(array2)  // GPU kernel launch
let eval3 = EvaluatedArray(array3)  // GPU kernel launch
```

**Why:** MLX uses lazy evaluation. Batching allows GPU kernel fusion and better memory transfer.

### 2. MLX-Native Gradient Computation

Location: `TORAXIntegration.swift:144-182`

```swift
// GPU-accelerated using MLX slicing
let gradInterior = (fNext - fPrev) / (xNext - xPrev)
// Uses: f[2..<n], f[0..<(n-2)] for vectorized computation
```

**Do NOT** convert to Swift arrays for gradient calculation - that defeats GPU acceleration.

### 3. Input Validation Pattern

```swift
// Always validate before prediction
try QLKNN.validateInputs(inputs)   // Check all parameters present
try QLKNN.validateShapes(inputs)   // Check shapes, NaN/Inf, grid size

// Predict
let network = try QLKNNNetwork.loadDefault()
let outputs = try network.predict(inputs)
```

Validation includes:
- All 10 parameters present
- Consistent 1D shapes
- No NaN or Inf values
- Grid size: 2 ≤ n ≤ 10000

## Common Pitfalls

### 1. Wrong Output Parameter Order

**CRITICAL**: The output order must match ONNX model exactly:
```swift
// ✅ CORRECT (verified against ONNX)
["efeITG", "efiITG", "pfeITG", "efeTEM", "efiTEM", "pfeTEM", "efeETG", "gamma_max"]

// ❌ WRONG (old incorrect order)
["efiITG", "efeITG", "efeTEM", "efeETG", "efiTEM", "pfeITG", "pfeTEM", "gamma_max"]
```

Use `QLKNN.outputParameterNames` constant - do not manually construct this list.

### 2. Incorrect Parameter Names

Use the exact names defined in `QLKNN.inputParameterNames`:
- `"Ate"` (not "R_L_Te")
- `"smag"` (not "s_hat")
- `"x"` (not "r_R")
- `"LogNuStar"` (not "log_nu_star")

### 3. Returning EvaluatedArray from FusionSurrogates

FusionSurrogates returns `MLXArray` (Float32), NOT `EvaluatedArray`. Conversion is swift-TORAX's responsibility.

### 4. Using Double Instead of Float

❌ Wrong: `let inputs: [String: Double]`
✅ Correct: `let inputs: [String: Float]`

All public APIs and internal calculations use Float32 for GPU efficiency.

## Testing Strategy

### Automated Tests (swift test)

**Structure Tests:**
- `MLXNetworkStructureTests`: Network architecture, parameter counts, bundled resources (no Metal needed)

**Inference Tests:**
- `MLXNetworkTests`: Inference validation, shape tests, physical validity (requires Metal runtime)

**API Tests:**
- `BasicAPITests`: Parameter names, error descriptions (no dependencies)
- `FusionSurrogatesTests`: Package import verification
- `Float32PrecisionTests`: Numeric precision validation
- `InputValidationTests`: Input validation logic

**Integration Tests:**
- `TORAXIntegrationTests`: Helper functions for swift-TORAX

## Critical Files for Understanding

### Implementation
1. **MLX_IMPLEMENTATION.md** - Complete MLX implementation guide (⭐ ESSENTIAL)
2. **MLX_COMPLETION_SUMMARY.md** - Implementation summary and verification steps
3. **QLKNNNetwork.swift** - Core MLX network implementation

### Integration
4. **TORAX_INTEGRATION.md** - swift-TORAX integration patterns
5. **TORAXIntegration.swift** - Helper functions

### Scripts
6. **Scripts/convert_onnx_to_safetensors.py** - Model conversion tool (one-time use)
7. **Scripts/verify_output_order.py** - Output ordering verification

## When Modifying Code

### Adding New Transport Models

1. Create new MLX network class similar to `QLKNNNetwork`
2. Convert model weights to SafeTensors format
3. Add bundled resources to Package.swift
4. Create corresponding test suite
5. Update documentation

### Updating Model Weights

1. Obtain ONNX model from fusion_surrogates
2. Run `Scripts/convert_onnx_to_safetensors.py`
3. Replace `qlknn_7_11_weights.safetensors` in Resources/
4. Verify output order with `Scripts/verify_output_order.py`
5. Update tests if architecture changed

### Performance Optimization

- **Do NOT** optimize array conversions unless profiling shows >1% impact
- **Do** optimize gradient calculations (currently MLX-native)
- **Do** use batch evaluation patterns
- Total overhead target: <1% of simulation time

## Current Status

- Pure MLX neural network inference (QLKNNNetwork)
- Bundled model weights in SafeTensors format (289 KB)
- Float32 precision throughout (no Double/Float64)
- Metal GPU acceleration
- Input/output parameter validation
- MLX-native gradient computation
- swift-TORAX integration helpers

## Quick Start

```swift
import FusionSurrogates
import MLX

// 1. Load network (Metal-accelerated)
let network = try QLKNNNetwork.loadDefault()

// 2. Prepare inputs (example: single cell)
let inputs: [String: MLXArray] = [
    "Ati": MLXArray([5.0], [1]),
    "Ate": MLXArray([5.0], [1]),
    "Ane": MLXArray([1.0], [1]),
    "Ani": MLXArray([1.0], [1]),
    "q": MLXArray([2.0], [1]),
    "smag": MLXArray([1.0], [1]),
    "x": MLXArray([0.3], [1]),
    "Ti_Te": MLXArray([1.0], [1]),
    "LogNuStar": MLXArray([-10.0], [1]),
    "normni": MLXArray([1.0], [1])
]

// 3. Predict transport fluxes
let outputs = try network.predict(inputs)

// 4. Access results
let ionFlux = outputs["efiITG"]!      // Ion thermal flux (ITG)
let electronFlux = outputs["efeITG"]! // Electron thermal flux (ITG)
let growthRate = outputs["gamma_max"]! // Growth rate
```

For batch predictions, simply provide arrays with `[batch_size]` shape instead of `[1]`.
