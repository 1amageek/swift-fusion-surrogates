# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**swift-fusion-surrogates** is a Swift wrapper for Google DeepMind's fusion_surrogates Python library, designed for integration with swift-TORAX (a Swift reimplementation of the TORAX tokamak plasma transport simulator for Apple Silicon).

**Key Architectural Principle:** This is a generic wrapper that uses standard MLX types (`MLXArray`), NOT swift-TORAX-specific types like `EvaluatedArray`. Type conversion to `EvaluatedArray` is the responsibility of swift-TORAX, not this library.

## Build & Test Commands

```bash
# Build the package
swift build

# Run all tests (environment-independent tests only)
swift test

# Run specific test suite
swift test --filter BasicAPITests

# List all tests
swift test --list-tests

# Clean build artifacts
swift package clean
```

**Note:** Python integration tests are disabled (`.disabled` extension) due to environment dependencies. Use the verification scripts instead:

```bash
# Verify Python API
python3 verify_python_api.py

# Run full integration test
python3 test_new_api_final.py
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
│  - Uses standard MLXArray (Float32)                     │
│  - Generic wrapper, not TORAX-specific                  │
│  - Returns [String: MLXArray]                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  fusion_surrogates (Python)                             │
│  - QLKNNModel API (v0.4.2+)                            │
│  - Submodule: fusion_surrogates/                        │
└─────────────────────────────────────────────────────────┘
```

### Numeric Precision Policy

**All numeric types use Float32 (32-bit floating point) exclusively throughout the codebase:**

- **Public APIs**: Accept `Float` (not `Double`)
- **Internal calculations**: Use `Float` for MLX operations
- **Python conversion**: Convert between Float32 and Python numpy float32
- **Strict Policy**: `Double` and `Float64` are **completely absent** from the codebase
  - ✅ Verified: No `Double` or `Float64` in Sources/, Tests/, or scripts
  - ✅ All MLXArray operations use `Float.self`
- **Rationale**: GPU efficiency, memory bandwidth optimization, and consistency with MLX best practices

### Core Components (Sources/FusionSurrogates/)

1. **FusionSurrogates.swift**
   - Low-level PythonKit wrapper
   - `QLKNN` class: Loads model via `QLKNNModel.load_default_model()`
   - `predictPython()`: Returns `PythonObject` (raw Python interface)

2. **QLKNN+MLX.swift**
   - High-level MLX API (recommended)
   - `predict()`: Converts MLXArray → Python → MLXArray
   - Input/output parameter name definitions
   - Input validation: shape checking, NaN/Inf detection

3. **MLXConversion.swift**
   - Bidirectional MLXArray ↔ Python numpy conversion
   - **Critical:** `batchToPythonArray()` converts Dict[String: MLXArray] → 2D numpy array
   - Order matters: [Ati, Ate, Ane, Ani, q, smag, x, Ti_Te, LogNuStar, normni]

4. **TORAXIntegration.swift**
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
    "smag",       // Magnetic shear (was s_hat in legacy)
    "x",          // r/R (was r_R in legacy)
    "Ti_Te",      // Temperature ratio
    "LogNuStar",  // Collisionality (was log_nu_star)
    "normni"      // ni/ne (was ni_ne)
]
```

**Output Parameters (8):**
```swift
QLKNN.outputParameterNames = [
    "efiITG",     // Ion thermal flux (ITG mode)
    "efeITG",     // Electron thermal flux (ITG mode)
    "efeTEM",     // Electron thermal flux (TEM mode)
    "efeETG",     // Electron thermal flux (ETG mode)
    "efiTEM",     // Ion thermal flux (TEM mode)
    "pfeITG",     // Particle flux (ITG mode)
    "pfeTEM",     // Particle flux (TEM mode)
    "gamma_max"   // Growth rate
]
```

**Note:** See `API_MIGRATION.md` for technical details on parameter definitions.

## Type Flow Pattern

```swift
// 1. FusionSurrogates returns MLXArray (Float32)
let mlxOutputs: [String: MLXArray] = try qlknn.predict(inputs)

// 2. Combine mode-specific fluxes
let combined = TORAXIntegration.combineFluxes(mlxOutputs)
// Returns: ["chi_ion", "chi_electron", "particle_flux", "growth_rate"]

// 3. swift-TORAX converts to EvaluatedArray (NOT FusionSurrogates' job)
let evaluated = EvaluatedArray.evaluatingBatch([
    combined["chi_ion"]!,
    combined["chi_electron"]!,
    combined["particle_flux"]!
])
```

**Critical Design Decisions:**
- FusionSurrogates MUST NOT return `EvaluatedArray`. It uses standard MLX types for generic compatibility.
- All MLXArrays use Float32 (not Float64) for GPU efficiency and memory bandwidth optimization.

## Python Dependency

This library wraps Python's fusion_surrogates instead of reimplementing models:

**Pros:**
- Automatic upstream tracking (fusion_surrogates is a git submodule)
- No manual weight conversion
- Validated against reference implementation

**Cons:**
- Requires Python 3.12+ with fusion_surrogates installed
- Small performance overhead (~1-10ms per prediction, <1% of total simulation time)

**Setup:**
```bash
pip install fusion-surrogates
```

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
let outputs = try qlknn.predict(inputs)
```

Validation includes:
- All 10 parameters present
- Consistent 1D shapes
- No NaN or Inf values
- Grid size: 2 ≤ n ≤ 10000

## Common Pitfalls

### 1. Wrong Input Order in batchToPythonArray

The API requires a 2D numpy array with features in EXACT order:
```swift
[Ati, Ate, Ane, Ani, q, smag, x, Ti_Te, LogNuStar, normni]
```

`MLXConversion.batchToPythonArray()` handles this automatically. Do NOT manually construct 2D arrays.

### 2. Incorrect Parameter Names

Use the exact names defined in `QLKNN.inputParameterNames`:
- `"Ate"` (not "R_L_Te")
- `"smag"` (not "s_hat")
- `"x"` (not "r_R")
- `"LogNuStar"` (not "log_nu_star")

### 3. Assuming Dict Input to Python

The `QLKNNModel` API requires:
```python
inputs = np.array([[...]])  # 2D array, NOT dict
outputs = model.predict(inputs)
```

`batchToPythonArray()` handles this conversion.

### 4. Returning EvaluatedArray from FusionSurrogates

FusionSurrogates returns `MLXArray` (Float32), NOT `EvaluatedArray`. Conversion is swift-TORAX's responsibility.

### 5. Using Double Instead of Float

❌ Wrong: `let inputs: [String: Double]`
✅ Correct: `let inputs: [String: Float]`

All public APIs and internal calculations use Float32 for GPU efficiency.

## Testing Strategy

### Automated Tests (swift test)
- `BasicAPITests`: Parameter names, error descriptions (no Python/MLX dependencies)
- `FusionSurrogatesTests`: Package import verification

### Manual Verification
- `verify_python_api.py`: Validates Python API directly
- `test_new_api_final.py`: Full integration test with sample data

### Integration Tests (Disabled)
- `PythonIntegrationTests.swift.disabled`: Requires Python environment
- `MLXIntegrationTests.swift.disabled`: Requires MLX Metal library

**Reason for disabling:** Environment-dependent tests fail in CI. Use verification scripts instead.

## Critical Files for Understanding

1. **TORAX_INTEGRATION.md** - swift-TORAX integration patterns (essential)
2. **API_MIGRATION.md** - Parameter definitions and Python API details
3. **IMPLEMENTATION_NOTES.md** - Technical details and design decisions
4. **TESTING.md** - Testing guide and manual verification

## Model Information

- **Current Model:** qlknn_7_11_v1
- **Input Ranges:** Significantly expanded (e.g., Ati: 0-150 vs old 0-16)
- **Model Format:** .qlknn (custom format, NOT qlknn-hyper JSON)
- **Model Location:** Loaded from fusion_surrogates Python package, NOT from qlknn-hyper submodule

**Note:** qlknn-hyper is referenced in docs but NOT used directly. fusion_surrogates manages model loading.

## When Modifying Code

### Adding New Transport Models
1. Create new class similar to `QLKNN` in `FusionSurrogates.swift`
2. Add MLX extensions in new file (e.g., `NewModel+MLX.swift`)
3. Update `TORAXIntegration.swift` if needed for helper functions
4. Add parameter name constants to `QLKNN+MLX.swift` pattern

### Updating for fusion_surrogates API Changes
1. Check `verify_python_api.py` for API changes
2. Update `FusionSurrogates.swift` (model loading)
3. Update `MLXConversion.swift` (input/output format)
4. Update `QLKNN+MLX.swift` (parameter names)
5. Update `API_MIGRATION.md` with changes
6. Run `test_new_api_final.py` to verify

### Performance Optimization
- **Do NOT** optimize array conversions unless profiling shows >1% impact
- **Do** optimize gradient calculations (currently MLX-native)
- **Do** use batch evaluation patterns
- Total overhead target: <1% of simulation time

## Dependencies

- **PythonKit**: Python interoperability (master branch)
- **MLX-Swift**: Array operations (≥0.29.1)
- **fusion_surrogates**: Python library (≥0.4.2, via pip)

**Platform:** macOS 13.3+ (for MLX Metal support)

## Current Status

- Uses fusion_surrogates `QLKNNModel` API
- Parameter names: Ati, Ate, etc. (QuaLiKiz standard)
- Input ranges: Expanded for QLKNN 7_11_v1 model
- Float32 precision throughout (no Double/Float64)
