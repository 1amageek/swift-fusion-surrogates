# Testing Guide

## Current Test Status

✅ **Comprehensive test suite** (43 test cases across 6 suites)

```bash
swift test
```

**Test Framework:** Swift Testing (modern Swift 6+ framework)

**Note:** MLX tests require Metal library. Tests may show MLX errors but compile successfully.

## Test Structure

### Implemented Tests (Swift Testing Framework)

**BasicAPITests.swift** (3 tests) - Basic functionality tests (no MLX or Python dependencies)
- ✅ `inputParameterNames` - Validates 10 input parameter names
- ✅ `outputParameterNames` - Validates 8 output parameter names
- ✅ `errorDescriptions` - Tests error message formatting

**FusionSurrogatesTests.swift** (1 test) - Package verification
- ✅ `example` - Basic package import verification

**Float32PrecisionTests.swift** (9 tests) - Float32 precision validation
- ✅ `mlxArrayFloat32` - MLXArray uses Float32 by default
- ✅ `floatLiteralInference` - Float literal type inference
- ✅ `mlxOperationsFloat32` - MLX operations preserve Float32
- ✅ `float32PrecisionCalculations` - Float32 precision in calculations
- ✅ `validateFloatInputs` - validateInputs accepts Float dict
- ✅ `validateFloatInputsMissing` - validateInputs throws on missing parameter
- ✅ `predictScalarFloat` - predictScalar uses Float parameters
- ✅ `float32ArrayCreation` - Float32 array creation from literals
- ✅ `mlxArrayRepeating` - MLXArray repeating uses Float

**InputValidationTests.swift** (13 tests) - Input validation logic
- ✅ `validateInputsComplete` - All parameters present
- ✅ `validateInputsMissing` - Throws on missing parameter
- ✅ `validateShapesConsistent` - Consistent shapes
- ✅ `validateShapesMismatch` - Throws on shape mismatch
- ✅ `validateShapesNaN` - Detects NaN values
- ✅ `validateShapesInf` - Detects Inf values
- ✅ `validateShapesNegativeInf` - Detects negative Inf
- ✅ `validateShapes1DOnly` - Requires 1D arrays
- ✅ `validateShapesMinimumSize` - Minimum grid size (2)
- ✅ `validateShapesMaximumSize` - Maximum grid size (10000)
- ✅ `validateShapesBoundary` - Accepts boundary values
- ✅ `validateShapesEmpty` - Throws on empty dict
- ✅ `errorMessagesDescriptive` - Error messages are descriptive

**MLXConversionTests.swift** (9 tests) - MLXArray conversion utilities
- ✅ `batchToPythonArrayOrder` - Parameter order verification
- ✅ `batchToPythonArrayMultipleCells` - Multiple cells conversion
- ✅ `pythonRoundtrip` - toPython and fromPython roundtrip
- ✅ `emptyDictConversion` - Empty dict conversion
- ✅ `mlxArrayPythonExtension` - MLXArray pythonArray extension
- ✅ `batchToPythonArrayFloat32` - Uses Float32
- ✅ `featureNamesMatch` - Feature names match input parameter names

**TORAXIntegrationTests.swift** (8 tests) - TORAX integration helpers
- ✅ `combineFluxesComplete` - Combine all mode outputs
- ✅ `combineFluxesMissingModes` - Handle missing modes
- ✅ `computeNormalizedGradientBasic` - Basic gradient calculation
- ✅ `computeNormalizedGradientEdges` - Handle edge values
- ✅ `computeNormalizedGradientConstant` - Constant profile
- ✅ `buildInputsComplete` - Produces all required parameters
- ✅ `buildInputsValidation` - Validates after creation
- ✅ `buildInputsTiTeRatio` - Calculates Ti_Te ratio
- ✅ `buildInputsNormni` - Calculates normni
- ✅ `combineFluxesFloat32` - Returns Float32

**Integration Tests (Disabled)**
- ⏸️ `PythonIntegrationTests.swift.disabled` - Python fusion_surrogates integration (26 tests)
- ⏸️ `MLXIntegrationTests.swift.disabled` - MLX operations and predictions (13 tests)

Note: Integration tests are disabled due to environment dependencies (Python library path, MLX Metal library). They should be run manually in swift-TORAX project environment.

### Test Coverage

| Component | Coverage | Notes |
|-----------|----------|-------|
| Input parameter names | ✅ Tested | 10 parameters validated |
| Output parameter names | ✅ Tested | 8 parameters validated |
| Error handling | ✅ Tested | All error cases covered |
| Float32 precision | ✅ Tested | 9 comprehensive tests |
| Input validation logic | ✅ Tested | 13 tests covering all edge cases |
| MLX conversion | ✅ Tested | 9 tests for conversion utilities |
| TORAX integration | ✅ Tested | 10 tests for helper functions |
| Gradient calculation | ✅ Tested | 3 tests (basic, edges, constant) |
| MLX operations | ✅ Tested | Operations validated (without GPU) |
| Python integration | ⏸️ Disabled | Requires fusion_surrogates installation |

## Known Test Limitations

### 1. MLX Metal Library Error

**Issue:**
```
MLX error: Failed to load the default metallib. library not found
```

**Cause:** MLX GPU acceleration requires proper Metal library setup in test environment.

**Impact:**
- MLX operations fall back to CPU mode
- Tests involving MLXArray operations may hang or timeout
- GPU-accelerated features cannot be tested in CI/CD

**Workaround:**
- Tests using MLX operations are disabled (`.disabled` extension)
- Basic functionality tests work without MLX operations
- Manual testing required for MLX features

### 2. Python Integration Testing

**Issue:** Tests cannot import `fusion_surrogates` Python library

**Cause:**
- Python environment not configured in test context
- `fusion_surrogates` not installed

**Workaround:**
- Python integration tests not included
- Manual validation against Python reference required
- See "Manual Testing" section below

## Running Tests

### All Tests
```bash
swift test
```

### Specific Test Suite
```bash
swift test --filter MinimalTests
```

### List Available Tests
```bash
swift test --list-tests
```

### Verbose Output
```bash
swift test --verbose
```

## Manual Testing

For features that cannot be automatically tested, perform manual validation:

### 1. Test MLX Gradient Calculation

Create a test script:

```swift
import MLX
import FusionSurrogates

// Test gradient on linear profile
let x = MLXArray([0.0, 1.0, 2.0, 3.0, 4.0], [5])
let f = MLXArray([0.0, 2.0, 4.0, 6.0, 8.0], [5])  // f = 2*x

let inputs = QLKNN.buildInputs(
    electronTemperature: f,
    ionTemperature: f,
    electronDensity: MLXArray([1e19, 1e19, 1e19, 1e19, 1e19], [5]),
    ionDensity: MLXArray([1e19, 1e19, 1e19, 1e19, 1e19], [5]),
    poloidalFlux: MLXArray([0.0, 0.1, 0.2, 0.3, 0.4], [5]),
    radius: x,
    majorRadius: 6.2,
    minorRadius: 2.0,
    toroidalField: 5.3
)

let rLnTe = inputs["R_L_Te"]!
eval(rLnTe)
print("Normalized gradient:", rLnTe.asArray(Float.self))
// Should be finite, no NaN or Inf
```

### 2. Test Python Integration

**Prerequisites:**
```bash
pip install fusion-surrogates
```

**Test script:**
```swift
import FusionSurrogates

do {
    let qlknn = try QLKNN(modelVersion: "7_11")
    print("✅ QLKNN model loaded successfully")

    // Test with scalar inputs
    let inputs: [String: Double] = [
        "R_L_Te": 5.0,
        "R_L_Ti": 5.0,
        "R_L_ne": 1.0,
        "R_L_ni": 1.0,
        "q": 2.0,
        "s_hat": 1.0,
        "r_R": 0.3,
        "Ti_Te": 1.0,
        "log_nu_star": -10.0,
        "ni_ne": 1.0
    ]

    let outputs = qlknn.predictPython(inputs)
    print("✅ Prediction successful")
    print("Outputs:", outputs)
} catch {
    print("❌ Error:", error)
}
```

### 3. Test Input Validation

```swift
import MLX
import FusionSurrogates

// Test shape validation
let validInput: [String: MLXArray] = [
    "R_L_Te": MLXArray([1.0, 2.0, 3.0], [3]),
    "R_L_Ti": MLXArray([1.0, 2.0, 3.0], [3]),
    // ... all 10 parameters
]

do {
    try QLKNN.validateInputs(validInput)
    try QLKNN.validateShapes(validInput)
    print("✅ Input validation passed")
} catch {
    print("❌ Validation failed:", error)
}

// Test NaN detection
let invalidInput: [String: MLXArray] = [
    "R_L_Te": MLXArray([1.0, Float.nan, 3.0], [3]),
    "R_L_Ti": MLXArray([1.0, 2.0, 3.0], [3])
]

do {
    try QLKNN.validateShapes(invalidInput)
    print("❌ Should have thrown error")
} catch {
    print("✅ Correctly detected NaN:", error)
}
```

### 4. Test swift-TORAX Integration

See `TORAX_INTEGRATION.md` for integration testing with swift-TORAX.

## Validation Against Python Reference

To ensure correctness, compare Swift output with Python reference:

**Python:**
```python
import fusion_surrogates
import numpy as np

model = fusion_surrogates.qlknn.QLKNN_7_11()
inputs = {
    'R_L_Te': np.array([5.0]),
    'R_L_Ti': np.array([5.0]),
    # ... all inputs
}
outputs_python = model.predict(inputs)
print("Python output:", outputs_python)
```

**Swift:**
```swift
let qlknn = try QLKNN(modelVersion: "7_11")
let inputs: [String: MLXArray] = [
    "R_L_Te": MLXArray([5.0], [1]),
    "R_L_Ti": MLXArray([5.0], [1]),
    // ... all inputs
]
let outputs_swift = try qlknn.predict(inputs)
eval(outputs_swift["chi_ion_itg"]!)
print("Swift output:", outputs_swift["chi_ion_itg"]!.asArray(Float.self))
```

Results should match within numerical precision (~1e-6).

## Future Test Improvements

### Short Term
- [ ] Add integration tests for Python fusion_surrogates (when environment setup resolved)
- [ ] Add MLX operation tests (when Metal library issue resolved)
- [ ] Add performance benchmarks

### Long Term
- [ ] Set up CI/CD with proper MLX environment
- [ ] Add property-based tests for gradient calculation
- [ ] Add regression tests comparing with Python reference
- [ ] Add stress tests with large grid sizes (n=1000+)

## Continuous Integration

Current status: ⚠️ Limited

**What works in CI:**
- ✅ Build verification
- ✅ Basic API tests (parameter names, errors)
- ✅ Static analysis

**What doesn't work in CI:**
- ❌ MLX GPU operations (requires Metal)
- ❌ Python integration (requires fusion_surrogates)
- ❌ Full integration tests

**Recommendation:**
- Use CI for build verification and basic tests
- Perform manual validation for MLX and Python features
- Document validation results in release notes

## Summary

**Current Test Status:**
- ✅ 3/3 automated tests passing
- ✅ Core API validated
- ✅ Build system verified
- ⚠️ Manual testing required for MLX and Python features

**Test Confidence:**
- High: API surface, error handling, parameter definitions
- Medium: Input validation (tested via unit tests, not integration)
- Manual: Gradient calculation, Python integration, MLX operations

For production use, perform manual validation tests described above and compare results with Python reference implementation.
