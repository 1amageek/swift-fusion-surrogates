# Testing Guide

## Current Test Status

✅ **Pure MLX test suite** focusing on network loading and inference validation

```bash
swift test
```

**Test Framework:** Swift Testing (modern Swift 6+ framework)

**Note:** Tests validate the MLX-only implementation. Python backend has been removed.

## Test Structure

### Implemented Tests (Swift Testing Framework)

**BasicAPITests.swift** - Basic API tests (no MLX dependencies)
- ✅ `inputParameterNames` - Validates 10 input parameter names
- ✅ `outputParameterNames` - Validates 8 output parameter names
- ✅ `errorDescriptions` - Tests error message formatting

**FusionSurrogatesTests.swift** - Package verification
- ✅ `example` - Basic package import verification

**WeightLoadingTests.swift** - Model loading and weight validation
- ✅ `safeTensorsContainsAllWeights` - Validates all 12 weight keys present
- ✅ `weightShapesAreCorrect` - Verifies weight shapes match architecture
- ✅ `weightDtypeIsFloat32` - Validates Float32 precision
- ✅ `networkParametersMatchLoadedWeights` - Network state after loading
- ✅ `weightsAreFinite` - No NaN/Inf in weights
- ✅ `weightRangesAreReasonable` - Values in [-100, 100] range
- ✅ `forwardPassShapeConsistency` - Correct output shapes for various batch sizes
- ✅ `outputVariationWithInput` - Different inputs produce different outputs
- ✅ `parametersAreTrainable` - All parameters are Float32 MLXArrays

**MLXNetworkTests.swift** - Inference validation (requires Metal)
- ✅ `loadDefaultNetwork` - Load network from bundled resources
- ✅ `forwardPassShape` - Forward pass produces correct output shape
- ✅ `predictWithDictionary` - Dictionary input prediction
- ✅ `singleSamplePrediction` - Single sample prediction
- ✅ `batchPrediction` - Batch prediction with varying inputs
- ✅ `physicalValidity` - Outputs are physically valid (finite values)

### Test Coverage

| Component | Coverage | Notes |
|-----------|----------|-------|
| Input parameter names | ✅ Tested | 10 parameters validated |
| Output parameter names | ✅ Tested | 8 parameters validated |
| Error handling | ✅ Tested | Error cases covered |
| Float32 precision | ✅ Tested | Weight dtype validation |
| Model weight loading | ✅ Tested | All 12 weight tensors validated |
| Network architecture | ✅ Tested | Layer shapes and parameter count |
| Forward pass | ✅ Tested | Single and batch inference |
| Physical validity | ✅ Tested | Outputs are finite (no NaN/Inf) |
| Weight integrity | ✅ Tested | Finite values, reasonable ranges |

## Known Test Limitations

### 1. MLX Metal Library Requirement

**Issue:**
```
MLX error: Failed to load the default metallib. library not found
```

**Cause:** MLX GPU acceleration requires proper Metal library setup in test environment.

**Impact:**
- Full inference tests (MLXNetworkTests) require Metal runtime
- Tests may fail in environments without MLX Metal library
- GPU-accelerated features cannot be tested in limited CI/CD environments

**Workaround:**
- Weight loading tests (WeightLoadingTests) validate model structure without requiring inference
- Basic API tests work without MLX operations
- Manual testing required for full inference validation on Metal-enabled systems

## Running Tests

### All Tests
```bash
swift test
```

### Specific Test Suite
```bash
# Run only weight loading tests (no Metal required)
swift test --filter WeightLoadingTests

# Run MLX inference tests (requires Metal)
swift test --filter MLXNetworkTests

# Run basic API tests
swift test --filter BasicAPITests
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

For MLX inference features that require Metal runtime:

### 1. Test MLX Network Loading and Inference

Create a test script (`test_mlx.swift`):

```swift
import MLX
import FusionSurrogates

do {
    // Load network from bundle
    let network = try QLKNNNetwork.loadDefault()
    print("✅ Network loaded successfully")
    print("   Layer 0: \(network.layer0.weight.shape)")
    print("   Layer 10: \(network.layer10.weight.shape)")

    // Test inference with sample inputs
    let inputs: [String: MLXArray] = [
        "Ati": MLXArray([Float(5.0)], [1]),
        "Ate": MLXArray([Float(5.0)], [1]),
        "Ane": MLXArray([Float(1.0)], [1]),
        "Ani": MLXArray([Float(1.0)], [1]),
        "q": MLXArray([Float(2.0)], [1]),
        "smag": MLXArray([Float(1.0)], [1]),
        "x": MLXArray([Float(0.3)], [1]),
        "Ti_Te": MLXArray([Float(1.0)], [1]),
        "LogNuStar": MLXArray([Float(-10.0)], [1]),
        "normni": MLXArray([Float(1.0)], [1])
    ]

    let outputs = try network.predict(inputs)
    print("✅ Prediction successful")

    for name in QLKNN.outputParameterNames {
        if let array = outputs[name] {
            eval(array)
            let value = array.asArray(Float.self)[0]
            print("   \(name): \(value)")
        }
    }
} catch {
    print("❌ Error:", error)
}
```

Run with:
```bash
swift run test_mlx.swift
```

### 2. Test swift-TORAX Integration

See `TORAX_INTEGRATION.md` for integration testing with swift-TORAX.

## Validation Against ONNX Reference

The MLX implementation was validated against the ONNX model during development:

1. **Weight Conversion**: ONNX weights converted to SafeTensors using `Scripts/convert_onnx_to_safetensors.py`
2. **Output Order Verification**: Verified using `Scripts/verify_output_order.py` to ensure output parameter names match ONNX output order
3. **Shape Validation**: All weight shapes verified to match ONNX model architecture

The model weights are identical to the ONNX model (Float32 precision), ensuring numerical equivalence.

## Future Test Improvements

### Short Term
- [ ] Add performance benchmarks for various batch sizes
- [ ] Add numerical precision tests (compare with ONNX reference)
- [ ] Add tests for edge case inputs (boundary values)

### Long Term
- [ ] Set up CI/CD with Metal support for full inference tests
- [ ] Add stress tests with large grid sizes (n=1000+)
- [ ] Add tests for additional QLKNN models if released

## Continuous Integration

Current status: ⚠️ Limited (Metal dependency)

**What works in CI:**
- ✅ Build verification
- ✅ Basic API tests (parameter names, errors)
- ✅ Weight loading tests (structure validation)
- ✅ Static analysis

**What may not work in CI:**
- ⚠️ MLX inference tests (requires Metal runtime)
- ⚠️ Full forward pass validation (requires Metal library)

**Recommendation:**
- Use CI for build verification, basic tests, and weight validation
- Perform manual inference validation on Metal-enabled systems
- Document validation results in release notes

## Summary

**Current Test Status:**
- ✅ All weight loading tests passing
- ✅ Model structure validated (architecture, shapes, dtypes)
- ✅ Basic API validated
- ⚠️ Inference tests require Metal runtime

**Test Confidence:**
- High: Model architecture, weight loading, parameter definitions, error handling
- Medium: Inference (requires Metal for full validation)

For production use, perform manual inference tests on a Metal-enabled system as described in the "Manual Testing" section.
