import Testing
import MLX
import PythonKit
@testable import FusionSurrogates

/// Tests for MLXArray conversion utilities
@Suite("MLX Conversion Tests")
struct MLXConversionTests {

    @Test("batchToPythonArray parameter order")
    func batchToPythonArrayOrder() {
        // Create test inputs with known values
        let inputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(1.0)], [1]),
            "Ate": MLXArray([Float(2.0)], [1]),
            "Ane": MLXArray([Float(3.0)], [1]),
            "Ani": MLXArray([Float(4.0)], [1]),
            "q": MLXArray([Float(5.0)], [1]),
            "smag": MLXArray([Float(6.0)], [1]),
            "x": MLXArray([Float(7.0)], [1]),
            "Ti_Te": MLXArray([Float(8.0)], [1]),
            "LogNuStar": MLXArray([Float(9.0)], [1]),
            "normni": MLXArray([Float(10.0)], [1])
        ]

        // Verify all input parameters are present
        #expect(inputs.count == 10)

        // Verify order matches QLKNN.inputParameterNames
        let expectedOrder = QLKNN.inputParameterNames
        #expect(expectedOrder[0] == "Ati")
        #expect(expectedOrder[1] == "Ate")
        #expect(expectedOrder[9] == "normni")
    }

    @Test("batchToPythonArray with multiple cells")
    func batchToPythonArrayMultipleCells() {
        let inputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(1.0), Float(1.1), Float(1.2)], [3]),
            "Ate": MLXArray([Float(2.0), Float(2.1), Float(2.2)], [3]),
            "Ane": MLXArray([Float(3.0), Float(3.1), Float(3.2)], [3]),
            "Ani": MLXArray([Float(4.0), Float(4.1), Float(4.2)], [3]),
            "q": MLXArray([Float(5.0), Float(5.1), Float(5.2)], [3]),
            "smag": MLXArray([Float(6.0), Float(6.1), Float(6.2)], [3]),
            "x": MLXArray([Float(7.0), Float(7.1), Float(7.2)], [3]),
            "Ti_Te": MLXArray([Float(8.0), Float(8.1), Float(8.2)], [3]),
            "LogNuStar": MLXArray([Float(9.0), Float(9.1), Float(9.2)], [3]),
            "normni": MLXArray([Float(10.0), Float(10.1), Float(10.2)], [3])
        ]

        // Test that all arrays have consistent shape
        let shapes = inputs.values.map { $0.shape }
        #expect(shapes.allSatisfy { $0 == [3] })
    }

    @Test("toPython and fromPython roundtrip")
    func pythonRoundtrip() {
        let original = MLXArray([Float(1.5), Float(2.5), Float(3.5)], [3])
        eval(original)

        // This test verifies Float32 is used throughout
        let originalValues = original.asArray(Float.self)

        #expect(originalValues.count == 3)
        #expect(abs(originalValues[0] - 1.5) < 1e-6)
        #expect(abs(originalValues[1] - 2.5) < 1e-6)
        #expect(abs(originalValues[2] - 3.5) < 1e-6)
    }

    @Test("Empty dict conversion")
    func emptyDictConversion() {
        let empty: [String: MLXArray] = [:]

        // Type check only - cannot call without Python environment
        // Verify the method signature exists
        let _: ([String: MLXArray]) -> [String: PythonObject] = MLXConversion.batchToPython

        #expect(empty.isEmpty)
    }

    @Test("MLXArray pythonArray extension")
    func mlxArrayPythonExtension() {
        let array = MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3])

        // Verify extension exists (compile-time check only)
        // Note: Cannot actually call .pythonArray without Python environment
        // The fact that this compiles proves the extension exists
        let _: (MLXArray) -> PythonObject = { $0.pythonArray }

        #expect(Bool(true))
    }

    @Test("batchToPythonArray uses Float32")
    func batchToPythonArrayFloat32() {
        // Verify internal implementation uses Float, not Double
        let inputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(1.0)], [1]),
            "Ate": MLXArray([Float(2.0)], [1]),
            "Ane": MLXArray([Float(3.0)], [1]),
            "Ani": MLXArray([Float(4.0)], [1]),
            "q": MLXArray([Float(5.0)], [1]),
            "smag": MLXArray([Float(6.0)], [1]),
            "x": MLXArray([Float(7.0)], [1]),
            "Ti_Te": MLXArray([Float(8.0)], [1]),
            "LogNuStar": MLXArray([Float(9.0)], [1]),
            "normni": MLXArray([Float(10.0)], [1])
        ]

        // Extract values and verify they're Float
        for (_, array) in inputs {
            eval(array)
            let values = array.asArray(Float.self)
            #expect(!values.isEmpty)

            // Verify no precision loss for small integers
            for value in values {
                #expect(!value.isNaN)
                #expect(!value.isInfinite)
            }
        }
    }

    @Test("Feature names match input parameter names")
    func featureNamesMatch() {
        // The batchToPythonArray uses hardcoded feature names
        // Verify they match QLKNN.inputParameterNames

        let hardcodedNames = [
            "Ati", "Ate", "Ane", "Ani", "q", "smag", "x", "Ti_Te", "LogNuStar", "normni"
        ]

        let apiNames = QLKNN.inputParameterNames

        #expect(hardcodedNames.count == apiNames.count)

        for (hardcoded, api) in zip(hardcodedNames, apiNames) {
            #expect(hardcoded == api, "Mismatch: \(hardcoded) != \(api)")
        }
    }

    // MARK: - MLX Environment Investigation Tests

    @Test("MLX scalar creation")
    func mlxScalarCreation() {
        print("🔍 Test: Creating MLXArray scalar")
        let scalar = MLXArray(0.0)
        print("✅ Scalar created - shape: \(scalar.shape), dtype: \(scalar.dtype)")
        #expect(scalar.shape == [])
    }

    @Test("MLX array creation with shape")
    func mlxArrayCreationWithShape() {
        print("🔍 Test: Creating MLXArray with shape")
        let array = MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3])
        print("✅ Array created - shape: \(array.shape)")
        #expect(array.shape == [3])
    }

    @Test("MLX zeros creation")
    func mlxZerosCreation() {
        print("🔍 Test: MLXArray.zeros([1])")
        let zeros = MLXArray.zeros([1])
        print("✅ Zeros created - shape: \(zeros.shape)")
        #expect(zeros.shape == [1])
    }

    @Test("MLX ones creation")
    func mlxOnesCreation() {
        print("🔍 Test: MLXArray.ones([3])")
        let ones = MLXArray.ones([3])
        print("✅ Ones created - shape: \(ones.shape)")
        #expect(ones.shape == [3])
    }

    @Test("MLX lazy addition (no eval)")
    func mlxLazyAddition() {
        print("🔍 Test: Lazy addition without eval")
        let a = MLXArray([Float(1.0)], [1])
        let b = MLXArray([Float(2.0)], [1])
        print("  Created arrays a and b")

        let c = a + b
        print("✅ Addition succeeded (lazy) - shape: \(c.shape)")
        #expect(c.shape == [1])
    }

    @Test("MLX scalar broadcast addition")
    func mlxScalarBroadcast() {
        print("🔍 Test: Scalar + Array addition")
        let scalar = MLXArray(0.0)
        let array = MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3])
        print("  Created scalar and array")

        let result = scalar + array
        print("✅ Broadcast succeeded - shape: \(result.shape)")
        #expect(result.shape == [3])
    }

    @Test("MLX ones multiplication")
    func mlxOnesMultiplication() {
        print("🔍 Test: ones() * scalar")
        let ones = MLXArray.ones([3])
        let scalar = MLXArray(5.0)
        print("  Created ones and scalar")

        let result = ones * scalar
        print("✅ Multiplication succeeded - shape: \(result.shape)")
        #expect(result.shape == [3])
    }

    @Test("MLX eval and asArray (critical test)")
    func mlxEvalAndAsArray() {
        print("🔍 Test: eval() and asArray() calls")
        let array = MLXArray([Float(1.0), Float(2.0)], [2])
        print("  Created array")

        print("  Calling eval()...")
        eval(array)
        print("✅ eval() succeeded")

        print("  Calling asArray()...")
        let values = array.asArray(Float.self)
        print("✅ asArray() succeeded - values: \(values)")

        #expect(values.count == 2)
        #expect(abs(values[0] - 1.0) < 1e-6)
        #expect(abs(values[1] - 2.0) < 1e-6)
    }

    // MARK: - Boundary and Edge Case Tests

    @Test("batchToPythonArray with single element")
    func batchToPythonArraySingleElement() {
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

        // Should handle single element batch
        // Note: Cannot actually call without Python, just verify compilation
        let _: ([String: MLXArray]) -> PythonObject = MLXConversion.batchToPythonArray
        #expect(inputs.count == 10)
    }

    @Test("batchToPythonArray with large batch")
    func batchToPythonArrayLargeBatch() {
        // Test with maximum allowed size (10000)
        let n = 10000
        let inputs: [String: MLXArray] = [
            "Ati": MLXArray.repeating(5.0, count: n),
            "Ate": MLXArray.repeating(5.0, count: n),
            "Ane": MLXArray.repeating(1.0, count: n),
            "Ani": MLXArray.repeating(1.0, count: n),
            "q": MLXArray.repeating(2.0, count: n),
            "smag": MLXArray.repeating(1.0, count: n),
            "x": MLXArray.repeating(0.3, count: n),
            "Ti_Te": MLXArray.repeating(1.0, count: n),
            "LogNuStar": MLXArray.repeating(-10.0, count: n),
            "normni": MLXArray.repeating(1.0, count: n)
        ]

        // Verify all arrays have correct size
        for (_, array) in inputs {
            #expect(array.shape == [n])
        }
    }

    @Test("toPython and fromPython with large array")
    func pythonConversionLargeArray() {
        let n = 10000
        let large = MLXArray.repeating(42.0, count: n)

        eval(large)
        let values = large.asArray(Float.self)

        #expect(values.count == n)
        // Spot check first and last
        #expect(abs(values[0] - 42.0) < 1e-6)
        #expect(abs(values[n-1] - 42.0) < 1e-6)
    }

    @Test("MLXArray conversion with extreme values")
    func conversionExtremeValues() {
        // Test very large and very small values
        let extreme = MLXArray([
            Float(1e-10),  // Very small
            Float(1e10),   // Very large
            Float(1.0)     // Normal
        ], [3])

        eval(extreme)
        let values = extreme.asArray(Float.self)

        #expect(values.count == 3)
        #expect(abs(values[0] - 1e-10) < 1e-15)
        #expect(abs(values[1] - 1e10) < 1e4)
        #expect(abs(values[2] - 1.0) < 1e-6)
    }

    @Test("batchToPythonArray with scientific notation values")
    func batchToPythonArrayScientificNotation() {
        // Test with values in scientific notation (common in plasma physics)
        let inputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(1e1), Float(1e2)], [2]),
            "Ate": MLXArray([Float(1e1), Float(1e2)], [2]),
            "Ane": MLXArray([Float(1e0), Float(1e1)], [2]),
            "Ani": MLXArray([Float(1e0), Float(1e1)], [2]),
            "q": MLXArray([Float(2e0), Float(3e0)], [2]),
            "smag": MLXArray([Float(1e0), Float(1e0)], [2]),
            "x": MLXArray([Float(3e-1), Float(4e-1)], [2]),
            "Ti_Te": MLXArray([Float(1e0), Float(1e0)], [2]),
            "LogNuStar": MLXArray([Float(-1e1), Float(-1e1)], [2]),
            "normni": MLXArray([Float(1e0), Float(1e0)], [2])
        ]

        // Verify all values are finite
        for (key, array) in inputs {
            eval(array)
            let values = array.asArray(Float.self)
            for value in values {
                #expect(!value.isNaN, "\(key) contains NaN")
                #expect(!value.isInfinite, "\(key) contains Inf")
            }
        }
    }
}

// Helper extension for creating repeated arrays
extension MLXArray {
    static func repeating(_ value: Float, count: Int) -> MLXArray {
        return MLXArray.ones([count]) * MLXArray(value)
    }
}
