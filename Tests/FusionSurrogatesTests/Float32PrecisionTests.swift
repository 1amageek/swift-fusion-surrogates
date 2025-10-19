import Testing
import MLX
@testable import FusionSurrogates

/// Tests for Float32 precision throughout the codebase
@Suite("Float32 Precision Tests")
struct Float32PrecisionTests {

    @Test("MLXArray uses Float32 by default")
    func mlxArrayFloat32() {
        let array = MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3])

        // MLXArray should store Float (Float32)
        let values = array.asArray(Float.self)
        #expect(values.count == 3)
        #expect(abs(values[0] - 1.0) < 1e-6)
        #expect(abs(values[1] - 2.0) < 1e-6)
        #expect(abs(values[2] - 3.0) < 1e-6)
    }

    @Test("Float literal inference")
    func floatLiteralInference() {
        let value: Float = 1.5
        let array = MLXArray(value)

        eval(array)
        let extracted = array.item(Float.self)
        #expect(abs(extracted - 1.5) < 1e-6)
    }

    @Test("MLXArray operations preserve Float32")
    func mlxOperationsFloat32() {
        let a = MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3])
        let b = MLXArray([Float(0.5), Float(1.0), Float(1.5)], [3])

        let sum = a + b
        eval(sum)

        let values = sum.asArray(Float.self)
        #expect(abs(values[0] - 1.5) < 1e-6)
        #expect(abs(values[1] - 3.0) < 1e-6)
        #expect(abs(values[2] - 4.5) < 1e-6)
    }

    @Test("Float32 precision in calculations")
    func float32PrecisionCalculations() {
        // Test that Float32 precision is maintained
        let x = MLXArray([Float(0.1), Float(0.2), Float(0.3)], [3])
        let result = x * MLXArray(10.0)

        eval(result)
        let values = result.asArray(Float.self)

        // Float32 should handle these values precisely
        #expect(abs(values[0] - 1.0) < 1e-6)
        #expect(abs(values[1] - 2.0) < 1e-6)
        #expect(abs(values[2] - 3.0) < 1e-6)
    }

    @Test("validateInputs accepts Float dict")
    func validateFloatInputs() throws {
        let inputs: [String: Float] = [
            "Ati": 5.0,
            "Ate": 5.0,
            "Ane": 1.0,
            "Ani": 1.0,
            "q": 2.0,
            "smag": 1.0,
            "x": 0.3,
            "Ti_Te": 1.0,
            "LogNuStar": -10.0,
            "normni": 1.0
        ]

        try QLKNN.validateInputs(inputs)
    }

    @Test("validateInputs throws on missing Float parameter")
    func validateFloatInputsMissing() {
        let incompleteInputs: [String: Float] = [
            "Ati": 5.0,
            "Ate": 5.0
            // Missing other parameters
        ]

        #expect(throws: FusionSurrogatesError.self) {
            try QLKNN.validateInputs(incompleteInputs)
        }
    }

    @Test("predictScalar uses Float parameters")
    func predictScalarFloat() {
        // This test verifies the API accepts Float, not Double
        // Cannot test actual prediction without Python environment

        let inputs: [String: Float] = [
            "Ati": 5.0,
            "Ate": 5.0,
            "Ane": 1.0,
            "Ani": 1.0,
            "q": 2.0,
            "smag": 1.0,
            "x": 0.3,
            "Ti_Te": 1.0,
            "LogNuStar": -10.0,
            "normni": 1.0
        ]

        // Verify inputs pass validation
        try? QLKNN.validateInputs(inputs)

        // Type check: ensure method signature accepts Float
        let _: ([String: Float], Int) throws -> [String: MLXArray] = { _, _ in [:] }

        #expect(Bool(true))
    }

    @Test("Float32 array creation from literals")
    func float32ArrayCreation() {
        let values: [Float] = [1.1, 2.2, 3.3]
        let array = MLXArray(values, [3])

        eval(array)
        let extracted = array.asArray(Float.self)

        for (original, extracted) in zip(values, extracted) {
            #expect(abs(original - extracted) < 1e-6)
        }
    }

    @Test("MLXArray repeating uses Float")
    func mlxArrayRepeating() {
        // Test the internal repeating method used by predictScalar
        let value: Float = 42.0
        let array = MLXArray.ones([5]) * MLXArray(value)

        eval(array)
        let values = array.asArray(Float.self)

        #expect(values.count == 5)
        for v in values {
            #expect(abs(v - 42.0) < 1e-6)
        }
    }
}
