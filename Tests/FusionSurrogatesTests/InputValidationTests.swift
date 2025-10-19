import Testing
import MLX
@testable import FusionSurrogates

/// Tests for input validation logic
@Suite("Input Validation Tests")
struct InputValidationTests {

    @Test("validateInputs with all parameters")
    func validateInputsComplete() throws {
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

        try QLKNN.validateInputs(inputs)
    }

    @Test("validateInputs throws on missing parameter")
    func validateInputsMissing() {
        let incompleteInputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(5.0)], [1]),
            "Ate": MLXArray([Float(5.0)], [1])
            // Missing 8 parameters
        ]

        #expect(throws: FusionSurrogatesError.self) {
            try QLKNN.validateInputs(incompleteInputs)
        }
    }

    @Test("validateShapes with consistent shapes")
    func validateShapesConsistent() throws {
        print("🔍 [validateShapesConsistent] Starting test")

        print("🔍 Creating MLXArray for Ati...")
        let ati = MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3])
        print("✅ Ati created: shape \(ati.shape)")

        print("🔍 Creating MLXArray for Ate...")
        let ate = MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3])
        print("✅ Ate created: shape \(ate.shape)")

        print("🔍 Creating remaining arrays...")
        let inputs: [String: MLXArray] = [
            "Ati": ati,
            "Ate": ate,
            "Ane": MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3]),
            "Ani": MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3]),
            "q": MLXArray([Float(2.0), Float(2.0), Float(2.0)], [3]),
            "smag": MLXArray([Float(1.0), Float(1.0), Float(1.0)], [3]),
            "x": MLXArray([Float(0.3), Float(0.3), Float(0.3)], [3]),
            "Ti_Te": MLXArray([Float(1.0), Float(1.0), Float(1.0)], [3]),
            "LogNuStar": MLXArray([Float(-10.0), Float(-10.0), Float(-10.0)], [3]),
            "normni": MLXArray([Float(1.0), Float(1.0), Float(1.0)], [3])
        ]
        print("✅ All inputs created, count: \(inputs.count)")

        print("🔍 Calling QLKNN.validateShapes()...")
        try QLKNN.validateShapes(inputs)
        print("✅ validateShapes completed successfully")
    }

    @Test("validateShapes throws on shape mismatch")
    func validateShapesMismatch() {
        let mismatchedInputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3]),
            "Ate": MLXArray([Float(1.0), Float(2.0)], [2])  // Different shape
        ]

        #expect(throws: FusionSurrogatesError.self) {
            try QLKNN.validateShapes(mismatchedInputs)
        }
    }

    @Test("validateShapes detects NaN")
    func validateShapesNaN() {
        let nanInputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(1.0), Float.nan, Float(3.0)], [3]),
            "Ate": MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3])
        ]

        #expect(throws: FusionSurrogatesError.self) {
            try QLKNN.validateShapes(nanInputs)
        }
    }

    @Test("validateShapes detects Inf")
    func validateShapesInf() {
        let infInputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(1.0), Float.infinity, Float(3.0)], [3]),
            "Ate": MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3])
        ]

        #expect(throws: FusionSurrogatesError.self) {
            try QLKNN.validateShapes(infInputs)
        }
    }

    @Test("validateShapes detects negative Inf")
    func validateShapesNegativeInf() {
        let negInfInputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(1.0), -Float.infinity, Float(3.0)], [3]),
            "Ate": MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3])
        ]

        #expect(throws: FusionSurrogatesError.self) {
            try QLKNN.validateShapes(negInfInputs)
        }
    }

    @Test("validateShapes requires 1D arrays")
    func validateShapes1DOnly() {
        // 2D array should be rejected
        let array2D = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0)], [2, 2])

        let inputs2D: [String: MLXArray] = [
            "Ati": array2D
        ]

        #expect(throws: FusionSurrogatesError.self) {
            try QLKNN.validateShapes(inputs2D)
        }
    }

    @Test("validateShapes minimum grid size")
    func validateShapesMinimumSize() {
        // Grid size of 1 should be rejected (minimum is 2)
        let tooSmall: [String: MLXArray] = [
            "Ati": MLXArray([Float(1.0)], [1]),
            "Ate": MLXArray([Float(1.0)], [1])
        ]

        #expect(throws: FusionSurrogatesError.self) {
            try QLKNN.validateShapes(tooSmall)
        }
    }

    @Test("validateShapes maximum grid size")
    func validateShapesMaximumSize() {
        // Grid size of 10001 should be rejected (maximum is 10000)
        let tooLarge: [String: MLXArray] = [
            "Ati": MLXArray.repeating(1.0, count: 10001),
            "Ate": MLXArray.repeating(1.0, count: 10001)
        ]

        #expect(throws: FusionSurrogatesError.self) {
            try QLKNN.validateShapes(tooLarge)
        }
    }

    @Test("validateShapes accepts boundary values")
    func validateShapesBoundary() throws {
        // Test minimum valid size (2)
        let minSize: [String: MLXArray] = [
            "Ati": MLXArray([Float(1.0), Float(2.0)], [2]),
            "Ate": MLXArray([Float(1.0), Float(2.0)], [2]),
            "Ane": MLXArray([Float(1.0), Float(2.0)], [2]),
            "Ani": MLXArray([Float(1.0), Float(2.0)], [2]),
            "q": MLXArray([Float(2.0), Float(2.0)], [2]),
            "smag": MLXArray([Float(1.0), Float(1.0)], [2]),
            "x": MLXArray([Float(0.3), Float(0.3)], [2]),
            "Ti_Te": MLXArray([Float(1.0), Float(1.0)], [2]),
            "LogNuStar": MLXArray([Float(-10.0), Float(-10.0)], [2]),
            "normni": MLXArray([Float(1.0), Float(1.0)], [2])
        ]

        try QLKNN.validateShapes(minSize)

        // Test maximum valid size (10000)
        let maxSize: [String: MLXArray] = [
            "Ati": MLXArray.repeating(1.0, count: 10000),
            "Ate": MLXArray.repeating(1.0, count: 10000),
            "Ane": MLXArray.repeating(1.0, count: 10000),
            "Ani": MLXArray.repeating(1.0, count: 10000),
            "q": MLXArray.repeating(2.0, count: 10000),
            "smag": MLXArray.repeating(1.0, count: 10000),
            "x": MLXArray.repeating(0.3, count: 10000),
            "Ti_Te": MLXArray.repeating(1.0, count: 10000),
            "LogNuStar": MLXArray.repeating(-10.0, count: 10000),
            "normni": MLXArray.repeating(1.0, count: 10000)
        ]

        try QLKNN.validateShapes(maxSize)
    }

    @Test("validateShapes throws on empty dict")
    func validateShapesEmpty() {
        let empty: [String: MLXArray] = [:]

        #expect(throws: FusionSurrogatesError.self) {
            try QLKNN.validateShapes(empty)
        }
    }

    @Test("Error messages are descriptive")
    func errorMessagesDescriptive() {
        let missingParam: [String: MLXArray] = [:]

        do {
            try QLKNN.validateInputs(missingParam)
            #expect(Bool(false), "Should have thrown error")
        } catch let error as FusionSurrogatesError {
            let message = error.localizedDescription
            #expect(message.contains("Missing required input parameter"))
        } catch {
            #expect(Bool(false), "Wrong error type")
        }
    }
}
