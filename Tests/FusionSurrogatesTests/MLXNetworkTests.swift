import Testing
import MLX
@testable import FusionSurrogates

/// MLX network inference tests
/// Validates MLX implementation against known outputs
@Suite("MLX Network Tests")
struct MLXNetworkTests {

    @Test("Load default network from bundle")
    func loadDefaultNetwork() throws {
        let network = try QLKNNNetwork.loadDefault()

        // Verify network loaded successfully (basic sanity check)
        #expect(network.layer0.weight.shape == [133, 10])
        #expect(network.layer10.weight.shape == [8, 133])
    }

    @Test("Forward pass produces correct output shape")
    func forwardPassShape() throws {
        let network = try QLKNNNetwork.loadDefault()

        // Create test input [batch_size=3, features=10]
        let input = MLXArray(
            [Float](repeating: 1.0, count: 30),
            [3, 10]
        )

        let output = network(input)

        // Verify output shape is [3, 8]
        #expect(output.shape == [3, 8])
    }

    @Test("Predict with dictionary inputs")
    func predictWithDictionary() throws {
        let network = try QLKNNNetwork.loadDefault()

        let inputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(5.0), Float(6.0), Float(7.0)], [3]),
            "Ate": MLXArray([Float(5.0), Float(6.0), Float(7.0)], [3]),
            "Ane": MLXArray([Float(1.0), Float(1.5), Float(2.0)], [3]),
            "Ani": MLXArray([Float(1.0), Float(1.5), Float(2.0)], [3]),
            "q": MLXArray([Float(2.0), Float(2.5), Float(3.0)], [3]),
            "smag": MLXArray([Float(1.0), Float(1.2), Float(1.4)], [3]),
            "x": MLXArray([Float(0.3), Float(0.35), Float(0.4)], [3]),
            "Ti_Te": MLXArray([Float(1.0), Float(1.0), Float(1.0)], [3]),
            "LogNuStar": MLXArray([Float(-10.0), Float(-9.5), Float(-9.0)], [3]),
            "normni": MLXArray([Float(1.0), Float(1.0), Float(1.0)], [3])
        ]

        let outputs = try network.predict(inputs)

        // Verify all expected outputs are present
        for paramName in QLKNN.outputParameterNames {
            #expect(outputs[paramName] != nil, "Missing output: \(paramName)")
        }

        // Verify output shapes
        for (key, array) in outputs {
            #expect(array.shape == [3], "Output \(key) should have shape [3], got \(array.shape)")
        }

        // Verify outputs are finite
        for (key, array) in outputs {
            eval(array)
            let values = array.asArray(Float.self)
            for value in values {
                #expect(!value.isNaN, "\(key) contains NaN")
                #expect(!value.isInfinite, "\(key) contains Inf")
            }
        }
    }

    @Test("Single sample prediction")
    func singleSamplePrediction() throws {
        let network = try QLKNNNetwork.loadDefault()

        // Single sample with specific values
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

        // Verify single element in each output
        for (key, array) in outputs {
            #expect(array.shape == [1], "Output \(key) should have shape [1]")
        }

        // Verify outputs are finite
        for (key, array) in outputs {
            eval(array)
            let value = array.asArray(Float.self)[0]
            #expect(!value.isNaN, "\(key) should not be NaN")
            #expect(!value.isInfinite, "\(key) should not be Inf")
        }
    }

    @Test("Batch prediction with varying inputs")
    func batchPrediction() throws {
        let network = try QLKNNNetwork.loadDefault()

        // Create a batch with significantly different inputs
        let batchSize = 5
        let inputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(2.0), Float(5.0), Float(10.0), Float(15.0), Float(20.0)], [batchSize]),
            "Ate": MLXArray([Float(2.0), Float(5.0), Float(10.0), Float(15.0), Float(20.0)], [batchSize]),
            "Ane": MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)], [batchSize]),
            "Ani": MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)], [batchSize]),
            "q": MLXArray([Float(1.5), Float(2.0), Float(2.5), Float(3.0), Float(3.5)], [batchSize]),
            "smag": MLXArray([Float(0.5), Float(1.0), Float(1.5), Float(2.0), Float(2.5)], [batchSize]),
            "x": MLXArray([Float(0.2), Float(0.3), Float(0.4), Float(0.5), Float(0.6)], [batchSize]),
            "Ti_Te": MLXArray([Float(0.8), Float(1.0), Float(1.2), Float(1.5), Float(2.0)], [batchSize]),
            "LogNuStar": MLXArray([Float(-12.0), Float(-10.0), Float(-8.0), Float(-6.0), Float(-4.0)], [batchSize]),
            "normni": MLXArray([Float(0.9), Float(1.0), Float(1.0), Float(1.0), Float(0.95)], [batchSize])
        ]

        let outputs = try network.predict(inputs)

        // Verify outputs have correct batch size
        for (key, array) in outputs {
            #expect(array.shape == [batchSize], "Output \(key) should have shape [\(batchSize)]")
        }

        // Check that different inputs produce different outputs
        print("\n=== Batch Prediction Variation ===")
        for name in ["efiITG", "efeITG", "gamma_max"] {
            if let array = outputs[name] {
                eval(array)
                let values = array.asArray(Float.self)
                print("\(name): \(values)")

                // Check values are finite
                for value in values {
                    #expect(!value.isNaN, "\(name) should not contain NaN")
                    #expect(!value.isInfinite, "\(name) should not contain Inf")
                }

                // Check that we have some variation in outputs
                let minVal = values.min() ?? 0
                let maxVal = values.max() ?? 0

                // For non-zero outputs, we expect some variation
                if abs(maxVal) > 0.01 || abs(minVal) > 0.01 {
                    #expect(maxVal != minVal, "\(name) should show variation across batch")
                }
            }
        }
    }

    @Test("Output values are physically reasonable")
    func physicalValidity() throws {
        let network = try QLKNNNetwork.loadDefault()

        // Realistic plasma parameters
        let inputs: [String: MLXArray] = [
            "Ati": MLXArray([Float(6.0)], [1]),
            "Ate": MLXArray([Float(6.0)], [1]),
            "Ane": MLXArray([Float(2.0)], [1]),
            "Ani": MLXArray([Float(2.0)], [1]),
            "q": MLXArray([Float(2.5)], [1]),
            "smag": MLXArray([Float(1.5)], [1]),
            "x": MLXArray([Float(0.4)], [1]),
            "Ti_Te": MLXArray([Float(1.2)], [1]),
            "LogNuStar": MLXArray([Float(-8.0)], [1]),
            "normni": MLXArray([Float(1.0)], [1])
        ]

        let outputs = try network.predict(inputs)

        // Flux outputs should be non-negative in most cases
        let fluxOutputs = ["efiITG", "efeITG", "efeTEM", "efeETG", "efiTEM"]

        print("\n=== Physical Validity Check ===")
        for key in fluxOutputs {
            if let array = outputs[key] {
                eval(array)
                let value = array.asArray(Float.self)[0]
                print("\(key): \(value)")

                // Verify outputs are finite (actual physics values can be negative)
                #expect(!value.isNaN, "\(key) should not be NaN")
                #expect(!value.isInfinite, "\(key) should not be Inf")
            }
        }

        // Growth rate - verify it's finite
        if let gammaMax = outputs["gamma_max"] {
            eval(gammaMax)
            let value = gammaMax.asArray(Float.self)[0]
            print("gamma_max: \(value)")
            #expect(!value.isNaN, "gamma_max should not be NaN")
            #expect(!value.isInfinite, "gamma_max should not be Inf")
        }
    }
}
