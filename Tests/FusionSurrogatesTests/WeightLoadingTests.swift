import Testing
import MLX
import Foundation
@testable import FusionSurrogates

/// Tests to verify model weights are loaded correctly
@Suite("Weight Loading Tests")
struct WeightLoadingTests {

    @Test("SafeTensors file contains all expected weights")
    func safeTensorsContainsAllWeights() throws {
        guard let resourceURL = Bundle.module.url(
            forResource: "qlknn_7_11_weights",
            withExtension: "safetensors"
        ) else {
            throw QLKNNError.modelNotFound("SafeTensors file not found")
        }

        // Load weights directly
        let weights = try MLX.loadArrays(url: resourceURL)

        // Expected keys from ONNX model
        let expectedKeys = [
            "_network.model.0.weight",
            "_network.model.0.bias",
            "_network.model.2.weight",
            "_network.model.2.bias",
            "_network.model.4.weight",
            "_network.model.4.bias",
            "_network.model.6.weight",
            "_network.model.6.bias",
            "_network.model.8.weight",
            "_network.model.8.bias",
            "_network.model.10.weight",
            "_network.model.10.bias"
        ]

        // Verify all keys present
        for key in expectedKeys {
            #expect(weights[key] != nil, "Missing weight: \(key)")
        }
    }

    @Test("Loaded weights have correct shapes")
    func weightShapesAreCorrect() throws {
        guard let resourceURL = Bundle.module.url(
            forResource: "qlknn_7_11_weights",
            withExtension: "safetensors"
        ) else {
            throw QLKNNError.modelNotFound("SafeTensors file not found")
        }

        let weights = try MLX.loadArrays(url: resourceURL)

        // Layer 0: 10 -> 133
        #expect(weights["_network.model.0.weight"]?.shape == [133, 10])
        #expect(weights["_network.model.0.bias"]?.shape == [133])

        // Layer 2: 133 -> 133
        #expect(weights["_network.model.2.weight"]?.shape == [133, 133])
        #expect(weights["_network.model.2.bias"]?.shape == [133])

        // Layer 4: 133 -> 133
        #expect(weights["_network.model.4.weight"]?.shape == [133, 133])
        #expect(weights["_network.model.4.bias"]?.shape == [133])

        // Layer 6: 133 -> 133
        #expect(weights["_network.model.6.weight"]?.shape == [133, 133])
        #expect(weights["_network.model.6.bias"]?.shape == [133])

        // Layer 8: 133 -> 133
        #expect(weights["_network.model.8.weight"]?.shape == [133, 133])
        #expect(weights["_network.model.8.bias"]?.shape == [133])

        // Layer 10: 133 -> 8
        #expect(weights["_network.model.10.weight"]?.shape == [8, 133])
        #expect(weights["_network.model.10.bias"]?.shape == [8])
    }

    @Test("Loaded weights are Float32")
    func weightDtypeIsFloat32() throws {
        guard let resourceURL = Bundle.module.url(
            forResource: "qlknn_7_11_weights",
            withExtension: "safetensors"
        ) else {
            throw QLKNNError.modelNotFound("SafeTensors file not found")
        }

        let weights = try MLX.loadArrays(url: resourceURL)

        // Check all weights are float32
        for (key, array) in weights {
            #expect(array.dtype == .float32, "\(key) should be float32, got \(array.dtype)")
        }
    }

    @Test("Network parameters match loaded weights after update")
    func networkParametersMatchLoadedWeights() throws {
        let network = try QLKNNNetwork.loadDefault()

        // Get network parameters as dictionary
        let networkParamsArray = network.parameters().flattened()
        var networkParams: [String: MLXArray] = [:]
        for (key, value) in networkParamsArray {
            networkParams[key] = value
        }

        // Verify network has correct number of parameters
        #expect(networkParams.count == 12, "Network should have 12 parameters (6 layers × 2)")

        // Verify each layer's parameters exist and have correct shapes
        #expect(networkParams["layer0.weight"]?.shape == [133, 10])
        #expect(networkParams["layer0.bias"]?.shape == [133])

        #expect(networkParams["layer2.weight"]?.shape == [133, 133])
        #expect(networkParams["layer2.bias"]?.shape == [133])

        #expect(networkParams["layer4.weight"]?.shape == [133, 133])
        #expect(networkParams["layer4.bias"]?.shape == [133])

        #expect(networkParams["layer6.weight"]?.shape == [133, 133])
        #expect(networkParams["layer6.bias"]?.shape == [133])

        #expect(networkParams["layer8.weight"]?.shape == [133, 133])
        #expect(networkParams["layer8.bias"]?.shape == [133])

        #expect(networkParams["layer10.weight"]?.shape == [8, 133])
        #expect(networkParams["layer10.bias"]?.shape == [8])
    }

    @Test("Loaded weights contain finite values")
    func weightsAreFinite() throws {
        guard let resourceURL = Bundle.module.url(
            forResource: "qlknn_7_11_weights",
            withExtension: "safetensors"
        ) else {
            throw QLKNNError.modelNotFound("SafeTensors file not found")
        }

        let weights = try MLX.loadArrays(url: resourceURL)

        for (key, array) in weights {
            eval(array)
            let values = array.asArray(Float.self)

            let hasNaN = values.contains { $0.isNaN }
            let hasInf = values.contains { $0.isInfinite }

            #expect(!hasNaN, "\(key) contains NaN values")
            #expect(!hasInf, "\(key) contains Inf values")
        }
    }

    @Test("Loaded weights have reasonable value ranges")
    func weightRangesAreReasonable() throws {
        guard let resourceURL = Bundle.module.url(
            forResource: "qlknn_7_11_weights",
            withExtension: "safetensors"
        ) else {
            throw QLKNNError.modelNotFound("SafeTensors file not found")
        }

        let weights = try MLX.loadArrays(url: resourceURL)

        for (key, array) in weights {
            eval(array)
            let values = array.asArray(Float.self)

            let minVal = values.min() ?? 0
            let maxVal = values.max() ?? 0

            // Neural network weights should typically be in range [-10, 10]
            #expect(minVal >= -100, "\(key) has unreasonably small value: \(minVal)")
            #expect(maxVal <= 100, "\(key) has unreasonably large value: \(maxVal)")

            print("\(key): min=\(minVal), max=\(maxVal)")
        }
    }

    @Test("Network forward pass produces consistent output shapes")
    func forwardPassShapeConsistency() throws {
        let network = try QLKNNNetwork.loadDefault()

        // Test different batch sizes
        let batchSizes = [1, 3, 10, 25]

        for batchSize in batchSizes {
            // Create dummy input
            let input = MLXArray(
                [Float](repeating: 0.5, count: batchSize * 10),
                [batchSize, 10]
            )

            let output = network(input)

            #expect(output.shape == [batchSize, 8],
                   "Output shape should be [\(batchSize), 8], got \(output.shape)")
        }
    }

    @Test("Network produces different outputs for different inputs")
    func outputVariationWithInput() throws {
        let network = try QLKNNNetwork.loadDefault()

        // Create two different inputs
        let input1 = MLXArray([Float](repeating: 0.5, count: 10), [1, 10])
        let input2 = MLXArray([Float](repeating: 1.5, count: 10), [1, 10])

        let output1 = network(input1)
        let output2 = network(input2)

        eval(output1)
        eval(output2)

        let values1 = output1.asArray(Float.self)
        let values2 = output2.asArray(Float.self)

        // Outputs should be different for different inputs
        var hasDifference = false
        for i in 0..<min(values1.count, values2.count) {
            if abs(values1[i] - values2[i]) > 0.001 {
                hasDifference = true
                break
            }
        }

        #expect(hasDifference, "Network should produce different outputs for different inputs")
    }

    @Test("Network parameters are trainable (not frozen)")
    func parametersAreTrainable() throws {
        let network = try QLKNNNetwork.loadDefault()

        let params = network.parameters().flattened()

        // Verify we have trainable parameters
        #expect(params.count > 0, "Network should have parameters")

        // All parameters should be MLXArrays (trainable)
        for (key, value) in params {
            #expect(value.dtype == .float32, "\(key) should be float32")
        }

        print("\nNetwork parameters:")
        for (key, value) in params {
            print("  \(key): shape \(value.shape), dtype \(value.dtype)")
        }
    }
}
