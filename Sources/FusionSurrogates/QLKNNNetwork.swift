import Foundation
import MLX
import MLXNN

/// QLKNN neural network implemented in MLX
///
/// Architecture: 5 hidden layers (133 units each) + output layer (8 outputs)
/// Activation: ReLU
/// Input: 10 parameters [Ati, Ate, Ane, Ani, q, smag, x, Ti_Te, LogNuStar, normni]
/// Output: 8 fluxes [efiITG, efeITG, pfeITG, efeTEM, efiTEM, pfeTEM, efeETG, gamma_max]
public class QLKNNNetwork: Module, UnaryLayer {

    // Network layers
    @ModuleInfo var layer0: Linear
    @ModuleInfo var layer2: Linear
    @ModuleInfo var layer4: Linear
    @ModuleInfo var layer6: Linear
    @ModuleInfo var layer8: Linear
    @ModuleInfo var layer10: Linear

    /// Initialize network with random weights (for testing)
    public init(inputDimensions: Int = 10, outputDimensions: Int = 8, hiddenDimensions: Int = 133) {
        self.layer0 = Linear(inputDimensions, hiddenDimensions)
        self.layer2 = Linear(hiddenDimensions, hiddenDimensions)
        self.layer4 = Linear(hiddenDimensions, hiddenDimensions)
        self.layer6 = Linear(hiddenDimensions, hiddenDimensions)
        self.layer8 = Linear(hiddenDimensions, hiddenDimensions)
        self.layer10 = Linear(hiddenDimensions, outputDimensions)
        super.init()
    }

    /// Load default network from bundled resources
    public static func loadDefault() throws -> QLKNNNetwork {
        guard let resourceURL = Bundle.module.url(forResource: "qlknn_7_11_weights", withExtension: "safetensors") else {
            throw QLKNNError.modelNotFound("Default model weights not found in bundle")
        }

        return try load(weightsPath: resourceURL.path)
    }

    /// Load network from SafeTensors weights file
    public static func load(weightsPath: String) throws -> QLKNNNetwork {
        guard FileManager.default.fileExists(atPath: weightsPath) else {
            throw QLKNNError.modelNotFound("Weights file not found: \(weightsPath)")
        }

        // Load weights directly using MLX
        let url = URL(fileURLWithPath: weightsPath)
        let loadedWeights = try MLX.loadArrays(url: url)

        // Create network
        let network = QLKNNNetwork()

        // Build flattened parameters dictionary
        var params: [String: MLXArray] = [:]

        // Map ONNX layer names to our layer names
        params["layer0.weight"] = loadedWeights["_network.model.0.weight"]!
        params["layer0.bias"] = loadedWeights["_network.model.0.bias"]!

        params["layer2.weight"] = loadedWeights["_network.model.2.weight"]!
        params["layer2.bias"] = loadedWeights["_network.model.2.bias"]!

        params["layer4.weight"] = loadedWeights["_network.model.4.weight"]!
        params["layer4.bias"] = loadedWeights["_network.model.4.bias"]!

        params["layer6.weight"] = loadedWeights["_network.model.6.weight"]!
        params["layer6.bias"] = loadedWeights["_network.model.6.bias"]!

        params["layer8.weight"] = loadedWeights["_network.model.8.weight"]!
        params["layer8.bias"] = loadedWeights["_network.model.8.bias"]!

        params["layer10.weight"] = loadedWeights["_network.model.10.weight"]!
        params["layer10.bias"] = loadedWeights["_network.model.10.bias"]!

        // Update network with loaded parameters
        network.update(parameters: ModuleParameters.unflattened(params))

        return network
    }

    /// Forward pass
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = input

        // Layer 0: Dense(10 → 133) + ReLU
        x = layer0(x)
        x = relu(x)

        // Layer 2: Dense(133 → 133) + ReLU
        x = layer2(x)
        x = relu(x)

        // Layer 4: Dense(133 → 133) + ReLU
        x = layer4(x)
        x = relu(x)

        // Layer 6: Dense(133 → 133) + ReLU
        x = layer6(x)
        x = relu(x)

        // Layer 8: Dense(133 → 133) + ReLU
        x = layer8(x)
        x = relu(x)

        // Layer 10: Dense(133 → 8) - no activation (linear output)
        x = layer10(x)

        return x
    }

    /// Predict outputs from inputs
    /// - Parameter inputs: Dictionary of input arrays [String: MLXArray]
    /// - Returns: Dictionary of output arrays [String: MLXArray]
    public func predict(_ inputs: [String: MLXArray]) throws -> [String: MLXArray] {
        // Convert inputs to 2D array [batch_size, 10]
        let batchInput = Self.batchToInputArray(inputs)

        // Forward pass
        let outputs = self(batchInput)  // [batch_size, 8]

        // Split outputs into individual fluxes
        let outputNames = QLKNN.outputParameterNames
        var result: [String: MLXArray] = [:]

        for (idx, name) in outputNames.enumerated() {
            // Extract column idx from outputs: [batch_size, 8] -> [batch_size]
            let column = outputs[0..., idx]
            // column is already 1D, no need to squeeze
            result[name] = column
        }

        return result
    }

    /// Convert dictionary inputs to 2D array for network input
    static func batchToInputArray(_ inputs: [String: MLXArray]) -> MLXArray {
        let inputNames = QLKNN.inputParameterNames

        var columns: [MLXArray] = []
        for name in inputNames {
            guard let array = inputs[name] else {
                fatalError("Missing input parameter: \(name)")
            }
            // Ensure shape is [batch_size, 1] for stacking
            let reshaped = reshaped(array, [-1, 1])
            columns.append(reshaped)
        }

        // Stack columns: [batch, features]
        return concatenated(columns, axis: 1)
    }
}

// MARK: - Error Types

public enum QLKNNError: Error {
    case modelNotFound(String)
    case invalidWeights(String)
    case missingParameter(String)
}
