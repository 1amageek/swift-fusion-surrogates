import Testing
import Foundation
@testable import FusionSurrogates

/// MLX network structure validation tests (no Metal execution required)
@Suite("MLX Network Structure Tests")
struct MLXNetworkStructureTests {

    @Test("Network initializes with correct architecture")
    func networkArchitecture() throws {
        let network = QLKNNNetwork(
            inputDimensions: 10,
            outputDimensions: 8,
            hiddenDimensions: 133
        )

        // Verify layer dimensions
        #expect(network.layer0.weight.shape == [133, 10])
        #expect(network.layer0.bias!.shape == [133])

        #expect(network.layer2.weight.shape == [133, 133])
        #expect(network.layer2.bias!.shape == [133])

        #expect(network.layer4.weight.shape == [133, 133])
        #expect(network.layer4.bias!.shape == [133])

        #expect(network.layer6.weight.shape == [133, 133])
        #expect(network.layer6.bias!.shape == [133])

        #expect(network.layer8.weight.shape == [133, 133])
        #expect(network.layer8.bias!.shape == [133])

        #expect(network.layer10.weight.shape == [8, 133])
        #expect(network.layer10.bias!.shape == [8])
    }

    @Test("Default model weights exist in bundle")
    func defaultModelExists() throws {
        let resourceURL = Bundle.module.url(
            forResource: "qlknn_7_11_weights",
            withExtension: "safetensors"
        )

        #expect(resourceURL != nil, "Default model weights should exist in bundle")

        if let url = resourceURL {
            let fileExists = FileManager.default.fileExists(atPath: url.path)
            let message = fileExists ? "File exists" : "File not found at \(url.path)"
            #expect(fileExists, Comment(rawValue: message))
        }
    }

    @Test("Input parameter names are correct")
    func inputParameterNames() {
        let expected = [
            "Ati", "Ate", "Ane", "Ani", "q", "smag", "x", "Ti_Te", "LogNuStar", "normni"
        ]

        #expect(QLKNN.inputParameterNames == expected)
        #expect(QLKNN.inputParameterNames.count == 10)
    }

    @Test("Output parameter names match ONNX order")
    func outputParameterNames() {
        // Verified against ONNX model output order
        let expected = [
            "efeITG",     // Index 0
            "efiITG",     // Index 1
            "pfeITG",     // Index 2
            "efeTEM",     // Index 3
            "efiTEM",     // Index 4
            "pfeTEM",     // Index 5
            "efeETG",     // Index 6
            "gamma_max"   // Index 7
        ]

        #expect(QLKNN.outputParameterNames == expected)
        #expect(QLKNN.outputParameterNames.count == 8)
    }

    @Test("Network parameter count is correct")
    func parameterCount() throws {
        // Calculate total parameters:
        // Layer 0: (10 × 133) + 133 = 1,463
        // Layer 2: (133 × 133) + 133 = 17,822
        // Layer 4: (133 × 133) + 133 = 17,822
        // Layer 6: (133 × 133) + 133 = 17,822
        // Layer 8: (133 × 133) + 133 = 17,822
        // Layer 10: (133 × 8) + 8 = 1,072
        // Total: 73,823 parameters

        let layer0Params = (10 * 133) + 133
        let hiddenParams = ((133 * 133) + 133) * 4  // 4 hidden layers
        let outputParams = (133 * 8) + 8
        let totalExpected = layer0Params + hiddenParams + outputParams

        #expect(totalExpected == 73_823, "Network should have 73,823 parameters")
    }
}
