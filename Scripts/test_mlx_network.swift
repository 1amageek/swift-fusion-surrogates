#!/usr/bin/env swift

import Foundation
import MLX
@testable import FusionSurrogates

print("=== Testing MLX QLKNN Network ===\n")

// Load the network
print("Loading QLKNN network from NPZ weights...")
let weightsPath = "Resources/qlknn_7_11_weights.npz"

do {
    let network = try QLKNNNetwork.load(weightsPath: weightsPath)
    print("✅ Network loaded successfully\n")

    // Create test inputs
    print("Creating test inputs...")
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

    // Run prediction
    print("Running prediction...")
    let outputs = try network.predict(inputs)

    // Display results
    print("\n=== Outputs ===")
    for (name, array) in outputs.sorted(by: { $0.key < $1.key }) {
        eval(array)
        let values = array.asArray(Float.self)
        print("\(name): \(values)")
    }

    print("\n✅ Test completed successfully!")

} catch {
    print("❌ Error: \(error)")
}
