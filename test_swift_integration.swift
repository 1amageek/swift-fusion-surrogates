#!/usr/bin/env swift

import Foundation
import FusionSurrogates
import MLX

print("=" * 60)
print("Testing Swift-Python Integration with New QLKNN API")
print("=" * 60)

do {
    // 1. Initialize QLKNN model
    print("\n1. Loading QLKNN model...")
    let qlknn = try QLKNN(modelName: "qlknn_7_11_v1")
    print("✅ Model loaded successfully")

    // Check configuration
    print("\n2. Model configuration:")
    let config = qlknn.config
    print("   Input names: \(config.input_names)")
    print("   Target names: \(config.target_names)")

    // 3. Prepare test inputs
    print("\n3. Preparing test inputs...")
    let batchSize = 3
    let inputs: [String: MLXArray] = [
        "Ati": MLXArray([5.0, 6.0, 7.0], [batchSize]),
        "Ate": MLXArray([5.0, 6.0, 7.0], [batchSize]),
        "Ane": MLXArray([1.0, 1.5, 2.0], [batchSize]),
        "Ani": MLXArray([1.0, 1.5, 2.0], [batchSize]),
        "q": MLXArray([2.0, 2.5, 3.0], [batchSize]),
        "smag": MLXArray([1.0, 1.2, 1.4], [batchSize]),
        "x": MLXArray([0.3, 0.35, 0.4], [batchSize]),
        "Ti_Te": MLXArray([1.0, 1.0, 1.0], [batchSize]),
        "LogNuStar": MLXArray([-3.0, -2.5, -2.0], [batchSize]),
        "normni": MLXArray([1.0, 1.0, 1.0], [batchSize])
    ]

    print("   ✅ Created \(inputs.count) input parameters")
    print("   ✅ Batch size: \(batchSize)")

    // 4. Validate inputs
    print("\n4. Validating inputs...")
    try QLKNN.validateInputs(inputs)
    try QLKNN.validateShapes(inputs)
    print("   ✅ Input validation passed")

    // 5. Run prediction
    print("\n5. Running prediction...")
    let outputs = try qlknn.predict(inputs)
    print("   ✅ Prediction successful!")
    print("   ✅ Got \(outputs.count) output parameters")

    // 6. Display outputs
    print("\n6. Output parameters:")
    for (key, value) in outputs.sorted(by: { $0.key < $1.key }) {
        eval(value)
        let values = value.asArray(Float.self)
        print("   \(key): [\(values[0]), \(values[1]), \(values[2])]")
    }

    // 7. Combine fluxes
    print("\n7. Combining fluxes for TORAX...")
    let combined = TORAXIntegration.combineFluxes(outputs)
    print("   ✅ Combined \(combined.count) transport coefficients")

    for (key, value) in combined.sorted(by: { $0.key < $1.key }) {
        eval(value)
        let values = value.asArray(Float.self)
        if values.count >= 3 {
            print("   \(key): [\(values[0]), \(values[1]), \(values[2])]")
        } else if values.count > 0 {
            print("   \(key): \(values)")
        }
    }

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)

} catch {
    print("\n❌ Error: \(error)")
    exit(1)
}
