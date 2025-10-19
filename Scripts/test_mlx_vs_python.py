#!/usr/bin/env python3
"""
Generate expected outputs for MLX network tests
Runs Python fusion_surrogates and outputs values for Swift test comparison
"""

import numpy as np
from fusion_surrogates.qlknn.qlknn_model import QLKNNModel
import json

print("=== Generating Expected Outputs for MLX Tests ===\n")

# Load Python model
print("Loading Python QLKNNModel...")
model = QLKNNModel.load_default_model()
print("✅ Model loaded\n")

# Test cases (matching Swift tests)
test_cases = {
    "single_sample": np.array([
        [5.0, 5.0, 1.0, 1.0, 2.0, 1.0, 0.3, 1.0, -10.0, 1.0]
    ], dtype=np.float32),

    "three_samples": np.array([
        [5.0, 5.0, 1.0, 1.0, 2.0, 1.0, 0.3, 1.0, -10.0, 1.0],
        [6.0, 6.0, 1.5, 1.5, 2.5, 1.2, 0.35, 1.0, -9.5, 1.0],
        [7.0, 7.0, 2.0, 2.0, 3.0, 1.4, 0.4, 1.0, -9.0, 1.0],
    ], dtype=np.float32),

    "realistic_plasma": np.array([
        [6.0, 6.0, 2.0, 2.0, 2.5, 1.5, 0.4, 1.2, -8.0, 1.0]
    ], dtype=np.float32),

    "batch_varying": np.array([
        [2.0, 2.0, 1.0, 1.0, 1.5, 0.5, 0.2, 0.8, -12.0, 0.9],
        [5.0, 5.0, 2.0, 2.0, 2.0, 1.0, 0.3, 1.0, -10.0, 1.0],
        [10.0, 10.0, 3.0, 3.0, 2.5, 1.5, 0.4, 1.2, -8.0, 1.0],
        [15.0, 15.0, 4.0, 4.0, 3.0, 2.0, 0.5, 1.5, -6.0, 1.0],
        [20.0, 20.0, 5.0, 5.0, 3.5, 2.5, 0.6, 2.0, -4.0, 0.95],
    ], dtype=np.float32),
}

expected_outputs = {}

for test_name, inputs in test_cases.items():
    print(f"\n=== {test_name} ===")
    outputs = model.predict(inputs)

    # Convert to Python native types for JSON
    result = {}
    for key in sorted(outputs.keys()):
        values = np.array(outputs[key]).flatten()
        result[key] = [float(v) for v in values]
        print(f"{key}: {result[key]}")

    expected_outputs[test_name] = result

# Save to JSON for Swift tests to reference
output_file = "Tests/FusionSurrogatesTests/expected_outputs.json"
with open(output_file, 'w') as f:
    json.dump(expected_outputs, f, indent=2)

print(f"\n✅ Expected outputs saved to {output_file}")
print("\nYou can now run Swift tests:")
print("  swift test --filter MLXNetworkTests")
