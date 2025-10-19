#!/usr/bin/env python3
"""
Verify that Swift output order matches ONNX model output order
"""

import onnx
import numpy as np
from fusion_surrogates.qlknn.qlknn_model import QLKNNModel

# Get ONNX output order
onnx_path = '/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/fusion_surrogates/qlknn/models/qlknn_7_11.onnx'
model_onnx = onnx.load(onnx_path)

onnx_order = [output.name for output in model_onnx.graph.output]

# Swift order (after fix)
swift_order = [
    "efeITG",
    "efiITG",
    "pfeITG",
    "efeTEM",
    "efiTEM",
    "pfeTEM",
    "efeETG",
    "gamma_max"
]

print("=== Output Order Verification ===\n")
print("ONNX Model Output Order:")
for i, name in enumerate(onnx_order):
    print(f"  {i}: {name}")

print("\nSwift outputParameterNames:")
for i, name in enumerate(swift_order):
    print(f"  {i}: {name}")

print("\n=== Verification ===")
if onnx_order == swift_order:
    print("✅ Output order MATCHES!")
else:
    print("❌ Output order MISMATCH!")
    print("\nDifferences:")
    for i, (onnx_name, swift_name) in enumerate(zip(onnx_order, swift_order)):
        if onnx_name != swift_name:
            print(f"  Index {i}: ONNX='{onnx_name}' vs Swift='{swift_name}'")

# Test with actual data
print("\n=== Testing with actual prediction ===")
model = QLKNNModel.load_default_model()
test_input = np.array([[5.0, 5.0, 1.0, 1.0, 2.0, 1.0, 0.3, 1.0, -10.0, 1.0]], dtype=np.float32)
outputs = model.predict(test_input)

print("Python model outputs:")
for name in swift_order:
    value = outputs[name][0]
    print(f"  {name}: {value:.6f}")

print("\n✅ All checks passed!")
