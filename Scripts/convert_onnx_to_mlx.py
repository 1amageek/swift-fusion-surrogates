#!/usr/bin/env python3
"""
Convert QLKNN ONNX model to MLX-compatible format (NumPy .npz)
"""

import onnx
import numpy as np
from pathlib import Path

def convert_onnx_to_mlx(onnx_path: str, output_path: str):
    """Convert ONNX model weights to NPZ format for MLX"""

    model = onnx.load(onnx_path)

    # Extract all weights
    weights = {}
    for initializer in model.graph.initializer:
        name = initializer.name
        # Convert ONNX tensor to numpy
        data = onnx.numpy_helper.to_array(initializer)
        weights[name] = data.astype(np.float32)  # Ensure float32
        print(f"{name}: {data.shape} ({data.dtype} -> float32)")

    # Extract model architecture info
    print("\n=== Model Architecture ===")
    print(f"Input: {model.graph.input[0].name}")
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")

    print(f"\nOutputs:")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")

    # Save to NPZ
    np.savez(output_path, **weights)
    print(f"\n✅ Saved weights to {output_path}")
    print(f"   Total size: {Path(output_path).stat().st_size / 1024:.2f} KB")

    # Also save metadata
    metadata = {
        'input_names': ['Ati', 'Ate', 'Ane', 'Ani', 'q', 'smag', 'x', 'Ti_Te', 'LogNuStar', 'normni'],
        'output_names': ['efiITG', 'efeITG', 'pfeITG', 'efeTEM', 'efiTEM', 'pfeTEM', 'efeETG', 'gamma_max'],
        'num_layers': len([n for n in model.graph.node if n.op_type == 'Gemm' or n.op_type == 'MatMul'])
    }

    import json
    metadata_path = output_path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {metadata_path}")

if __name__ == '__main__':
    import sys

    # Default paths
    onnx_path = '/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/fusion_surrogates/qlknn/models/qlknn_7_11.onnx'
    output_path = 'Resources/qlknn_7_11_weights.npz'

    if len(sys.argv) > 1:
        onnx_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    print(f"Converting {onnx_path}")
    print(f"       to {output_path}\n")

    convert_onnx_to_mlx(onnx_path, output_path)
