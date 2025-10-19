#!/usr/bin/env python3
"""
Convert QLKNN ONNX model to SafeTensors format for MLX
SafeTensors is a safe, fast format supported by MLX natively
"""

import onnx
import numpy as np
from safetensors.numpy import save_file
from pathlib import Path
import json

def convert_onnx_to_safetensors(onnx_path: str, output_dir: str):
    """Convert ONNX model to SafeTensors format"""

    model = onnx.load(onnx_path)

    # Extract all weights
    weights = {}
    for initializer in model.graph.initializer:
        name = initializer.name
        # Convert ONNX tensor to numpy
        data = onnx.numpy_helper.to_array(initializer)

        # Convert to float32
        weights[name] = data.astype(np.float32)
        print(f"{name}: {data.shape} ({data.dtype} -> float32)")

    # Save as SafeTensors
    output_path = Path(output_dir) / "qlknn_7_11_weights.safetensors"
    save_file(weights, output_path)
    print(f"\n✅ Saved SafeTensors to {output_path}")
    print(f"   Total size: {output_path.stat().st_size / 1024:.2f} KB")

    # Save metadata
    metadata = {
        'model_name': 'qlknn_7_11',
        'architecture': {
            'layers': [
                {'type': 'Linear', 'in': 10, 'out': 133, 'activation': 'ReLU'},
                {'type': 'Linear', 'in': 133, 'out': 133, 'activation': 'ReLU'},
                {'type': 'Linear', 'in': 133, 'out': 133, 'activation': 'ReLU'},
                {'type': 'Linear', 'in': 133, 'out': 133, 'activation': 'ReLU'},
                {'type': 'Linear', 'in': 133, 'out': 133, 'activation': 'ReLU'},
                {'type': 'Linear', 'in': 133, 'out': 8, 'activation': None}
            ]
        },
        'input_names': ['Ati', 'Ate', 'Ane', 'Ani', 'q', 'smag', 'x', 'Ti_Te', 'LogNuStar', 'normni'],
        'output_names': ['efiITG', 'efeITG', 'pfeITG', 'efeTEM', 'efiTEM', 'pfeTEM', 'efeETG', 'gamma_max'],
        'weight_names': list(weights.keys()),
        'precision': 'float32'
    }

    metadata_path = Path(output_dir) / "qlknn_7_11_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {metadata_path}")

    # Display model info
    print("\n=== Model Architecture ===")
    print(f"Input: {metadata['input_names']}")
    print(f"Output: {metadata['output_names']}")
    print(f"Layers: {len(metadata['architecture']['layers'])}")
    print(f"Parameters: {sum(w.size for w in weights.values()):,}")

if __name__ == '__main__':
    import sys

    # Default paths
    onnx_path = '/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/fusion_surrogates/qlknn/models/qlknn_7_11.onnx'
    output_dir = 'Resources'

    if len(sys.argv) > 1:
        onnx_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print(f"Converting {onnx_path}")
    print(f"       to {output_dir}/\n")

    # Install safetensors if needed
    try:
        import safetensors
    except ImportError:
        print("Installing safetensors...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "--quiet"])
        from safetensors.numpy import save_file

    convert_onnx_to_safetensors(onnx_path, output_dir)
