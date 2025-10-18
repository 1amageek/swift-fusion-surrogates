#!/usr/bin/env python3
"""Verify fusion_surrogates Python API and discover the correct usage."""

import sys
import numpy as np

def test_qlknn_model():
    """Test the new QLKNNModel API."""
    print("=" * 60)
    print("Testing QLKNNModel (New API)")
    print("=" * 60)

    try:
        from fusion_surrogates.qlknn.qlknn_model import QLKNNModel
        from fusion_surrogates.qlknn.models import registry

        print(f"✅ Import successful")
        print(f"Default model: {registry.DEFAULT_MODEL_NAME}")
        print(f"Available models: {list(registry.MODELS.keys())}")
        print(f"Available ONNX models: {list(registry.ONNX_MODELS.keys())}")

        # Create model instance using load_default_model
        print("\nCreating QLKNNModel instance...")
        model = QLKNNModel.load_default_model()
        print(f"✅ Model created successfully")

        # Check model attributes
        print(f"\nModel attributes:")
        if hasattr(model, 'config'):
            print(f"  - config: {model.config}")
        if hasattr(model, 'metadata'):
            print(f"  - metadata: {model.metadata}")

        # Prepare test inputs - must be in the order specified by config.input_names
        print("\nPreparing test inputs...")
        print(f"Expected input order: {model.config.input_names}")

        # Create a 2D array: (batch_size, num_features)
        # Order: ['Ati', 'Ate', 'Ane', 'Ani', 'q', 'smag', 'x', 'Ti_Te', 'LogNuStar', 'normni']
        inputs = np.array([
            [5.0, 5.0, 1.0, 1.0, 2.0, 1.0, 0.3, 1.0, -3.0, 1.0],    # Sample 1
            [6.0, 6.0, 1.5, 1.5, 2.5, 1.2, 0.35, 1.0, -2.5, 1.0],   # Sample 2
            [7.0, 7.0, 2.0, 2.0, 3.0, 1.4, 0.4, 1.0, -2.0, 1.0],    # Sample 3
        ])

        print(f"Input shape: {inputs.shape} (batch_size, num_features)")
        print(f"Number of features: {model.num_inputs}")

        # Run prediction using predict() method
        print("\nRunning prediction...")
        outputs = model.predict(inputs)

        print(f"✅ Prediction successful!")
        print(f"\nOutput keys: {list(outputs.keys())}")

        for key, value in outputs.items():
            # Convert JAX array to numpy for printing
            value_np = np.array(value)
            print(f"  {key}: shape={value_np.shape}, sample={float(value_np[0]):.6f}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_old_api():
    """Test if the old QLKNN_7_11 API still exists."""
    print("\n" + "=" * 60)
    print("Testing Old API (QLKNN_7_11)")
    print("=" * 60)

    try:
        from fusion_surrogates.qlknn import QLKNN_7_11
        print("✅ Old API (QLKNN_7_11) still available")

        model = QLKNN_7_11()
        print("✅ Old model created successfully")
        return True

    except ImportError as e:
        print(f"❌ Old API not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Python version:", sys.version)
    print()

    # Test new API
    new_api_works = test_qlknn_model()

    # Test old API
    old_api_works = test_old_api()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"New API (QLKNNModel): {'✅ WORKS' if new_api_works else '❌ FAILED'}")
    print(f"Old API (QLKNN_7_11): {'✅ WORKS' if old_api_works else '❌ NOT AVAILABLE'}")

    if new_api_works:
        print("\n✅ Use QLKNNModel() - New API is available and working")
    elif old_api_works:
        print("\n⚠️  Use QLKNN_7_11() - Only old API is available")
    else:
        print("\n❌ No working API found")

    return 0 if (new_api_works or old_api_works) else 1

if __name__ == "__main__":
    sys.exit(main())
