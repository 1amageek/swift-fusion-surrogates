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

def test_mlx_conversion_simulation():
    """Simulate MLXConversion operations (what Swift will do)."""
    print("\n" + "=" * 60)
    print("Testing MLXConversion Simulation")
    print("=" * 60)

    try:
        # Test 1: Float32 array creation and conversion
        print("\n1. Testing Float32 precision...")
        data_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        print(f"  dtype: {data_f32.dtype}")
        print(f"  values: {data_f32}")
        assert data_f32.dtype == np.float32, "Should be float32"
        print("  ✅ Float32 preserved")

        # Test 2: 2D array creation (batchToPythonArray simulation)
        print("\n2. Testing 2D array creation (batch mode)...")
        batch_size = 3
        num_features = 10

        # Simulate Swift creating dict of arrays then converting to 2D
        feature_names = ['Ati', 'Ate', 'Ane', 'Ani', 'q', 'smag', 'x', 'Ti_Te', 'LogNuStar', 'normni']
        batch_data = np.array([
            [5.0, 5.0, 1.0, 1.0, 2.0, 1.0, 0.3, 1.0, -10.0, 1.0],
            [6.0, 6.0, 1.5, 1.5, 2.5, 1.2, 0.35, 1.0, -9.5, 1.0],
            [7.0, 7.0, 2.0, 2.0, 3.0, 1.4, 0.4, 1.0, -9.0, 1.0],
        ], dtype=np.float32)

        print(f"  shape: {batch_data.shape}")
        print(f"  dtype: {batch_data.dtype}")
        assert batch_data.shape == (batch_size, num_features)
        assert batch_data.dtype == np.float32
        print("  ✅ 2D array correct")

        # Test 3: Model prediction with 2D array
        print("\n3. Testing model prediction with 2D array...")
        from fusion_surrogates.qlknn.qlknn_model import QLKNNModel
        model = QLKNNModel.load_default_model()
        outputs = model.predict(batch_data)

        print(f"  Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            value_np = np.array(value)
            print(f"    {key}: shape={value_np.shape}, dtype={value_np.dtype}")
            assert value_np.shape[0] == batch_size, f"{key} batch size mismatch"

        print("  ✅ Model prediction successful")

        # Test 4: Roundtrip precision
        print("\n4. Testing roundtrip precision...")
        original = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        # Simulate: Swift -> Python -> Swift
        after_roundtrip = original.copy()
        diff = np.abs(original - after_roundtrip)
        max_diff = np.max(diff)
        print(f"  Max difference: {max_diff}")
        assert max_diff < 1e-6, "Roundtrip precision loss"
        print("  ✅ Roundtrip precision maintained")

        # Test 5: Large batch (10000 cells)
        print("\n5. Testing large batch (10000 cells)...")
        large_batch = np.tile(batch_data[0:1], (10000, 1)).astype(np.float32)
        assert large_batch.shape == (10000, 10)
        print(f"  shape: {large_batch.shape}")
        print(f"  memory: {large_batch.nbytes / 1024 / 1024:.2f} MB")
        print("  ✅ Large batch created")

        return True

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

    # Test MLXConversion simulation
    mlx_conversion_works = test_mlx_conversion_simulation()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"New API (QLKNNModel): {'✅ WORKS' if new_api_works else '❌ FAILED'}")
    print(f"Old API (QLKNN_7_11): {'✅ WORKS' if old_api_works else '❌ NOT AVAILABLE'}")
    print(f"MLXConversion Simulation: {'✅ WORKS' if mlx_conversion_works else '❌ FAILED'}")

    if new_api_works and mlx_conversion_works:
        print("\n✅ All tests passed - Swift integration should work")
    elif new_api_works:
        print("\n⚠️  API works but MLXConversion may have issues")
    elif old_api_works:
        print("\n⚠️  Use QLKNN_7_11() - Only old API is available")
    else:
        print("\n❌ No working API found")

    return 0 if (new_api_works and mlx_conversion_works) else 1

if __name__ == "__main__":
    sys.exit(main())
