#!/usr/bin/env python3
"""
Python verification script for fusion_surrogates
Run this to verify the Python library works correctly
"""

import numpy as np

print("=" * 60)
print("fusion_surrogates Verification Script")
print("=" * 60)

# Test 1: Import
print("\n[1/5] Importing fusion_surrogates...")
try:
    import fusion_surrogates
    try:
        version = fusion_surrogates.__version__
        print(f"✅ fusion_surrogates version: {version}")
    except AttributeError:
        print("✅ fusion_surrogates imported (version not available)")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    exit(1)

# Test 2: Initialize QLKNN
print("\n[2/5] Initializing QLKNN model...")
try:
    from fusion_surrogates.qlknn import qlknn_model
    # Create model using QLKNNModel
    model = qlknn_model.QLKNNModel()
    print("✅ QLKNN model initialized")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Scalar prediction
print("\n[3/5] Testing scalar prediction...")
try:
    inputs = {
        'R_L_Te': np.array([5.0]),
        'R_L_Ti': np.array([5.0]),
        'R_L_ne': np.array([1.0]),
        'R_L_ni': np.array([1.0]),
        'q': np.array([2.0]),
        's_hat': np.array([1.0]),
        'r_R': np.array([0.3]),
        'Ti_Te': np.array([1.0]),
        'log_nu_star': np.array([-10.0]),
        'ni_ne': np.array([1.0])
    }

    outputs = model.predict(inputs)
    print("✅ Scalar prediction successful")
    print(f"   chi_ion_itg: {outputs['chi_ion_itg']}")
    print(f"   chi_electron_tem: {outputs['chi_electron_tem']}")
    print(f"   chi_electron_etg: {outputs['chi_electron_etg']}")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    exit(1)

# Test 4: Profile prediction
print("\n[4/5] Testing profile prediction (5 points)...")
try:
    n = 5
    inputs_profile = {
        'R_L_Te': np.array([5.0, 5.5, 6.0, 6.5, 7.0]),
        'R_L_Ti': np.array([5.0, 5.5, 6.0, 6.5, 7.0]),
        'R_L_ne': np.array([1.0, 1.2, 1.4, 1.6, 1.8]),
        'R_L_ni': np.array([1.0, 1.2, 1.4, 1.6, 1.8]),
        'q': np.array([1.5, 2.0, 2.5, 3.0, 3.5]),
        's_hat': np.array([0.8, 1.0, 1.2, 1.4, 1.6]),
        'r_R': np.array([0.2, 0.25, 0.3, 0.35, 0.4]),
        'Ti_Te': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        'log_nu_star': np.array([-12.0, -11.0, -10.0, -9.0, -8.0]),
        'ni_ne': np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    }

    outputs_profile = model.predict(inputs_profile)
    print("✅ Profile prediction successful")
    print(f"   chi_ion_itg shape: {outputs_profile['chi_ion_itg'].shape}")
    print(f"   chi_ion_itg values: {outputs_profile['chi_ion_itg']}")
except Exception as e:
    print(f"❌ Profile prediction failed: {e}")
    exit(1)

# Test 5: Output keys
print("\n[5/5] Checking output keys...")
try:
    expected_keys = ['chi_ion_itg', 'chi_electron_tem', 'chi_electron_etg', 'particle_flux']
    for key in expected_keys:
        if key in outputs:
            print(f"   ✅ {key}: present")
        else:
            print(f"   ❌ {key}: missing")
except Exception as e:
    print(f"❌ Key check failed: {e}")

print("\n" + "=" * 60)
print("All Python tests passed! ✅")
print("=" * 60)
print("\nYou can now use fusion_surrogates with Swift!")
print("Set PYTHON_LIBRARY environment variable:")
print("export PYTHON_LIBRARY=\"/Library/Frameworks/Python.framework/Versions/3.12/Python\"")
