#!/usr/bin/env swift

import Foundation
import PythonKit

// Test 1: Import fusion_surrogates
print("=" * 60)
print("Test 1: Import fusion_surrogates")
print("=" * 60)

do {
    let fusionSurrogates = Python.import("fusion_surrogates")
    print("✅ Successfully imported fusion_surrogates")
    print("   Version:", fusionSurrogates.__version__)
} catch {
    print("❌ Failed to import fusion_surrogates:", error)
    exit(1)
}

// Test 2: Initialize QLKNN model
print("\n" + "=" * 60)
print("Test 2: Initialize QLKNN_7_11 model")
print("=" * 60)

do {
    let fusionSurrogates = Python.import("fusion_surrogates")
    let qlknn = fusionSurrogates.qlknn.QLKNN_7_11()
    print("✅ QLKNN_7_11 model initialized successfully")
} catch {
    print("❌ Failed to initialize QLKNN:", error)
    exit(1)
}

// Test 3: Make a prediction with Python types
print("\n" + "=" * 60)
print("Test 3: Prediction with scalar inputs")
print("=" * 60)

do {
    let fusionSurrogates = Python.import("fusion_surrogates")
    let np = Python.import("numpy")
    let qlknn = fusionSurrogates.qlknn.QLKNN_7_11()

    // Create test inputs (typical tokamak parameters)
    let inputs: [String: PythonObject] = [
        "R_L_Te": np.array([5.0]),      // Normalized electron temperature gradient
        "R_L_Ti": np.array([5.0]),      // Normalized ion temperature gradient
        "R_L_ne": np.array([1.0]),      // Normalized electron density gradient
        "R_L_ni": np.array([1.0]),      // Normalized ion density gradient
        "q": np.array([2.0]),           // Safety factor
        "s_hat": np.array([1.0]),       // Magnetic shear
        "r_R": np.array([0.3]),         // Inverse aspect ratio
        "Ti_Te": np.array([1.0]),       // Ion-electron temperature ratio
        "log_nu_star": np.array([-10.0]), // Logarithmic collisionality
        "ni_ne": np.array([1.0])        // Normalized density ratio
    ]

    let outputs = qlknn.predict(inputs)

    print("✅ Prediction successful")
    print("\nOutputs:")
    for key in ["chi_ion_itg", "chi_electron_tem", "chi_electron_etg", "particle_flux"] {
        if let value = outputs[Python.builtins.str(key)] {
            print("  \(key): \(value)")
        }
    }
} catch {
    print("❌ Prediction failed:", error)
    exit(1)
}

// Test 4: Prediction with multiple grid points
print("\n" + "=" * 60)
print("Test 4: Prediction with profile (5 grid points)")
print("=" * 60)

do {
    let fusionSurrogates = Python.import("fusion_surrogates")
    let np = Python.import("numpy")
    let qlknn = fusionSurrogates.qlknn.QLKNN_7_11()

    let nCells = 5

    // Create profile inputs
    let inputs: [String: PythonObject] = [
        "R_L_Te": np.array([5.0, 5.5, 6.0, 6.5, 7.0]),
        "R_L_Ti": np.array([5.0, 5.5, 6.0, 6.5, 7.0]),
        "R_L_ne": np.array([1.0, 1.2, 1.4, 1.6, 1.8]),
        "R_L_ni": np.array([1.0, 1.2, 1.4, 1.6, 1.8]),
        "q": np.array([1.5, 2.0, 2.5, 3.0, 3.5]),
        "s_hat": np.array([0.8, 1.0, 1.2, 1.4, 1.6]),
        "r_R": np.array([0.2, 0.25, 0.3, 0.35, 0.4]),
        "Ti_Te": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "log_nu_star": np.array([-12.0, -11.0, -10.0, -9.0, -8.0]),
        "ni_ne": np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    ]

    let outputs = qlknn.predict(inputs)

    print("✅ Profile prediction successful")
    print("\nOutput shapes:")
    for key in ["chi_ion_itg", "chi_electron_tem", "chi_electron_etg", "particle_flux"] {
        if let value = outputs[Python.builtins.str(key)] {
            print("  \(key): shape = \(value.shape)")
        }
    }

    print("\nchi_ion_itg values:")
    let chiIon = outputs["chi_ion_itg"]
    print("  \(chiIon)")

} catch {
    print("❌ Profile prediction failed:", error)
    exit(1)
}

// Test 5: Check available models
print("\n" + "=" * 60)
print("Test 5: Available QLKNN models")
print("=" * 60)

do {
    let fusionSurrogates = Python.import("fusion_surrogates")
    let qlknn_module = fusionSurrogates.qlknn

    print("Available QLKNN models:")
    let dir = Python.dir(qlknn_module)
    for item in dir {
        let itemStr = String(item)!
        if itemStr.hasPrefix("QLKNN_") {
            print("  - \(itemStr)")
        }
    }
} catch {
    print("❌ Failed to list models:", error)
}

print("\n" + "=" * 60)
print("All Python tests passed! ✅")
print("=" * 60)
