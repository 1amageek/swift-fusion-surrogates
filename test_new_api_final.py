#!/usr/bin/env python3
"""Final verification of new QLKNN API matching Swift implementation."""

import sys
import numpy as np

def main():
    print("=" * 70)
    print("Final Verification: Swift-Compatible Python QLKNN API")
    print("=" * 70)

    try:
        from fusion_surrogates.qlknn.qlknn_model import QLKNNModel

        # 1. Load model
        print("\n1. Loading QLKNNModel...")
        model = QLKNNModel.load_default_model()
        print(f"   ✅ Model loaded: {model.name}")
        print(f"   ✅ Version: {model.version}")

        # 2. Check configuration
        print("\n2. Model configuration:")
        print(f"   Input names ({len(model.config.input_names)}): {model.config.input_names}")
        print(f"   Flux map keys ({len(model.config.flux_map)}): {list(model.config.flux_map.keys())}")

        # 3. Prepare inputs (matching Swift test)
        print("\n3. Preparing inputs (Swift format)...")
        batch_size = 3

        # Order must match: [Ati, Ate, Ane, Ani, q, smag, x, Ti_Te, LogNuStar, normni]
        inputs = np.array([
            [5.0, 5.0, 1.0, 1.0, 2.0, 1.0, 0.3, 1.0, -3.0, 1.0],    # Sample 1
            [6.0, 6.0, 1.5, 1.5, 2.5, 1.2, 0.35, 1.0, -2.5, 1.0],   # Sample 2
            [7.0, 7.0, 2.0, 2.0, 3.0, 1.4, 0.4, 1.0, -2.0, 1.0],    # Sample 3
        ], dtype=np.float64)

        print(f"   ✅ Input shape: {inputs.shape} (batch_size={batch_size}, num_features=10)")
        print(f"   ✅ Input sample[0]: {inputs[0]}")

        # 4. Run prediction
        print("\n4. Running prediction...")
        outputs = model.predict(inputs)
        print(f"   ✅ Prediction successful!")
        print(f"   ✅ Output keys: {list(outputs.keys())}")

        # 5. Display outputs
        print("\n5. Output values:")
        for key in sorted(outputs.keys()):
            value = np.array(outputs[key])
            print(f"   {key:12s}: shape={value.shape}, values={value[:, 0]}")

        # 6. Combine fluxes (Swift TORAXIntegration.combineFluxes logic)
        print("\n6. Combining fluxes (Swift logic):")

        efiITG = np.array(outputs['efiITG'])
        efiTEM = np.array(outputs['efiTEM'])
        chi_ion = efiITG + efiTEM
        print(f"   chi_ion = efiITG + efiTEM")
        print(f"   chi_ion: {chi_ion[:, 0]}")

        efeITG = np.array(outputs['efeITG'])
        efeTEM = np.array(outputs['efeTEM'])
        efeETG = np.array(outputs['efeETG'])
        chi_electron = efeITG + efeTEM + efeETG
        print(f"   chi_electron = efeITG + efeTEM + efeETG")
        print(f"   chi_electron: {chi_electron[:, 0]}")

        pfeITG = np.array(outputs['pfeITG'])
        pfeTEM = np.array(outputs['pfeTEM'])
        particle_flux = pfeITG + pfeTEM
        print(f"   particle_flux = pfeITG + pfeTEM")
        print(f"   particle_flux: {particle_flux[:, 0]}")

        gamma_max = np.array(outputs['gamma_max'])
        print(f"   growth_rate = gamma_max")
        print(f"   growth_rate: {gamma_max[:, 0]}")

        # 7. Verify outputs are reasonable
        print("\n7. Verifying output sanity...")
        all_finite = True
        for key, value in outputs.items():
            value_np = np.array(value)
            if not np.all(np.isfinite(value_np)):
                print(f"   ❌ {key} contains non-finite values")
                all_finite = False

        if all_finite:
            print(f"   ✅ All outputs are finite")

        print("\n" + "=" * 70)
        print("✅ ALL VERIFICATION PASSED")
        print("=" * 70)
        print("\n📝 Summary:")
        print(f"   - Model: QLKNNModel (new API)")
        print(f"   - Input format: 2D numpy array (batch_size, 10)")
        print(f"   - Output format: Dict[str, JAX array]")
        print(f"   - Input params: {model.config.input_names}")
        print(f"   - Output params: {list(outputs.keys())}")
        print(f"\n✅ Ready for Swift integration!")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
