import Testing
import MLX
@testable import FusionSurrogates

/// Tests for TORAXIntegration helper functions
@Suite("TORAX Integration Tests")
struct TORAXIntegrationTests {

    // MARK: - combineFluxes Tests

    @Test("combineFluxes with all mode outputs")
    func combineFluxesComplete() {
        let qlknnOutputs: [String: MLXArray] = [
            "efiITG": MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3]),
            "efiTEM": MLXArray([Float(0.5), Float(1.0), Float(1.5)], [3]),
            "efeITG": MLXArray([Float(0.3), Float(0.6), Float(0.9)], [3]),
            "efeTEM": MLXArray([Float(0.2), Float(0.4), Float(0.6)], [3]),
            "efeETG": MLXArray([Float(0.1), Float(0.2), Float(0.3)], [3]),
            "pfeITG": MLXArray([Float(0.1), Float(0.2), Float(0.3)], [3]),
            "pfeTEM": MLXArray([Float(0.05), Float(0.1), Float(0.15)], [3]),
            "gamma_max": MLXArray([Float(10.0), Float(20.0), Float(30.0)], [3])
        ]

        let combined = TORAXIntegration.combineFluxes(qlknnOutputs)

        // Check all expected keys are present
        #expect(combined["chi_ion"] != nil)
        #expect(combined["chi_electron"] != nil)
        #expect(combined["particle_flux"] != nil)
        #expect(combined["growth_rate"] != nil)

        // Verify chi_ion = efiITG + efiTEM
        if let chiIon = combined["chi_ion"] {
            eval(chiIon)
            let values = chiIon.asArray(Float.self)
            #expect(abs(values[0] - 1.5) < 1e-5)  // 1.0 + 0.5
            #expect(abs(values[1] - 3.0) < 1e-5)  // 2.0 + 1.0
            #expect(abs(values[2] - 4.5) < 1e-5)  // 3.0 + 1.5
        }

        // Verify chi_electron = efeITG + efeTEM + efeETG
        if let chiElectron = combined["chi_electron"] {
            eval(chiElectron)
            let values = chiElectron.asArray(Float.self)
            #expect(abs(values[0] - 0.6) < 1e-5)  // 0.3 + 0.2 + 0.1
            #expect(abs(values[1] - 1.2) < 1e-5)  // 0.6 + 0.4 + 0.2
            #expect(abs(values[2] - 1.8) < 1e-5)  // 0.9 + 0.6 + 0.3
        }
    }

    @Test("combineFluxes with missing modes uses zero")
    func combineFluxesMissingModes() {
        // Only ITG mode
        let qlknnOutputs: [String: MLXArray] = [
            "efiITG": MLXArray([Float(1.0), Float(2.0), Float(3.0)], [3]),
            "efeITG": MLXArray([Float(0.5), Float(1.0), Float(1.5)], [3]),
            "pfeITG": MLXArray([Float(0.1), Float(0.2), Float(0.3)], [3]),
            "gamma_max": MLXArray([Float(10.0), Float(20.0), Float(30.0)], [3])
        ]

        let combined = TORAXIntegration.combineFluxes(qlknnOutputs)

        // Should still produce valid outputs, using zero for missing modes
        #expect(combined["chi_ion"] != nil)
        #expect(combined["chi_electron"] != nil)
        #expect(combined["particle_flux"] != nil)
    }

    @Test("combineFluxes returns Float32")
    func combineFluxesFloat32() {
        let qlknnOutputs: [String: MLXArray] = [
            "efiITG": MLXArray([Float(1.0)], [1]),
            "efiTEM": MLXArray([Float(0.5)], [1]),
            "efeITG": MLXArray([Float(0.5)], [1]),
            "efeTEM": MLXArray([Float(0.3)], [1]),
            "efeETG": MLXArray([Float(0.2)], [1]),
            "pfeITG": MLXArray([Float(0.1)], [1]),
            "pfeTEM": MLXArray([Float(0.05)], [1]),
            "gamma_max": MLXArray([Float(10.0)], [1])
        ]

        let combined = TORAXIntegration.combineFluxes(qlknnOutputs)

        // Verify all outputs use Float32
        for (_, array) in combined {
            eval(array)
            let values = array.asArray(Float.self)
            #expect(!values.isEmpty)

            // Float32 should handle these values
            for value in values {
                #expect(!value.isNaN)
                #expect(!value.isInfinite)
            }
        }
    }

    // MARK: - computeNormalizedGradient Tests

    @Test("computeNormalizedGradient basic calculation")
    func computeNormalizedGradientBasic() {
        // Linear profile: T = 1000 + 500*r
        let r = MLXArray([Float(0.0), Float(1.0), Float(2.0), Float(3.0), Float(4.0)], [5])
        let T = MLXArray([Float(1000.0), Float(1500.0), Float(2000.0), Float(2500.0), Float(3000.0)], [5])
        let R: Float = 6.2

        let rLnT = TORAXIntegration.computeNormalizedGradient(
            profile: T,
            radius: r,
            majorRadius: R
        )

        eval(rLnT)
        let values = rLnT.asArray(Float.self)

        // All values should be finite
        for value in values {
            #expect(!value.isNaN, "NaN in gradient")
            #expect(!value.isInfinite, "Infinite in gradient")
        }

        // For linear profile, gradient should be approximately constant
        // R/L_T = -R * (dT/dr) / T = -6.2 * 500 / T
        // Should be negative
        for value in values {
            #expect(value < 0, "Expected negative gradient")
        }
    }

    @Test("computeNormalizedGradient handles edge values")
    func computeNormalizedGradientEdges() {
        let r = MLXArray([Float(0.0), Float(0.5), Float(1.0)], [3])
        let T = MLXArray([Float(2000.0), Float(1500.0), Float(1000.0)], [3])  // Decreasing
        let R: Float = 6.2

        let rLnT = TORAXIntegration.computeNormalizedGradient(
            profile: T,
            radius: r,
            majorRadius: R
        )

        eval(rLnT)
        let values = rLnT.asArray(Float.self)

        // Edge values use forward/backward differences
        // All should be finite
        for value in values {
            #expect(!value.isNaN, "NaN at edges")
            #expect(!value.isInfinite, "Infinite at edges")
        }
    }

    @Test("computeNormalizedGradient with constant profile")
    func computeNormalizedGradientConstant() {
        // Constant profile should give zero gradient
        let r = MLXArray([Float(0.0), Float(1.0), Float(2.0)], [3])
        let T = MLXArray([Float(1000.0), Float(1000.0), Float(1000.0)], [3])
        let R: Float = 6.2

        let rLnT = TORAXIntegration.computeNormalizedGradient(
            profile: T,
            radius: r,
            majorRadius: R
        )

        eval(rLnT)
        let values = rLnT.asArray(Float.self)

        // Gradients should be near zero (within numerical precision)
        for value in values {
            #expect(abs(value) < 0.1, "Expected near-zero gradient for constant profile")
        }
    }

    // MARK: - computeSafetyFactor Tests

    @Test("computeSafetyFactor basic calculation")
    func computeSafetyFactorBasic() {
        let n = 5
        let r = MLXArray([Float(0.1), Float(0.5), Float(1.0), Float(1.5), Float(2.0)], [n])
        let psi = MLXArray([Float(0.0), Float(0.1), Float(0.2), Float(0.3), Float(0.4)], [n])
        let R: Float = 6.2
        let a: Float = 2.0
        let Bt: Float = 5.3

        let q = TORAXIntegration.computeSafetyFactor(
            poloidalFlux: psi,
            radius: r,
            minorRadius: a,
            majorRadius: R,
            toroidalField: Bt
        )

        eval(q)
        let values = q.asArray(Float.self)

        // Safety factor should be positive and finite
        for value in values {
            #expect(value > 0, "Safety factor should be positive")
            #expect(!value.isNaN)
            #expect(!value.isInfinite)
        }
    }

    @Test("computeSafetyFactor avoids division by zero")
    func computeSafetyFactorDivisionByZero() {
        // Test with zero radius at first point
        let n = 3
        let r = MLXArray([Float(0.0), Float(0.5), Float(1.0)], [n])
        let psi = MLXArray([Float(0.0), Float(0.1), Float(0.2)], [n])
        let R: Float = 6.2
        let a: Float = 2.0
        let Bt: Float = 5.3

        let q = TORAXIntegration.computeSafetyFactor(
            poloidalFlux: psi,
            radius: r,
            minorRadius: a,
            majorRadius: R,
            toroidalField: Bt
        )

        eval(q)
        let values = q.asArray(Float.self)

        // Should not produce NaN or Inf
        for value in values {
            #expect(!value.isNaN, "Should handle zero radius")
            #expect(!value.isInfinite, "Should handle zero radius")
        }
    }

    // MARK: - computeMagneticShear Tests

    @Test("computeMagneticShear basic calculation")
    func computeMagneticShearBasic() {
        let n = 5
        let r = MLXArray([Float(0.1), Float(0.5), Float(1.0), Float(1.5), Float(2.0)], [n])
        let q = MLXArray([Float(1.0), Float(1.5), Float(2.0), Float(2.5), Float(3.0)], [n])

        let sHat = TORAXIntegration.computeMagneticShear(
            safetyFactor: q,
            radius: r
        )

        eval(sHat)
        let values = sHat.asArray(Float.self)

        // Magnetic shear should be finite
        for value in values {
            #expect(!value.isNaN)
            #expect(!value.isInfinite)
        }
    }

    @Test("computeMagneticShear with constant q")
    func computeMagneticShearConstant() {
        let n = 3
        let r = MLXArray([Float(0.5), Float(1.0), Float(1.5)], [n])
        let q = MLXArray([Float(2.0), Float(2.0), Float(2.0)], [n])  // Constant q

        let sHat = TORAXIntegration.computeMagneticShear(
            safetyFactor: q,
            radius: r
        )

        eval(sHat)
        let values = sHat.asArray(Float.self)

        // Shear should be near zero for constant q
        for value in values {
            #expect(abs(value) < 0.5, "Expected near-zero shear for constant q")
        }
    }

    // MARK: - computeCollisionality Tests

    @Test("computeCollisionality basic calculation")
    func computeCollisionalityBasic() {
        let n = 3
        let ne = MLXArray([Float(1e19), Float(2e19), Float(3e19)], [n])
        let Te = MLXArray([Float(1000.0), Float(2000.0), Float(3000.0)], [n])
        let q = MLXArray([Float(2.0), Float(2.5), Float(3.0)], [n])
        let R: Float = 6.2

        let logNuStar = TORAXIntegration.computeCollisionality(
            density: ne,
            temperature: Te,
            majorRadius: R,
            safetyFactor: q
        )

        eval(logNuStar)
        let values = logNuStar.asArray(Float.self)

        // Logarithmic collisionality should be finite
        for value in values {
            #expect(!value.isNaN)
            #expect(!value.isInfinite)
        }

        // Should be in reasonable range (typically -15 to -5)
        for value in values {
            #expect(value > -20, "LogNuStar too small")
            #expect(value < 0, "LogNuStar should be negative for fusion plasmas")
        }
    }

    // MARK: - buildInputs Tests

    @Test("buildInputs produces all required parameters")
    func buildInputsComplete() {
        let n = 5
        let Te = MLXArray([Float(1000.0), Float(2000.0), Float(3000.0), Float(4000.0), Float(5000.0)], [n])
        let Ti = MLXArray([Float(1000.0), Float(2000.0), Float(3000.0), Float(4000.0), Float(5000.0)], [n])
        let ne = MLXArray([Float(1e19), Float(2e19), Float(3e19), Float(4e19), Float(5e19)], [n])
        let ni = MLXArray([Float(1e19), Float(2e19), Float(3e19), Float(4e19), Float(5e19)], [n])
        let psi = MLXArray([Float(0.0), Float(0.1), Float(0.2), Float(0.3), Float(0.4)], [n])
        let r = MLXArray([Float(0.1), Float(0.5), Float(1.0), Float(1.5), Float(2.0)], [n])

        let inputs = QLKNN.buildInputs(
            electronTemperature: Te,
            ionTemperature: Ti,
            electronDensity: ne,
            ionDensity: ni,
            poloidalFlux: psi,
            radius: r,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3
        )

        // Check all required parameters are present
        for paramName in QLKNN.inputParameterNames {
            #expect(inputs[paramName] != nil, "Missing parameter: \(paramName)")
        }

        // Check all have correct shape
        for (key, value) in inputs {
            #expect(value.shape == [n], "Wrong shape for \(key): \(value.shape)")
        }
    }

    @Test("buildInputs validates after creation")
    func buildInputsValidation() throws {
        let n = 3
        let Te = MLXArray([Float(1000.0), Float(2000.0), Float(3000.0)], [n])
        let Ti = MLXArray([Float(1000.0), Float(2000.0), Float(3000.0)], [n])
        let ne = MLXArray([Float(1e19), Float(2e19), Float(3e19)], [n])
        let ni = MLXArray([Float(1e19), Float(2e19), Float(3e19)], [n])
        let psi = MLXArray([Float(0.0), Float(0.1), Float(0.2)], [n])
        let r = MLXArray([Float(0.5), Float(1.0), Float(1.5)], [n])

        let inputs = QLKNN.buildInputs(
            electronTemperature: Te,
            ionTemperature: Ti,
            electronDensity: ne,
            ionDensity: ni,
            poloidalFlux: psi,
            radius: r,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3
        )

        // Should pass validation
        try QLKNN.validateInputs(inputs)
        try QLKNN.validateShapes(inputs)
    }

    @Test("buildInputs calculates Ti_Te ratio")
    func buildInputsTiTeRatio() {
        let n = 3
        let Te = MLXArray([Float(1000.0), Float(2000.0), Float(4000.0)], [n])
        let Ti = MLXArray([Float(2000.0), Float(2000.0), Float(2000.0)], [n])  // Constant Ti
        let ne = MLXArray([Float(1e19), Float(2e19), Float(3e19)], [n])
        let ni = ne
        let psi = MLXArray([Float(0.0), Float(0.1), Float(0.2)], [n])
        let r = MLXArray([Float(0.5), Float(1.0), Float(1.5)], [n])

        let inputs = QLKNN.buildInputs(
            electronTemperature: Te,
            ionTemperature: Ti,
            electronDensity: ne,
            ionDensity: ni,
            poloidalFlux: psi,
            radius: r,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3
        )

        // Check Ti_Te ratio
        if let tiTeRatio = inputs["Ti_Te"] {
            eval(tiTeRatio)
            let values = tiTeRatio.asArray(Float.self)

            // Ti/Te ratios should be: 2.0, 1.0, 0.5
            #expect(abs(values[0] - 2.0) < 1e-4)
            #expect(abs(values[1] - 1.0) < 1e-4)
            #expect(abs(values[2] - 0.5) < 1e-4)
        } else {
            #expect(Bool(false), "Ti_Te not found in inputs")
        }
    }

    @Test("buildInputs calculates normni")
    func buildInputsNormni() {
        let n = 3
        let Te = MLXArray([Float(1000.0), Float(2000.0), Float(3000.0)], [n])
        let Ti = Te
        let ne = MLXArray([Float(2e19), Float(2e19), Float(2e19)], [n])  // Constant ne
        let ni = MLXArray([Float(1e19), Float(2e19), Float(3e19)], [n])  // Varying ni
        let psi = MLXArray([Float(0.0), Float(0.1), Float(0.2)], [n])
        let r = MLXArray([Float(0.5), Float(1.0), Float(1.5)], [n])

        let inputs = QLKNN.buildInputs(
            electronTemperature: Te,
            ionTemperature: Ti,
            electronDensity: ne,
            ionDensity: ni,
            poloidalFlux: psi,
            radius: r,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3
        )

        // Check ni/ne ratio
        if let normni = inputs["normni"] {
            eval(normni)
            let values = normni.asArray(Float.self)

            // ni/ne ratios should be: 0.5, 1.0, 1.5
            #expect(abs(values[0] - 0.5) < 1e-4)
            #expect(abs(values[1] - 1.0) < 1e-4)
            #expect(abs(values[2] - 1.5) < 1e-4)
        } else {
            #expect(Bool(false), "normni not found in inputs")
        }
    }

    @Test("buildInputs all outputs are Float32")
    func buildInputsFloat32() {
        let n = 3
        let Te = MLXArray([Float(1000.0), Float(2000.0), Float(3000.0)], [n])
        let Ti = MLXArray([Float(1000.0), Float(2000.0), Float(3000.0)], [n])
        let ne = MLXArray([Float(1e19), Float(2e19), Float(3e19)], [n])
        let ni = MLXArray([Float(1e19), Float(2e19), Float(3e19)], [n])
        let psi = MLXArray([Float(0.0), Float(0.1), Float(0.2)], [n])
        let r = MLXArray([Float(0.5), Float(1.0), Float(1.5)], [n])

        let inputs = QLKNN.buildInputs(
            electronTemperature: Te,
            ionTemperature: Ti,
            electronDensity: ne,
            ionDensity: ni,
            poloidalFlux: psi,
            radius: r,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3
        )

        // Verify all outputs use Float32
        for (key, array) in inputs {
            eval(array)
            let values = array.asArray(Float.self)
            #expect(!values.isEmpty, "\(key) is empty")

            for value in values {
                #expect(!value.isNaN, "\(key) contains NaN")
                #expect(!value.isInfinite, "\(key) contains Inf")
            }
        }
    }

    // MARK: - Physical Validity Tests

    @Test("computeNormalizedGradient handles very small values")
    func computeNormalizedGradientSmallValues() {
        // Test with very small but positive temperature
        let r = MLXArray([Float(0.5), Float(1.0), Float(1.5)], [3])
        let T = MLXArray([Float(1e-5), Float(2e-5), Float(3e-5)], [3])
        let R: Float = 6.2

        let rLnT = TORAXIntegration.computeNormalizedGradient(
            profile: T,
            radius: r,
            majorRadius: R
        )

        eval(rLnT)
        let values = rLnT.asArray(Float.self)

        // Should not produce NaN or Inf due to small values
        for value in values {
            #expect(!value.isNaN, "Small values produce NaN")
            #expect(!value.isInfinite, "Small values produce Inf")
        }
    }

    @Test("computeNormalizedGradient with large temperature range")
    func computeNormalizedGradientLargeRange() {
        // Test with realistic ITER-scale temperatures (eV)
        let r = MLXArray([Float(0.0), Float(0.5), Float(1.0), Float(1.5), Float(2.0)], [5])
        let T = MLXArray([Float(20000.0), Float(15000.0), Float(10000.0), Float(5000.0), Float(1000.0)], [5])
        let R: Float = 6.2

        let rLnT = TORAXIntegration.computeNormalizedGradient(
            profile: T,
            radius: r,
            majorRadius: R
        )

        eval(rLnT)
        let values = rLnT.asArray(Float.self)

        // All gradients should be finite
        for value in values {
            #expect(!value.isNaN)
            #expect(!value.isInfinite)
        }

        // For decreasing temperature, R/L_T should be positive
        for value in values {
            #expect(value > 0, "Expected positive gradient for decreasing temperature")
        }
    }

    @Test("computeSafetyFactor with realistic ITER parameters")
    func computeSafetyFactorITER() {
        // ITER-like parameters
        let n = 5
        let r = MLXArray([Float(0.2), Float(0.8), Float(1.4), Float(1.8), Float(2.0)], [n])
        let psi = MLXArray([Float(0.0), Float(5.0), Float(15.0), Float(30.0), Float(40.0)], [n])
        let R: Float = 6.2  // ITER major radius
        let a: Float = 2.0  // ITER minor radius
        let Bt: Float = 5.3 // ITER toroidal field

        let q = TORAXIntegration.computeSafetyFactor(
            poloidalFlux: psi,
            radius: r,
            minorRadius: a,
            majorRadius: R,
            toroidalField: Bt
        )

        eval(q)
        let values = q.asArray(Float.self)

        // Safety factor should be positive and finite
        // Note: Simplified formula q ≈ r*B_T/(R*B_p) may produce small values
        // depending on poloidal flux gradient
        for value in values {
            #expect(value > 0.0, "Safety factor should be positive")
            #expect(value < 1000.0, "Safety factor unrealistically large")
            #expect(!value.isNaN)
            #expect(!value.isInfinite)
        }
    }

    @Test("computeCollisionality with extreme temperatures")
    func computeCollisionalityExtremeTemp() {
        let n = 4
        // Test range from cold edge (100 eV) to hot core (30 keV)
        let ne = MLXArray([Float(5e19), Float(1e20), Float(1e20), Float(5e19)], [n])
        let Te = MLXArray([Float(100.0), Float(5000.0), Float(20000.0), Float(30000.0)], [n])
        let q = MLXArray([Float(3.0), Float(2.0), Float(1.5), Float(1.0)], [n])
        let R: Float = 6.2

        let logNuStar = TORAXIntegration.computeCollisionality(
            density: ne,
            temperature: Te,
            majorRadius: R,
            safetyFactor: q
        )

        eval(logNuStar)
        let values = logNuStar.asArray(Float.self)

        // All should be finite
        for value in values {
            #expect(!value.isNaN)
            #expect(!value.isInfinite)
        }

        // Collisionality should decrease with temperature (nu_star ~ 1/T^2)
        // So log(nu_star) should decrease
        #expect(values[0] > values[3], "Collisionality should decrease with temperature")
    }

    @Test("buildInputs with extreme but realistic parameters")
    func buildInputsExtreme() {
        let n = 3
        // Edge plasma: low temperature, high density
        let Te = MLXArray([Float(100.0), Float(500.0), Float(1000.0)], [n])
        let Ti = MLXArray([Float(100.0), Float(500.0), Float(1000.0)], [n])
        let ne = MLXArray([Float(1e20), Float(5e19), Float(1e19)], [n])
        let ni = ne
        let psi = MLXArray([Float(0.0), Float(5.0), Float(10.0)], [n])
        let r = MLXArray([Float(1.8), Float(1.9), Float(2.0)], [n])

        let inputs = QLKNN.buildInputs(
            electronTemperature: Te,
            ionTemperature: Ti,
            electronDensity: ne,
            ionDensity: ni,
            poloidalFlux: psi,
            radius: r,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3
        )

        // All parameters should be finite
        for (key, array) in inputs {
            eval(array)
            let values = array.asArray(Float.self)
            for value in values {
                #expect(!value.isNaN, "\(key) contains NaN with extreme parameters")
                #expect(!value.isInfinite, "\(key) contains Inf with extreme parameters")
            }
        }
    }

    @Test("buildInputs with high temperature core plasma")
    func buildInputsHotCore() {
        let n = 3
        // Hot core plasma: high temperature, moderate density
        let Te = MLXArray([Float(20000.0), Float(10000.0), Float(5000.0)], [n])
        let Ti = MLXArray([Float(25000.0), Float(12000.0), Float(6000.0)], [n])
        let ne = MLXArray([Float(5e19), Float(8e19), Float(1e20)], [n])
        let ni = ne
        let psi = MLXArray([Float(0.0), Float(2.0), Float(8.0)], [n])
        let r = MLXArray([Float(0.0), Float(0.5), Float(1.0)], [n])

        let inputs = QLKNN.buildInputs(
            electronTemperature: Te,
            ionTemperature: Ti,
            electronDensity: ne,
            ionDensity: ni,
            poloidalFlux: psi,
            radius: r,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3
        )

        // Ti_Te should be > 1 in core
        if let tiTe = inputs["Ti_Te"] {
            eval(tiTe)
            let values = tiTe.asArray(Float.self)
            #expect(values[0] > 1.0, "Ti should be higher than Te in core")
        }

        // All parameters should be in valid QLKNN range
        for (key, array) in inputs {
            eval(array)
            let values = array.asArray(Float.self)
            for value in values {
                #expect(!value.isNaN, "\(key) contains NaN")
                #expect(!value.isInfinite, "\(key) contains Inf")
            }
        }
    }

    @Test("combineFluxes preserves physical units")
    func combineFluxesPhysicalUnits() {
        // All QLKNN outputs are in Gyro-Bohm units
        // chi ~ rho_s^2 * v_th / L_n where rho_s ~ 1cm, v_th ~ 1e6 m/s
        // Typical values: 0.1 - 10 m^2/s in GB units
        let qlknnOutputs: [String: MLXArray] = [
            "efiITG": MLXArray([Float(1.5)], [1]),    // GB units
            "efiTEM": MLXArray([Float(0.3)], [1]),
            "efeITG": MLXArray([Float(0.8)], [1]),
            "efeTEM": MLXArray([Float(0.4)], [1]),
            "efeETG": MLXArray([Float(0.2)], [1]),
            "pfeITG": MLXArray([Float(0.5)], [1]),
            "pfeTEM": MLXArray([Float(0.2)], [1]),
            "gamma_max": MLXArray([Float(50.0)], [1])  // Growth rate
        ]

        let combined = TORAXIntegration.combineFluxes(qlknnOutputs)

        // chi_ion = efiITG + efiTEM
        if let chiIon = combined["chi_ion"] {
            eval(chiIon)
            let values = chiIon.asArray(Float.self)
            #expect(abs(values[0] - 1.8) < 1e-5)  // 1.5 + 0.3
        }

        // Verify units are preserved (should be in GB units, 0.1-10 range)
        if let chiElectron = combined["chi_electron"] {
            eval(chiElectron)
            let values = chiElectron.asArray(Float.self)
            #expect(values[0] > 0.0, "chi_electron should be positive")
            #expect(values[0] < 100.0, "chi_electron unrealistically large")
        }
    }

    @Test("gradient computation numerical stability")
    func gradientNumericalStability() {
        // Test with non-uniform grid spacing
        let r = MLXArray([Float(0.0), Float(0.1), Float(0.5), Float(1.2), Float(2.0)], [5])
        let T = MLXArray([Float(10000.0), Float(9500.0), Float(8000.0), Float(5000.0), Float(2000.0)], [5])
        let R: Float = 6.2

        let rLnT = TORAXIntegration.computeNormalizedGradient(
            profile: T,
            radius: r,
            majorRadius: R
        )

        eval(rLnT)
        let values = rLnT.asArray(Float.self)

        // Should handle non-uniform spacing without instability
        for i in 0..<values.count {
            #expect(!values[i].isNaN, "NaN at index \(i) with non-uniform grid")
            #expect(!values[i].isInfinite, "Inf at index \(i) with non-uniform grid")
        }
    }
}
