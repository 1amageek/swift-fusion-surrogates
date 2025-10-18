import Foundation
import MLX

// MARK: - TORAX Integration Helpers

/// Helper functions for integrating FusionSurrogates with swift-TORAX
///
/// These utilities provide the glue between QLKNN predictions and TORAX's
/// transport model interface, handling input preprocessing and output formatting.
public enum TORAXIntegration {

    // MARK: - Input Preprocessing

    /// Compute normalized logarithmic gradient R/L_X = -R * d(ln X)/dr
    ///
    /// - Parameters:
    ///   - profile: Radial profile (temperature, density, etc.)
    ///   - radius: Radial coordinate [m]
    ///   - majorRadius: Major radius R [m]
    /// - Returns: Normalized logarithmic gradient
    public static func computeNormalizedGradient(
        profile: MLXArray,
        radius: MLXArray,
        majorRadius: Float
    ) -> MLXArray {
        // Compute derivative using finite differences
        let gradProfile = gradient(profile, radius)

        // Avoid division by zero
        let safeProfile = maximum(profile, MLXArray(1e-10))

        // R/L_X = -R * (dX/dr) / X
        return -MLXArray(majorRadius) * gradProfile / safeProfile
    }

    /// Compute safety factor q from poloidal flux
    ///
    /// - Parameters:
    ///   - poloidalFlux: Poloidal flux psi [Wb]
    ///   - radius: Radial coordinate [m]
    ///   - minorRadius: Minor radius a [m]
    ///   - majorRadius: Major radius R [m]
    ///   - toroidalField: Toroidal magnetic field B_T [T]
    /// - Returns: Safety factor q
    public static func computeSafetyFactor(
        poloidalFlux: MLXArray,
        radius: MLXArray,
        minorRadius: Float,
        majorRadius: Float,
        toroidalField: Float
    ) -> MLXArray {
        // Simplified cylindrical approximation: q ≈ r*B_T / (R*B_p)
        // B_p ≈ d(psi)/dr / (2*pi*r)

        let dPsiDr = gradient(poloidalFlux, radius)
        let safePsi = maximum(abs(dPsiDr), MLXArray(1e-10))

        let bPoloidal = safePsi / (2.0 * Float.pi * maximum(radius, MLXArray(1e-6)))
        let q = radius * MLXArray(toroidalField) / (MLXArray(majorRadius) * bPoloidal)

        return q
    }

    /// Compute magnetic shear s_hat = (r/q) * dq/dr
    ///
    /// - Parameters:
    ///   - safetyFactor: Safety factor q
    ///   - radius: Radial coordinate [m]
    /// - Returns: Magnetic shear s_hat
    public static func computeMagneticShear(
        safetyFactor: MLXArray,
        radius: MLXArray
    ) -> MLXArray {
        let dqDr = gradient(safetyFactor, radius)
        let safeQ = maximum(abs(safetyFactor), MLXArray(1e-10))
        return radius * dqDr / safeQ
    }

    /// Compute logarithmic collisionality log(nu_star)
    ///
    /// - Parameters:
    ///   - density: Electron density [m^-3]
    ///   - temperature: Electron temperature [eV]
    ///   - majorRadius: Major radius R [m]
    ///   - safetyFactor: Safety factor q
    /// - Returns: Logarithmic collisionality
    public static func computeCollisionality(
        density: MLXArray,
        temperature: MLXArray,
        majorRadius: Float,
        safetyFactor: MLXArray
    ) -> MLXArray {
        // Simplified collisionality formula
        // nu_star ≈ 6.921e-18 * q * R * n_e / T_e^2

        let nuStar = 6.921e-18 * safetyFactor * MLXArray(majorRadius) * density
            / (temperature * temperature)

        // Return logarithm
        return log(maximum(nuStar, MLXArray(1e-10)))
    }

    // MARK: - Output Processing

    /// Combine QLKNN flux predictions into total transport coefficients
    ///
    /// QLKNN outputs separate contributions from ITG, TEM, and ETG modes.
    /// This combines them into total chi_ion and chi_electron.
    ///
    /// - Parameter qlknnOutputs: Dictionary of QLKNN predictions
    /// - Returns: Dictionary with combined transport coefficients
    public static func combineFluxes(_ qlknnOutputs: [String: MLXArray]) -> [String: MLXArray] {
        var combined: [String: MLXArray] = [:]

        // Total ion heat diffusivity (primarily from ITG)
        combined["chi_ion"] = qlknnOutputs["chi_ion_itg"] ?? MLXArray.zeros([1])

        // Total electron heat diffusivity (TEM + ETG)
        let chiTEM = qlknnOutputs["chi_electron_tem"] ?? MLXArray.zeros([1])
        let chiETG = qlknnOutputs["chi_electron_etg"] ?? MLXArray.zeros([1])
        combined["chi_electron"] = chiTEM + chiETG

        // Particle diffusivity (can be derived from particle flux)
        combined["particle_diffusivity"] = qlknnOutputs["particle_flux"] ?? MLXArray.zeros([1])

        // Convection velocity (assume zero for now)
        let nCells = (qlknnOutputs.values.first?.shape[0]) ?? 1
        combined["convection_velocity"] = MLXArray.zeros([nCells])

        return combined
    }

    // MARK: - Finite Difference Utilities

    /// Compute gradient using finite differences (MLX-native implementation)
    ///
    /// - Parameters:
    ///   - f: Function values on grid
    ///   - x: Coordinate values (grid points)
    /// - Returns: df/dx at each grid point
    ///
    /// Uses centered differences for interior points, forward/backward for boundaries.
    /// This implementation uses MLX slicing operations for GPU acceleration.
    private static func gradient(_ f: MLXArray, _ x: MLXArray) -> MLXArray {
        let n = f.shape[0]

        if n < 2 {
            return MLXArray.zeros(f.shape)
        }

        if n == 2 {
            // Only two points: use simple difference
            let df = f[1] - f[0]
            let dx = x[1] - x[0]
            let grad = df / dx
            // Broadcast scalar to shape [2]
            return broadcast(grad, to: [2])
        }

        // For n >= 3, use proper finite differences
        // Forward difference at first boundary: (f[1] - f[0]) / (x[1] - x[0])
        let gradFirst = (f[1] - f[0]) / (x[1] - x[0])

        // Centered differences for interior points: (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
        // Slicing: f[1 ..< n] is f[1], f[2], ..., f[n-1]
        //          f[0 ..< (n-1)] is f[0], f[1], ..., f[n-2]
        let fNext = f[2 ..< n]           // f[2], f[3], ..., f[n-1]
        let fPrev = f[0 ..< (n - 2)]     // f[0], f[1], ..., f[n-3]
        let xNext = x[2 ..< n]           // x[2], x[3], ..., x[n-1]
        let xPrev = x[0 ..< (n - 2)]     // x[0], x[1], ..., x[n-3]

        let gradInterior = (fNext - fPrev) / (xNext - xPrev)

        // Backward difference at last boundary: (f[n-1] - f[n-2]) / (x[n-1] - x[n-2])
        let gradLast = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2])

        // Concatenate: [gradFirst, gradInterior..., gradLast]
        // Need to expand scalars to shape [1] for concatenation
        let gradFirstArray = gradFirst.reshaped([1])
        let gradLastArray = gradLast.reshaped([1])

        return concatenated([gradFirstArray, gradInterior, gradLastArray], axis: 0)
    }
}

// MARK: - QLKNN Input Builder from TORAX Types

extension QLKNN {

    /// Build QLKNN inputs from TORAX-style profiles and geometry
    ///
    /// This is a convenience method that takes common TORAX data structures
    /// and produces the normalized input parameters required by QLKNN.
    ///
    /// - Parameters:
    ///   - electronTemperature: Electron temperature profile [eV]
    ///   - ionTemperature: Ion temperature profile [eV]
    ///   - electronDensity: Electron density profile [m^-3]
    ///   - ionDensity: Ion density profile [m^-3]
    ///   - poloidalFlux: Poloidal flux profile [Wb]
    ///   - radius: Radial coordinate [m]
    ///   - majorRadius: Major radius R [m]
    ///   - minorRadius: Minor radius a [m]
    ///   - toroidalField: Toroidal magnetic field B_T [T]
    /// - Returns: Dictionary of QLKNN input parameters
    public static func buildInputs(
        electronTemperature: MLXArray,
        ionTemperature: MLXArray,
        electronDensity: MLXArray,
        ionDensity: MLXArray,
        poloidalFlux: MLXArray,
        radius: MLXArray,
        majorRadius: Float,
        minorRadius: Float,
        toroidalField: Float
    ) -> [String: MLXArray] {

        // Compute normalized gradients
        let rLnTe = TORAXIntegration.computeNormalizedGradient(
            profile: electronTemperature,
            radius: radius,
            majorRadius: majorRadius
        )

        let rLnTi = TORAXIntegration.computeNormalizedGradient(
            profile: ionTemperature,
            radius: radius,
            majorRadius: majorRadius
        )

        let rLnNe = TORAXIntegration.computeNormalizedGradient(
            profile: electronDensity,
            radius: radius,
            majorRadius: majorRadius
        )

        let rLnNi = TORAXIntegration.computeNormalizedGradient(
            profile: ionDensity,
            radius: radius,
            majorRadius: majorRadius
        )

        // Compute safety factor and magnetic shear
        let q = TORAXIntegration.computeSafetyFactor(
            poloidalFlux: poloidalFlux,
            radius: radius,
            minorRadius: minorRadius,
            majorRadius: majorRadius,
            toroidalField: toroidalField
        )

        let sHat = TORAXIntegration.computeMagneticShear(
            safetyFactor: q,
            radius: radius
        )

        // Other parameters
        let rR = radius / MLXArray(majorRadius)
        let tiTe = ionTemperature / electronTemperature
        let logNuStar = TORAXIntegration.computeCollisionality(
            density: electronDensity,
            temperature: electronTemperature,
            majorRadius: majorRadius,
            safetyFactor: q
        )
        let niNe = ionDensity / electronDensity

        return [
            "R_L_Te": rLnTe,
            "R_L_Ti": rLnTi,
            "R_L_ne": rLnNe,
            "R_L_ni": rLnNi,
            "q": q,
            "s_hat": sHat,
            "r_R": rR,
            "Ti_Te": tiTe,
            "log_nu_star": logNuStar,
            "ni_ne": niNe
        ]
    }
}
