import Foundation
import MLX

// MARK: - QLKNN Model Interface

/// QLKNN neural network model for transport flux prediction
public struct QLKNN {

    /// Input parameter names expected by QLKNN model
    /// Order matters: must match model input layer
    public static let inputParameterNames: [String] = [
        "Ati",        // R/L_Ti - Normalized ion temperature gradient
        "Ate",        // R/L_Te - Normalized electron temperature gradient
        "Ane",        // R/L_ne - Normalized electron density gradient
        "Ani",        // R/L_ni - Normalized ion density gradient
        "q",          // Safety factor
        "smag",       // Magnetic shear (s_hat)
        "x",          // r/R - Inverse aspect ratio
        "Ti_Te",      // Ion-electron temperature ratio
        "LogNuStar",  // Logarithmic normalized collisionality
        "normni"      // Normalized ion density (ni/ne)
    ]

    /// Output parameter names returned by QLKNN model (ONNX order)
    public static let outputParameterNames: [String] = [
        "efeITG",     // Electron thermal flux (ITG mode) [GB units]
        "efiITG",     // Ion thermal flux (ITG mode)
        "pfeITG",     // Particle flux (ITG mode)
        "efeTEM",     // Electron thermal flux (TEM mode)
        "efiTEM",     // Ion thermal flux (TEM mode)
        "pfeTEM",     // Particle flux (TEM mode)
        "efeETG",     // Electron thermal flux (ETG mode)
        "gamma_max"   // Maximum growth rate
    ]

    /// Validate input dictionary has all required parameters
    ///
    /// - Parameter inputs: Dictionary to validate
    /// - Throws: FusionSurrogatesError if any required parameter is missing
    public static func validateInputs(_ inputs: [String: MLXArray]) throws {
        for paramName in inputParameterNames {
            guard inputs[paramName] != nil else {
                throw FusionSurrogatesError.predictionFailed(
                    "Missing required input parameter: \(paramName)"
                )
            }
        }
    }

    /// Validate input dictionary with Float values
    ///
    /// - Parameter inputs: Dictionary to validate
    /// - Throws: FusionSurrogatesError if any required parameter is missing
    public static func validateInputs(_ inputs: [String: Float]) throws {
        for paramName in inputParameterNames {
            guard inputs[paramName] != nil else {
                throw FusionSurrogatesError.predictionFailed(
                    "Missing required input parameter: \(paramName)"
                )
            }
        }
    }

    /// Validate that all input arrays have consistent shapes
    ///
    /// - Parameter inputs: Dictionary of input arrays
    /// - Throws: FusionSurrogatesError if shapes are inconsistent
    public static func validateShapes(_ inputs: [String: MLXArray]) throws {
        guard let firstArray = inputs.values.first else {
            throw FusionSurrogatesError.predictionFailed("No input arrays provided")
        }

        let expectedShape = firstArray.shape

        // Check all arrays have 1D shape
        if expectedShape.count != 1 {
            throw FusionSurrogatesError.predictionFailed(
                "Expected 1D arrays, got shape: \(expectedShape)"
            )
        }

        let nCells = expectedShape[0]

        // Check all arrays have the same shape
        for (key, array) in inputs {
            if array.shape != expectedShape {
                throw FusionSurrogatesError.predictionFailed(
                    "Shape mismatch for '\(key)': expected \(expectedShape), got \(array.shape)"
                )
            }

            // Check for NaN or Inf values
            eval(array)
            let values = array.asArray(Float.self)

            if values.contains(where: { $0.isNaN }) {
                throw FusionSurrogatesError.predictionFailed(
                    "NaN values detected in input '\(key)'"
                )
            }
            if values.contains(where: { $0.isInfinite }) {
                throw FusionSurrogatesError.predictionFailed(
                    "Infinite values detected in input '\(key)'"
                )
            }
        }

        // Validate reasonable grid size
        if nCells < 2 {
            throw FusionSurrogatesError.predictionFailed(
                "Grid size too small: \(nCells) (minimum 2 cells required)"
            )
        }

        if nCells > 10000 {
            throw FusionSurrogatesError.predictionFailed(
                "Grid size too large: \(nCells) (maximum 10000 cells)"
            )
        }
    }
}

// MARK: - Error Types

public enum FusionSurrogatesError: Error {
    case predictionFailed(String)
    case invalidInput(String)

    public var localizedDescription: String {
        switch self {
        case .predictionFailed(let message):
            return "Prediction failed: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        }
    }
}

// MARK: - Helper Extensions

extension MLXArray {
    /// Create array with repeated value
    public static func repeating(_ value: Float, count: Int) -> MLXArray {
        return broadcast(MLXArray(value), to: [count])
    }
}
