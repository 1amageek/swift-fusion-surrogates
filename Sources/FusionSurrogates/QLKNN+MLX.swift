import Foundation
import MLX
import PythonKit

// MARK: - QLKNN Extensions for MLX Integration

extension QLKNN {

    /// Predict transport fluxes using MLXArray inputs
    ///
    /// - Parameter inputs: Dictionary of input parameters as MLXArrays (1D arrays of length batch_size)
    /// - Returns: Dictionary of output parameters as MLXArrays
    /// - Throws: FusionSurrogatesError if prediction fails
    public func predict(_ inputs: [String: MLXArray]) throws -> [String: MLXArray] {
        // Validate inputs before prediction
        try QLKNN.validateInputs(inputs)

        // Validate shapes are consistent
        try QLKNN.validateShapes(inputs)

        // Convert MLXArrays to 2D numpy array (batch_size, 10)
        let pythonInputArray = MLXConversion.batchToPythonArray(inputs)

        // Call Python prediction (new API returns dict of JAX arrays)
        let pythonOutputs = model.predict(pythonInputArray)

        // Convert outputs back to MLXArrays
        return MLXConversion.batchFromPython(pythonOutputs)
    }

    /// Predict transport fluxes with scalar inputs (will be broadcast to grid size)
    ///
    /// - Parameters:
    ///   - inputs: Dictionary of scalar input parameters
    ///   - nCells: Number of cells in the grid
    /// - Returns: Dictionary of output arrays
    /// - Throws: FusionSurrogatesError if prediction fails
    public func predictScalar(
        _ inputs: [String: Double],
        nCells: Int
    ) throws -> [String: MLXArray] {
        // Convert scalars to MLXArrays
        var mlxInputs: [String: MLXArray] = [:]
        for (key, value) in inputs {
            mlxInputs[key] = MLXArray.repeating(value, count: nCells)
        }

        return try predict(mlxInputs)
    }

    /// Predict transport fluxes with profile-based inputs
    ///
    /// This method takes spatially-varying profiles as input and returns
    /// spatially-varying transport coefficients.
    ///
    /// - Parameter profiles: Dictionary of profile arrays
    /// - Returns: Dictionary of transport coefficients
    /// - Throws: FusionSurrogatesError if prediction fails
    public func predictProfiles(_ profiles: [String: MLXArray]) throws -> [String: MLXArray] {
        return try predict(profiles)
    }
}

// MARK: - QLKNN Input Builders

extension QLKNN {

    /// Input parameter names expected by QLKNN model (new API)
    /// Order matters: must match model.config.input_names
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

    /// Output parameter names returned by QLKNN model (new API)
    public static let outputParameterNames: [String] = [
        "efiITG",     // Ion thermal flux (ITG mode) [GB units]
        "efeITG",     // Electron thermal flux (ITG mode)
        "efeTEM",     // Electron thermal flux (TEM mode)
        "efeETG",     // Electron thermal flux (ETG mode)
        "efiTEM",     // Ion thermal flux (TEM mode)
        "pfeITG",     // Particle flux (ITG mode)
        "pfeTEM",     // Particle flux (TEM mode)
        "gamma_max"   // Maximum growth rate
    ]

    /// Validate input dictionary has all required parameters
    ///
    /// - Parameter inputs: Dictionary to validate
    /// - Throws: FusionSurrogatesError.missingInput if any required parameter is missing
    public static func validateInputs(_ inputs: [String: MLXArray]) throws {
        for paramName in inputParameterNames {
            guard inputs[paramName] != nil else {
                throw FusionSurrogatesError.predictionFailed(
                    "Missing required input parameter: \(paramName)"
                )
            }
        }
    }

    /// Validate input dictionary with Double values
    ///
    /// - Parameter inputs: Dictionary to validate
    /// - Throws: FusionSurrogatesError.missingInput if any required parameter is missing
    public static func validateInputs(_ inputs: [String: Double]) throws {
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
    /// - Throws: FusionSurrogatesError.predictionFailed if shapes are inconsistent
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
            let values = array.asArray(Double.self)
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

// MARK: - Helper Extensions

extension MLXArray {
    /// Create array with repeated value
    fileprivate static func repeating(_ value: Double, count: Int) -> MLXArray {
        return broadcast(MLXArray(value), to: [count])
    }
}
