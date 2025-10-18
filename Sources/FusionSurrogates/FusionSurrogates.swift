import Foundation
import PythonKit

/// A Swift wrapper for the fusion_surrogates Python library
public class FusionSurrogates {

    private let pythonModule: PythonObject

    /// Initialize the FusionSurrogates wrapper
    /// - Parameter pythonPath: Optional custom Python path. If nil, uses system Python
    public init(pythonPath: String? = nil) throws {
        if let path = pythonPath {
            PythonLibrary.useLibrary(at: path)
        }

        // Import the fusion_surrogates module
        self.pythonModule = Python.import("fusion_surrogates")
    }

    /// Access to the raw Python module for advanced usage
    public var module: PythonObject {
        return pythonModule
    }
}

/// QLKNN model wrapper for turbulent transport surrogate modeling
public class QLKNN {

    internal let model: PythonObject

    /// Initialize the QLKNN model
    /// - Parameter modelVersion: The QLKNN model version (default: "7_11")
    public init(modelVersion: String = "7_11") throws {
        let fusionSurrogates = Python.import("fusion_surrogates")
        let qlknn = fusionSurrogates.qlknn

        // Initialize the model based on version
        if modelVersion == "7_11" {
            self.model = qlknn.QLKNN_7_11()
        } else {
            throw FusionSurrogatesError.unsupportedModelVersion(modelVersion)
        }
    }

    /// Predict transport fluxes with PythonObject inputs (low-level)
    /// - Parameters:
    ///   - inputs: Dictionary of input parameters as PythonObjects
    /// - Returns: Dictionary of predicted outputs as PythonObject
    public func predictPython(_ inputs: [String: PythonObject]) -> PythonObject {
        return model.predict(inputs)
    }

    /// Predict transport fluxes with Double values (returns PythonObject)
    /// - Parameters:
    ///   - inputs: Dictionary of input parameters with Double values
    /// - Returns: Dictionary of predicted outputs as PythonObject
    public func predictPython(_ inputs: [String: Double]) -> PythonObject {
        return model.predict(inputs)
    }

    /// Access to the raw Python model
    public var rawModel: PythonObject {
        return model
    }
}

/// Errors that can occur when using FusionSurrogates
public enum FusionSurrogatesError: Error {
    case pythonImportFailed(String)
    case unsupportedModelVersion(String)
    case predictionFailed(String)

    public var localizedDescription: String {
        switch self {
        case .pythonImportFailed(let module):
            return "Failed to import Python module: \(module)"
        case .unsupportedModelVersion(let version):
            return "Unsupported model version: \(version)"
        case .predictionFailed(let reason):
            return "Prediction failed: \(reason)"
        }
    }
}
