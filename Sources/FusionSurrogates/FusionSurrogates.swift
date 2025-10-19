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

    /// Initialize the QLKNN model using the new QLKNNModel API
    /// - Parameter modelName: The QLKNN model name (default: "qlknn_7_11_v1")
    public init(modelName: String = "qlknn_7_11_v1") throws {
        let fusionSurrogates = Python.import("fusion_surrogates")
        let qlknnModule = fusionSurrogates.qlknn.qlknn_model
        let QLKNNModel = qlknnModule.QLKNNModel

        // Load model using QLKNNModel API
        if modelName == "qlknn_7_11_v1" || modelName == "default" {
            self.model = QLKNNModel.load_default_model()
        } else {
            self.model = QLKNNModel.load_model_from_name(modelName)
        }
    }

    /// Predict transport fluxes with numpy array input (low-level)
    /// - Parameters:
    ///   - inputs: 2D numpy array of shape (batch_size, 10) with features in order:
    ///             [Ati, Ate, Ane, Ani, q, smag, x, Ti_Te, LogNuStar, normni]
    /// - Returns: Dictionary of predicted flux outputs (JAX arrays)
    public func predictPython(_ inputs: PythonObject) -> PythonObject {
        return model.predict(inputs)
    }

    /// Get model configuration
    public var config: PythonObject {
        return model.config
    }

    /// Get model metadata
    public var metadata: PythonObject {
        return model._metadata
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
