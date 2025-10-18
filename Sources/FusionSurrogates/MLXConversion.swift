import Foundation
import MLX
import PythonKit

// MARK: - MLXArray ⇔ PythonObject Conversion

/// Utilities for converting between MLXArray and Python numpy arrays
public enum MLXConversion {

    /// Convert MLXArray to Python numpy array
    ///
    /// - Parameter mlxArray: MLXArray to convert
    /// - Returns: Python numpy array (PythonObject)
    public static func toPython(_ mlxArray: MLXArray) -> PythonObject {
        let np = Python.import("numpy")

        // Get shape and data
        let shape = mlxArray.shape
        let data = mlxArray.asArray(Float.self)

        // Create numpy array from Swift array
        let pythonArray = np.array(data)

        // Reshape to original shape
        return pythonArray.reshape(shape)
    }

    /// Convert Python numpy array to MLXArray
    ///
    /// - Parameter pythonArray: Python numpy array (PythonObject)
    /// - Returns: MLXArray
    public static func fromPython(_ pythonArray: PythonObject) -> MLXArray {
        // Get shape
        let shape = Array(pythonArray.shape)!.map { Int($0)! }

        // Flatten and convert to Swift array
        let flattened = pythonArray.flatten()
        let data = Array<Float>(flattened)!

        // Create MLXArray with proper shape
        return MLXArray(data, shape)
    }

    /// Convert Python numpy array to MLXArray (Double precision)
    ///
    /// - Parameter pythonArray: Python numpy array (PythonObject)
    /// - Returns: MLXArray with Float64 dtype
    public static func fromPythonDouble(_ pythonArray: PythonObject) -> MLXArray {
        let shape = Array(pythonArray.shape)!.map { Int($0)! }
        let flattened = pythonArray.flatten()
        let data = Array<Double>(flattened)!
        return MLXArray(data, shape)
    }

    /// Convert dictionary of MLXArrays to 2D numpy array for QLKNNModel
    ///
    /// New QLKNN API requires 2D array (batch_size, num_features) instead of dict
    /// Features must be in order: [Ati, Ate, Ane, Ani, q, smag, x, Ti_Te, LogNuStar, normni]
    ///
    /// - Parameter arrays: Dictionary of MLXArrays (each with shape [batch_size])
    /// - Returns: 2D numpy array of shape (batch_size, 10)
    public static func batchToPythonArray(_ arrays: [String: MLXArray]) -> PythonObject {
        let np = Python.import("numpy")

        // Feature order as defined by QLKNN.inputParameterNames
        let featureNames = [
            "Ati", "Ate", "Ane", "Ani", "q", "smag", "x", "Ti_Te", "LogNuStar", "normni"
        ]

        // Get batch size from first array
        guard let firstArray = arrays.values.first else {
            // Return empty 2D array
            return np.array([[Float]]())
        }
        let batchSize = firstArray.shape[0]

        // Stack features in correct order
        var features: [[Float]] = []
        for fname in featureNames {
            guard let array = arrays[fname] else {
                // Missing feature - this should have been caught by validation
                continue
            }
            eval(array)
            let values = array.asArray(Float.self)
            features.append(values)
        }

        // Transpose: features is [num_features][batch_size], need [batch_size][num_features]
        var transposed: [[Float]] = Array(repeating: Array(repeating: 0.0, count: featureNames.count), count: batchSize)
        for (featureIdx, featureValues) in features.enumerated() {
            for (batchIdx, value) in featureValues.enumerated() {
                transposed[batchIdx][featureIdx] = value
            }
        }

        // Convert to numpy array
        return np.array(transposed)
    }

    /// Batch convert multiple MLXArrays to Python numpy arrays (legacy dict format)
    ///
    /// - Parameter arrays: Dictionary of MLXArrays
    /// - Returns: Dictionary of Python numpy arrays
    public static func batchToPython(_ arrays: [String: MLXArray]) -> [String: PythonObject] {
        var result: [String: PythonObject] = [:]
        for (key, array) in arrays {
            result[key] = toPython(array)
        }
        return result
    }

    /// Batch convert multiple Python numpy arrays to MLXArrays
    ///
    /// - Parameter arrays: Dictionary of Python numpy arrays
    /// - Returns: Dictionary of MLXArrays
    public static func batchFromPython(_ arrays: PythonObject) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        // Python dict iteration
        for key in arrays.keys() {
            if let keyString = String(key) {
                result[keyString] = fromPython(arrays[key])
            }
        }

        return result
    }
}

// MARK: - MLXArray Extensions for Python Interop

extension MLXArray {
    /// Convert to Python numpy array
    public var pythonArray: PythonObject {
        MLXConversion.toPython(self)
    }

    /// Create MLXArray from Python numpy array
    ///
    /// - Parameter pythonArray: Python numpy array
    public static func from(pythonArray: PythonObject) -> MLXArray {
        return MLXConversion.fromPython(pythonArray)
    }
}
