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

    /// Batch convert multiple MLXArrays to Python numpy arrays
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
