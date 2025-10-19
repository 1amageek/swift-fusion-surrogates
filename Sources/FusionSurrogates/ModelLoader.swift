import Foundation
import MLX

/// Model weight loader supporting SafeTensors format
public struct ModelLoader {

    /// Load SafeTensors file (MLX native format)
    public static func loadSafeTensors(path: String) throws -> [String: MLXArray] {
        guard FileManager.default.fileExists(atPath: path) else {
            throw QLKNNError.modelNotFound("SafeTensors file not found: \(path)")
        }

        let url = URL(fileURLWithPath: path)

        // MLX provides loadArrays for SafeTensors
        do {
            let arrays = try MLX.loadArrays(url: url)
            return arrays
        } catch {
            throw QLKNNError.invalidWeights("Failed to load SafeTensors: \(error)")
        }
    }

    /// Load model weights (auto-detects format)
    public static func load(path: String) throws -> [String: MLXArray] {
        let fileExtension = (path as NSString).pathExtension.lowercased()

        switch fileExtension {
        case "safetensors":
            return try loadSafeTensors(path: path)
        default:
            throw QLKNNError.invalidWeights("Unsupported format: \(fileExtension). Use .safetensors")
        }
    }
}
