# Usage Examples

## 1. Basic QLKNN Model Usage

```swift
import FusionSurrogates

do {
    // Initialize the QLKNN model
    let qlknn = try QLKNN(modelVersion: "7_11")

    // Prepare input parameters
    let inputs: [String: Double] = [
        "R_L_Te": 5.0,    // Normalized logarithmic electron temperature gradient
        "R_L_Ti": 5.0,    // Normalized logarithmic ion temperature gradient
        "R_L_ne": 2.0,    // Normalized logarithmic electron density gradient
        "R_L_ni": 2.0,    // Normalized logarithmic ion density gradient
        "q": 2.0,         // Safety factor
        "s_hat": 1.0,     // Magnetic shear
        "r_R": 0.3,       // Local inverse aspect ratio
        "Ti_Te": 1.0,     // Ion-electron temperature ratio
        "log_nu_star": -2.0,  // Logarithmic normalized collisionality
        "ni_ne": 1.0      // Normalized density
    ]

    // Make prediction
    let outputs = try qlknn.predict(inputs)

    print("Prediction outputs:", outputs)
} catch {
    print("Error:", error)
}
```

## 2. Using with PythonObject directly

```swift
import FusionSurrogates
import PythonKit

do {
    let qlknn = try QLKNN(modelVersion: "7_11")

    // Create inputs using PythonObject
    let inputs: [String: PythonObject] = [
        "R_L_Te": PythonObject(5.0),
        "R_L_Ti": PythonObject(5.0),
        "R_L_ne": PythonObject(2.0),
        "R_L_ni": PythonObject(2.0),
        "q": PythonObject(2.0),
        "s_hat": PythonObject(1.0),
        "r_R": PythonObject(0.3),
        "Ti_Te": PythonObject(1.0),
        "log_nu_star": PythonObject(-2.0),
        "ni_ne": PythonObject(1.0)
    ]

    let outputs = try qlknn.predict(inputs)
    print(outputs)
} catch {
    print("Error:", error)
}
```

## 3. Advanced Usage with Direct Python Access

```swift
import FusionSurrogates
import PythonKit

do {
    // Access the Python module directly
    let fusion = try FusionSurrogates()
    let module = fusion.module

    // Access qlknn submodule
    let qlknnModule = module.qlknn

    // Create model instance
    let model = qlknnModule.QLKNN_7_11()

    // Use numpy for array operations
    let np = Python.import("numpy")
    let inputArray = np.array([5.0, 5.0, 2.0, 2.0, 2.0, 1.0, 0.3, 1.0, -2.0, 1.0])

    // Make predictions
    let result = model.predict(inputArray)
    print(result)
} catch {
    print("Error:", error)
}
```

## 4. Batch Predictions

```swift
import FusionSurrogates
import PythonKit

do {
    let qlknn = try QLKNN(modelVersion: "7_11")
    let np = Python.import("numpy")

    // Create batch inputs
    let batchSize = 10
    let inputData: [[Double]] = (0..<batchSize).map { i in
        [
            5.0 + Double(i) * 0.1,  // R_L_Te
            5.0,                    // R_L_Ti
            2.0,                    // R_L_ne
            2.0,                    // R_L_ni
            2.0,                    // q
            1.0,                    // s_hat
            0.3,                    // r_R
            1.0,                    // Ti_Te
            -2.0,                   // log_nu_star
            1.0                     // ni_ne
        ]
    }

    let inputArray = np.array(inputData)
    let outputs = qlknn.rawModel.predict(inputArray)

    print("Batch predictions:", outputs)
} catch {
    print("Error:", error)
}
```

## 5. Error Handling

```swift
import FusionSurrogates

do {
    // Try to initialize with unsupported version
    let qlknn = try QLKNN(modelVersion: "unsupported")
} catch FusionSurrogatesError.unsupportedModelVersion(let version) {
    print("Model version '\(version)' is not supported")
} catch {
    print("Unexpected error:", error)
}
```

## 6. Custom Python Path

```swift
import FusionSurrogates

do {
    // Use a specific Python installation
    let fusion = try FusionSurrogates(pythonPath: "/usr/local/bin/python3")

    // Continue with your code
    let qlknn = try QLKNN(modelVersion: "7_11")
    // ...
} catch {
    print("Error:", error)
}
```

## Environment Setup

### Setting PYTHONPATH

Before running your Swift code, make sure the fusion_surrogates module is in your Python path:

```bash
# If you installed fusion_surrogates in a virtual environment
export PYTHONPATH=/path/to/your/venv/lib/python3.x/site-packages:$PYTHONPATH

# Or if you're using the submodule directly
export PYTHONPATH=/path/to/swift-fusion-surrogates/fusion_surrogates:$PYTHONPATH
```

### macOS Example

```bash
# Activate your Python virtual environment
source .venv/bin/activate

# Install fusion_surrogates
pip install fusion_surrogates

# Run your Swift code
swift run
```

### iOS Considerations

For iOS applications, you'll need to:
1. Bundle Python runtime with your app
2. Include the fusion_surrogates library
3. Set up proper code signing
4. Consider using frameworks like Kivy-iOS or BeeWare

Note: Running Python on iOS has significant limitations and may not work out of the box.
