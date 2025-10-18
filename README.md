# FusionSurrogates

A Swift wrapper for the [fusion_surrogates](https://github.com/google-deepmind/fusion_surrogates) Python library, providing surrogate models for tokamak fusion simulations.

## Overview

This package allows you to use Google DeepMind's fusion_surrogates library from Swift, leveraging PythonKit for seamless Python-Swift interoperability.

## Features

- Swift wrapper for fusion_surrogates Python library
- QLKNN_7_11 model support for turbulent transport surrogate modeling
- Type-safe Swift API with error handling
- Direct access to underlying Python objects for advanced usage

## Requirements

- Swift 6.2 or later
- macOS 13.0+ / iOS 16.0+
- Python 3.x with fusion_surrogates installed
- PythonKit

## Installation

### 1. Install fusion_surrogates Python library

```bash
# Create a Python virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install fusion_surrogates
pip install fusion_surrogates
```

### 2. Add to your Swift Package

Add the following dependency to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/YOUR_USERNAME/FusionSurrogates.git", branch: "main")
]
```

### 3. Initialize the submodule

```bash
git submodule update --init --recursive
```

## Usage

### Basic Example

```swift
import FusionSurrogates

do {
    // Initialize the QLKNN model
    let qlknn = try QLKNN(modelVersion: "7_11")

    // Prepare input parameters
    let inputs: [String: PythonConvertible] = [
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

### Advanced Usage

For direct access to the Python module:

```swift
import FusionSurrogates

do {
    let fusion = try FusionSurrogates()
    let pythonModule = fusion.module

    // Use the Python module directly
    // ...
} catch {
    print("Error:", error)
}
```

## Project Structure

```
FusionSurrogates/
├── Package.swift
├── README.md
├── Sources/
│   └── FusionSurrogates/
│       └── FusionSurrogates.swift
├── Tests/
│   └── FusionSurrogatesTests/
│       └── FusionSurrogatesTests.swift
└── fusion_surrogates/          # Git submodule
    └── (Python library files)
```

## Input Parameters

The QLKNN_7_11 model accepts the following inputs:

- `R_L_Te`: Normalized logarithmic electron temperature gradient
- `R_L_Ti`: Normalized logarithmic ion temperature gradient
- `R_L_ne`: Normalized logarithmic electron density gradient
- `R_L_ni`: Normalized logarithmic ion density gradient
- `q`: Safety factor
- `s_hat`: Magnetic shear
- `r_R`: Local inverse aspect ratio (r/R_maj)
- `Ti_Te`: Ion-electron temperature ratio
- `log_nu_star`: Logarithmic ion-electron normalized collisionality
- `ni_ne`: Normalized density (n_i/n_e)

## Output

The model outputs ion and electron heat and particle fluxes for each transport mode:
- ITG (Ion Temperature Gradient)
- ETG (Electron Temperature Gradient)
- TEM (Trapped Electron Modes)

Plus the maximum growth rate on ion gyroradius scales.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project follows the same licensing as fusion_surrogates:
- Code: Apache License 2.0
- Models and data: Creative Commons Attribution 4.0 International License (CC-BY)

## References

- [fusion_surrogates GitHub](https://github.com/google-deepmind/fusion_surrogates)
- [PythonKit](https://github.com/pvieito/PythonKit)
- [QuaLiKiz](https://gitlab.com/qualikiz-group/QuaLiKiz)
- [Van de Plassche et al. PoP 2020](https://doi.org/10.1063/1.5134126)

## Disclaimer

This is not an official Google product.
