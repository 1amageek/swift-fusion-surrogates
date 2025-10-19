// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "FusionSurrogates",
    platforms: [
        .macOS(.v15),
        .iOS(.v16)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "FusionSurrogates",
            targets: ["FusionSurrogates"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.1")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "FusionSurrogates",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ],
            resources: [
                .copy("Resources/qlknn_7_11_weights.safetensors"),
                .copy("Resources/qlknn_7_11_metadata.json")
            ]
        ),
        .testTarget(
            name: "FusionSurrogatesTests",
            dependencies: [
                "FusionSurrogates"
            ]
        ),
    ]
)
