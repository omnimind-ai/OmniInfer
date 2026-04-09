// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "OmniInferServer",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "OmniInferServer", targets: ["OmniInferServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "2.29.0"),
    ],
    targets: [
        // Pre-built native library: llama.cpp + OmniInfer C bridge.
        // Build with: scripts/platforms/ios/llama.cpp-ios/build.sh
        .binaryTarget(
            name: "llama",
            path: "Frameworks/llama.xcframework"
        ),

        // Swift public API + Hummingbird HTTP server + MLX engine.
        .target(
            name: "OmniInferServer",
            dependencies: [
                "llama",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/OmniInferServer"
        ),
    ]
)
