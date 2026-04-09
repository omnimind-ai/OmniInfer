// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "OmniInferServer",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(name: "OmniInferServer", targets: ["OmniInferServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
    ],
    targets: [
        // Pre-built native library: llama.cpp + OmniInfer C bridge.
        // Build with: scripts/platforms/ios/llama.cpp-ios/build.sh
        .binaryTarget(
            name: "llama",
            path: "Frameworks/llama.xcframework"
        ),

        // Swift public API + Hummingbird HTTP server.
        .target(
            name: "OmniInferServer",
            dependencies: [
                "llama",
                .product(name: "Hummingbird", package: "hummingbird"),
            ],
            path: "Sources/OmniInferServer"
        ),
    ]
)
