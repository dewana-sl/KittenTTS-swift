// swift-tools-version: 5.9
// KittenSDK – on-device text-to-speech for iOS & macOS, powered by KittenTTS + ONNX Runtime.


import PackageDescription

let package = Package(
    name: "KittenSDK",
    platforms: [
        .iOS(.v16),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "KittenTTS",
            targets: ["KittenTTS"]
        ),
        .executable(
            name: "NativeVsONNXBenchmark",
            targets: ["NativeVsONNXBenchmark"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/microsoft/onnxruntime-swift-package-manager",
            from: "1.20.0"
        ),
    ],
    targets: [
        // Thin system-library shim that exposes <zlib.h> to Swift.
        .systemLibrary(
            name: "Czlib",
            path: "Sources/Czlib"
        ),

        // C++ phonemizer engine with C bridge for Swift interop.
        // Reads rule/dictionary data files to produce IPA output
        .target(
            name: "CEPhonemizer",
            path: "Sources/CEPhonemizer",
            sources: ["phonemizer.cpp", "swift_bridge.cpp"],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("."),
                .define("NDEBUG", .when(configuration: .release)),
                .unsafeFlags(["-std=c++17"]),
            ]
        ),

        .target(
            name: "CKittenNativeEngine",
            path: "Sources/CKittenNativeEngine",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("src"),
                .define("NDEBUG", .when(configuration: .release)),
                .unsafeFlags(["-std=c++17"]),
                .unsafeFlags([
                    "-O3",
                    "-ffast-math",
                    "-fno-finite-math-only",
                    "-funroll-loops",
                    "-ftree-vectorize",
                ], .when(platforms: [.iOS, .macOS])),
                .unsafeFlags(["-march=native"], .when(platforms: [.macOS])),
            ]
        ),

        .target(
            name: "KittenTTS",
            dependencies: [
                "Czlib",
                "CEPhonemizer",
                "CKittenNativeEngine",
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "Sources/KittenTTS",
            resources: [
                .process("Resources"),
            ]
        ),

        .testTarget(
            name: "KittenTTSTests",
            dependencies: ["KittenTTS"],
            path: "Tests/KittenTTSTests"
        ),

        .executableTarget(
            name: "NativeVsONNXBenchmark",
            dependencies: ["KittenTTS"],
            path: "Benchmarks/NativeVsONNXBenchmark"
        ),
    ]
)
