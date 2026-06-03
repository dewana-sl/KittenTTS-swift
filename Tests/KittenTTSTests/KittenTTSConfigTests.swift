import XCTest
@testable import KittenTTS

final class KittenTTSConfigTests: XCTestCase {

    func testDefaultValues() {
        let config = KittenTTSConfig()
        XCTAssertEqual(config.model,        .nano)
        XCTAssertEqual(config.defaultVoice, .bella)
        XCTAssertEqual(config.speed,        1.0, accuracy: 0.001)
        XCTAssertTrue(config.applySpeedPriors)
        XCTAssertNil(config.storageDirectory)
        XCTAssertNil(config.modelFiles)
        XCTAssertEqual(config.inferenceEngine, .onnx)
        XCTAssertEqual(config.nativeConfig, KittenTTSNativeConfig())
        XCTAssertEqual(config.ortNumThreads,     4)
        XCTAssertEqual(config.maxTokensPerChunk, 400)
    }

    func testSpeedClamping() {
        XCTAssertEqual(KittenTTSConfig(speed: 0.1).speed,  0.5, accuracy: 0.001)
        XCTAssertEqual(KittenTTSConfig(speed: 5.0).speed,  2.0, accuracy: 0.001)
        XCTAssertEqual(KittenTTSConfig(speed: 1.3).speed,  1.3, accuracy: 0.001)
    }

    func testOrtNumThreadsMinimum() {
        XCTAssertEqual(KittenTTSConfig(ortNumThreads: 0).ortNumThreads, 1)
        XCTAssertEqual(KittenTTSConfig(ortNumThreads: -5).ortNumThreads, 1)
    }

    func testMaxTokensPerChunkMinimum() {
        XCTAssertEqual(KittenTTSConfig(maxTokensPerChunk: 10).maxTokensPerChunk, 50)
        XCTAssertEqual(KittenTTSConfig(maxTokensPerChunk: 500).maxTokensPerChunk, 500)
    }

    func testOutputSampleRate() {
        XCTAssertEqual(KittenTTSConfig.outputSampleRate, 24_000)
    }

    func testResolvedStorageDirectoryContainsModelName() {
        let config = KittenTTSConfig()
        let dir = config.resolvedStorageDirectory
        XCTAssertTrue(dir.path.contains(KittenModel.nano.rawValue),
                      "Expected model name in storage path: \(dir.path)")
    }

    func testCustomStorageDirectory() {
        let custom = URL(fileURLWithPath: "/tmp/test-models")
        let config = KittenTTSConfig(storageDirectory: custom)
        XCTAssertTrue(config.resolvedStorageDirectory.path.hasPrefix("/tmp/test-models"))
    }

    func testModelFilesAreRetained() {
        let files = KittenTTSModelFiles(
            onnxURL: URL(fileURLWithPath: "/tmp/model.onnx"),
            voicesURL: URL(fileURLWithPath: "/tmp/voices.npz")
        )
        let config = KittenTTSConfig(model: .nanoInt8, modelFiles: files)

        XCTAssertEqual(config.model, .nanoInt8)
        XCTAssertEqual(config.modelFiles, files)
    }

    func testNativeConfigCanBeSelectedWithoutChangingOnnxDefaults() {
        let nativeConfig = KittenTTSNativeConfig(variant: .int8_15m)
        let config = KittenTTSConfig(inferenceEngine: .native, nativeConfig: nativeConfig)

        XCTAssertEqual(config.inferenceEngine, .native)
        XCTAssertEqual(config.nativeConfig.variant, .int8_15m)
        XCTAssertNil(KittenTTSConfig().nativeConfig.variant)
    }

    func testNativeVariantMappingMatchesPublicModels() {
        XCTAssertEqual(KittenModel.nano.nativeVariant, .fp32_15m)
        XCTAssertEqual(KittenModel.nanoInt8.nativeVariant, .int8_15m)
        XCTAssertEqual(KittenModel.micro.nativeVariant, .fp32_40m)
        XCTAssertEqual(KittenModel.mini.nativeVariant, .fp32_80m)
    }
}
