import XCTest
@testable import KittenTTS

private struct SmokePhonemizer: KittenPhonemizerProtocol {
    func phonemize(_ text: String) -> String {
        "həˈloʊ."
    }
}

final class NativeEngineTests: XCTestCase {
    func testNativeEngineGeneratesAudioWithLocalAssets() async throws {
        guard let assetRoot = ProcessInfo.processInfo.environment["KITTENTTS_NATIVE_TEST_ASSETS"] else {
            throw XCTSkip("Set KITTENTTS_NATIVE_TEST_ASSETS to a native weights directory to run this smoke test.")
        }

        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let archURL = packageRoot
            .appendingPathComponent("Sources/KittenTTS/Resources/native_arch/kitten_fp32_15m_arch.json")
        let assetRootURL = URL(fileURLWithPath: assetRoot, isDirectory: true)
        let modelFiles = KittenTTSNativeModelFiles(
            archURL: archURL,
            weightsURL: assetRootURL.appendingPathComponent("kitten_fp32_15m.bin"),
            voiceDirectoryURL: assetRootURL.appendingPathComponent("voices_kitten_15m", isDirectory: true)
        )
        let config = KittenTTSConfig(
            defaultVoice: .bella,
            phonemizer: .custom(SmokePhonemizer()),
            inferenceEngine: .native,
            nativeConfig: KittenTTSNativeConfig(modelFiles: modelFiles)
        )

        let tts = try await KittenTTS(config)
        let result = try await tts.generate("Hello.", voice: .bella)

        XCTAssertFalse(result.samples.isEmpty)
        XCTAssertGreaterThan(result.duration, 0)
        XCTAssertEqual(result.voice, .bella)

        let slow = try await tts.generate("Hello.", voice: .bella, speed: 0.5)
        let fast = try await tts.generate("Hello.", voice: .bella, speed: 2.0)
        XCTAssertGreaterThan(slow.duration, fast.duration)
    }
}
