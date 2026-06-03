import Foundation
import KittenTTS

private struct FixedPhonemizer: KittenPhonemizerProtocol {
    let ipa: String

    func phonemize(_ text: String) -> String {
        ipa
    }
}

private struct BenchmarkResult: Encodable {
    let platform: String
    let engine: String
    let model: String
    let voice: String
    let text: String
    let speed: Float
    let effectiveSpeed: Float
    let callSpeed: Float
    let runs: Int
    let durationSeconds: Double
    let generationTimeSecondsMean: Double
    let generationTimeSecondsMin: Double
    let generationTimeSecondsStd: Double
    let rtfMean: Double
    let sampleCount: Int
}

@main
struct NativeVsONNXBenchmark {
    static func main() async throws {
        let text = ProcessInfo.processInfo.environment["KITTENTTS_BENCH_TEXT"] ?? "The quick brown fox jumps over the lazy dog."
        let ipa = ProcessInfo.processInfo.environment["KITTENTTS_BENCH_IPA"] ?? "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ."
        let runs = Int(ProcessInfo.processInfo.environment["KITTENTTS_BENCH_RUNS"] ?? "3") ?? 3
        let warmup = Int(ProcessInfo.processInfo.environment["KITTENTTS_BENCH_WARMUP"] ?? "1") ?? 1
        let speed = Float(ProcessInfo.processInfo.environment["KITTENTTS_BENCH_SPEED"] ?? "1.0") ?? 1.0
        let callSpeed = speed
        let cacheDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["KITTENTTS_BENCH_CACHE"] ?? "/tmp/kittentts-swift-bench", isDirectory: true)
        let outputDir = ProcessInfo.processInfo.environment["KITTENTTS_BENCH_OUTPUT_DIR"].map {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
        let nativeAssetRoot = ProcessInfo.processInfo.environment["KITTENTTS_NATIVE_TEST_ASSETS"] ?? "/Volumes/DewansSSD/dev/project/cpp-convnet-pvt/weights"
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        let onnxConfig = KittenTTSConfig(
            model: .nano,
            defaultVoice: .bella,
            applySpeedPriors: false,
            phonemizer: .custom(FixedPhonemizer(ipa: ipa)),
            storageDirectory: cacheDir,
            inferenceEngine: .onnx
        )
        let onnx = try await KittenTTS(onnxConfig)
        let onnxResult = try await benchmark(
            tts: onnx,
            platform: "swift",
            engine: "onnx",
            text: text,
            effectiveSpeed: speed,
            callSpeed: callSpeed,
            outputDir: outputDir,
            runs: runs,
            warmup: warmup
        )

        let nativeRootURL = URL(fileURLWithPath: nativeAssetRoot, isDirectory: true)
        let nativeFiles = KittenTTSNativeModelFiles(
            archURL: packageRoot.appendingPathComponent("Sources/KittenTTS/Resources/native_arch/kitten_fp32_15m_arch.json"),
            weightsURL: nativeRootURL.appendingPathComponent("kitten_fp32_15m.bin"),
            voiceDirectoryURL: nativeRootURL.appendingPathComponent("voices_kitten_15m", isDirectory: true)
        )
        let nativeConfig = KittenTTSConfig(
            model: .nano,
            defaultVoice: .bella,
            applySpeedPriors: false,
            phonemizer: .custom(FixedPhonemizer(ipa: ipa)),
            storageDirectory: cacheDir,
            inferenceEngine: .native,
            nativeConfig: KittenTTSNativeConfig(modelFiles: nativeFiles)
        )
        let native = try await KittenTTS(nativeConfig)
        let nativeResult = try await benchmark(
            tts: native,
            platform: "swift",
            engine: "native",
            text: text,
            effectiveSpeed: speed,
            callSpeed: callSpeed,
            outputDir: outputDir,
            runs: runs,
            warmup: warmup
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode([onnxResult, nativeResult])
        print(String(data: data, encoding: .utf8)!)
    }

    private static func benchmark(
        tts: KittenTTS,
        platform: String,
        engine: String,
        text: String,
        effectiveSpeed: Float,
        callSpeed: Float,
        outputDir: URL?,
        runs: Int,
        warmup: Int
    ) async throws -> BenchmarkResult {
        for _ in 0 ..< warmup {
            _ = try await tts.generate(text, voice: .bella, speed: callSpeed)
        }

        var timings: [Double] = []
        var lastResult: KittenTTSResult?
        for _ in 0 ..< runs {
            let started = DispatchTime.now().uptimeNanoseconds
            let result = try await tts.generate(text, voice: .bella, speed: callSpeed)
            let ended = DispatchTime.now().uptimeNanoseconds
            timings.append(Double(ended - started) / 1_000_000_000.0)
            lastResult = result
        }

        let result = lastResult!
        if let outputDir {
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
            try result.writeWAV(to: outputDir.appendingPathComponent("swift-\(engine).wav"))
        }

        let mean = timings.reduce(0, +) / Double(timings.count)
        let minTime = timings.min() ?? mean
        let variance = timings.map { pow($0 - mean, 2) }.reduce(0, +) / Double(timings.count)
        let std = sqrt(variance)

        return BenchmarkResult(
            platform: platform,
            engine: engine,
            model: "nano/fp32_15m",
            voice: KittenVoice.bella.rawValue,
            text: text,
            speed: effectiveSpeed,
            effectiveSpeed: result.effectiveSpeed,
            callSpeed: callSpeed,
            runs: runs,
            durationSeconds: result.duration,
            generationTimeSecondsMean: mean,
            generationTimeSecondsMin: minTime,
            generationTimeSecondsStd: std,
            rtfMean: mean / result.duration,
            sampleCount: result.samples.count
        )
    }
}
