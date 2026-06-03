import CKittenNativeEngine
import Foundation

final class NativeTTSEngine: TTSInferenceEngine, @unchecked Sendable {
    private let handle: KTNativeModelHandle?
    private let voiceDirectoryURL: URL
    private let config: KittenTTSConfig
    private let phonemizer: any KittenPhonemizerProtocol
    private var styleCache: [KittenVoice: [Float]] = [:]

    init(
        archURL: URL,
        weightsURL: URL,
        voiceDirectoryURL: URL,
        config: KittenTTSConfig,
        phonemizer: any KittenPhonemizerProtocol
    ) throws {
        self.voiceDirectoryURL = voiceDirectoryURL
        self.config = config
        self.phonemizer = phonemizer

        var error: UnsafeMutablePointer<CChar>?
        let created = archURL.path.withCString { archPath in
            weightsURL.path.withCString { weightsPath in
                kt_native_model_create(archPath, weightsPath, &error)
            }
        }
        guard let created else {
            throw KittenTTSError.inferenceFailed(Self.consumeError(error) ?? "Native model creation failed.")
        }
        self.handle = created
    }

    deinit {
        kt_native_model_destroy(handle)
    }

    func generate(text: String, voice: KittenVoice, speed: Float) throws -> TTSOutput {
        let normalised = TextPreprocessor.process(text)
        let phonemes = phonemizer.phonemize(normalised)
        let tokens = TextCleaner.encodeNative(phonemes)
        let chunks = splitIntoChunks(tokens)
        let speedPrior = config.applySpeedPriors ? config.model.speedPrior(for: voice) : 1.0
        let effectiveSpeed = speed * speedPrior
        let style = try styleForVoice(voice)

        var allSamples: [Float] = []
        for chunk in chunks {
            let samples = try runChunk(tokens: chunk, style: style)
            allSamples.append(contentsOf: applySpeed(samples, speed: effectiveSpeed))
        }

        guard !allSamples.isEmpty else { throw KittenTTSError.emptyOutput }
        return TTSOutput(samples: allSamples, durations: [], phonemes: phonemes)
    }

    private func runChunk(tokens: [Float], style: [Float]) throws -> [Float] {
        guard let handle else {
            throw KittenTTSError.inferenceFailed("Native engine is not initialized.")
        }

        var output = KTNativeFloatArray()
        var error: UnsafeMutablePointer<CChar>?
        let ok = tokens.withUnsafeBufferPointer { tokenBuffer in
            style.withUnsafeBufferPointer { styleBuffer in
                kt_native_model_synthesize(
                    handle,
                    tokenBuffer.baseAddress,
                    CInt(tokens.count),
                    styleBuffer.baseAddress,
                    CInt(style.count),
                    &output,
                    &error
                )
            }
        }

        guard ok != 0, let data = output.data, output.length > 0 else {
            kt_native_float_array_free(&output)
            throw KittenTTSError.inferenceFailed(Self.consumeError(error) ?? "Native synthesis failed.")
        }

        let samples = Array(UnsafeBufferPointer(start: data, count: Int(output.length)))
        kt_native_float_array_free(&output)
        return samples
    }

    private func styleForVoice(_ voice: KittenVoice) throws -> [Float] {
        if let cached = styleCache[voice] {
            return cached
        }

        let url = voiceDirectoryURL.appendingPathComponent("\(voice.rawValue).bin")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw KittenTTSError.voicesFileNotFound(url)
        }

        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let count = data.count / MemoryLayout<Float>.stride
        guard count >= 256 else {
            throw KittenTTSError.invalidModelData("Invalid native voice file: \(url.lastPathComponent)")
        }

        let floats = [Float](unsafeUninitializedCapacity: count) { buffer, initializedCount in
            data.withUnsafeBytes { raw in
                _ = raw.copyBytes(to: buffer)
            }
            initializedCount = count
        }
        let style = Array(floats.prefix(256))
        styleCache[voice] = style
        return style
    }

    private func applySpeed(_ samples: [Float], speed: Float) -> [Float] {
        guard speed > 0 else { return samples }
        guard !samples.isEmpty, abs(speed - 1.0) >= 1e-6 else { return samples }

        let outputCount = max(1, Int((Float(samples.count) / speed).rounded()))
        if outputCount == samples.count { return samples }
        if outputCount == 1 { return [samples[0]] }

        let maxIndex = Float(samples.count - 1)
        return (0 ..< outputCount).map { i in
            let position = Float(i) * maxIndex / Float(outputCount - 1)
            let lower = Int(position)
            let upper = min(lower + 1, samples.count - 1)
            let fraction = position - Float(lower)
            return samples[lower] + (samples[upper] - samples[lower]) * fraction
        }
    }

    private func splitIntoChunks(_ tokens: [Float]) -> [[Float]] {
        guard tokens.count > config.maxTokensPerChunk else {
            return [tokens]
        }

        let body = Array(tokens.dropFirst().dropLast())
        let maxBody = max(1, config.maxTokensPerChunk - 2)
        var chunks: [[Float]] = []
        var index = 0

        while index < body.count {
            let slice = Array(body[index ..< min(index + maxBody, body.count)])
            chunks.append([Float(TextCleaner.startTokenID)] + slice + [Float(TextCleaner.padTokenID)])
            index += maxBody
        }
        return chunks
    }

    private static func consumeError(_ pointer: UnsafeMutablePointer<CChar>?) -> String? {
        guard let pointer else { return nil }
        defer { kt_native_string_free(pointer) }
        return String(cString: pointer)
    }
}
