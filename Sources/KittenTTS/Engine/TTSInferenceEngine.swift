import Foundation

/// Internal result returned by inference engines.
struct TTSOutput {
    let samples: [Float]
    /// Predicted frame count per input token when available.
    let durations: [Int64]
    /// The IPA phoneme string produced by the phonemizer.
    let phonemes: String
}

protocol TTSInferenceEngine: AnyObject, Sendable {
    func generate(text: String, voice: KittenVoice, speed: Float) throws -> TTSOutput
}
