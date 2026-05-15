import Foundation

/// Available KittenTTS model variants.
public enum KittenModel: String, CaseIterable, Sendable, Codable, Hashable, ExpressibleByStringLiteral {

    /// The fp32 nano model (~56 MB, 15M parameters).
    ///
    /// Smallest and fastest. Runs on any Apple Silicon chip and A-series device.
    case nano = "nano"

    /// The int8-quantised nano model (~25 MB, 15M parameters).
    ///
    /// Same quality as ``nano`` at roughly half the size.
    /// Some users have reported minor quality differences vs. the fp32 variant.
    case nanoInt8 = "nano-int8"

    /// The micro model (~41 MB, 40M parameters).
    ///
    /// Higher quality than nano with moderate resource use.
    case micro = "micro"

    /// The mini model (~80 MB, 80M parameters).
    ///
    /// Highest quality available. Best for demanding use cases where memory allows.
    case mini = "mini"

    public init(stringLiteral value: String) {
        guard let model = Self(id: value) else {
            preconditionFailure("Unknown KittenTTS model: \(value)")
        }
        self = model
    }

    public init?(id: String) {
        switch id {
        case "nano", "kitten-tts-nano-0.8": self = .nano
        case "nano-int8", "kitten-tts-nano-0.8-int8": self = .nanoInt8
        case "micro", "kitten-tts-micro-0.8": self = .micro
        case "mini", "kitten-tts-mini-0.8": self = .mini
        default: return nil
        }
    }

    // MARK: - Hugging Face Metadata

    /// The Hugging Face repository ID (e.g. `"KittenML/kitten-tts-nano-0.8"`).
    public var huggingFaceRepo: String {
        "KittenML/\(repositoryID)"
    }

    /// Base URL for direct file downloads from Hugging Face.
    public var huggingFaceBaseURL: URL {
        URL(string: "https://huggingface.co/\(huggingFaceRepo)/resolve/main")!
    }

    // MARK: - File Names

    /// File name of the ONNX model within the repository.
    public var onnxFileName: String {
        switch self {
        case .nano, .nanoInt8: return "kitten_tts_nano_v0_8.onnx"
        case .micro:           return "kitten_tts_micro_v0_8.onnx"
        case .mini:            return "kitten_tts_mini_v0_8.onnx"
        }
    }

    /// File name of the voice embeddings archive within the repository.
    public var voicesFileName: String { "voices.npz" }

    /// Internal Hugging Face repository slug.
    public var repositoryID: String {
        switch self {
        case .nano: return "kitten-tts-nano-0.8"
        case .nanoInt8: return "kitten-tts-nano-0.8-int8"
        case .micro: return "kitten-tts-micro-0.8"
        case .mini: return "kitten-tts-mini-0.8"
        }
    }

    // MARK: - Sizing

    /// Approximate total download size in bytes (for UI progress labels).
    public var approximateDownloadBytes: Int64 {
        switch self {
        case .nano:    return 59_000_000   // ~56 MB ONNX + 3 MB voices
        case .nanoInt8: return 28_000_000  // ~25 MB ONNX + 3 MB voices
        case .micro:   return 44_000_000   // ~41 MB ONNX + 3 MB voices
        case .mini:    return 83_000_000   // ~80 MB ONNX + 3 MB voices
        }
    }

    // MARK: - Speed Priors

    /// Per-voice speed multiplier applied on top of the user-supplied speed.
    ///
    /// Matches the `speed_priors` field in each model's `config.json`.
    /// Models without speed priors (``micro``, ``mini``) return `1.0` for all voices.
    public func speedPrior(for voice: KittenVoice) -> Float {
        switch self {
        case .nano, .nanoInt8:
            // From config.json speed_priors
            switch voice {
            case .hugo: return 0.9
            default:    return 0.8
            }
        case .micro, .mini:
            return 1.0
        }
    }

    // MARK: - Display

    /// Human-readable display name.
    public var displayName: String {
        switch self {
        case .nano:    return "Nano (fp32)"
        case .nanoInt8: return "Nano (int8)"
        case .micro:   return "Micro"
        case .mini:    return "Mini"
        }
    }

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer().decode(String.self)
        guard let model = Self(id: value) else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "Unknown KittenTTS model: \(value)")
            )
        }
        self = model
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}
