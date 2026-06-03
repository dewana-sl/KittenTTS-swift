import Foundation

/// Inference backend used by ``KittenTTS``.
public enum KittenTTSInferenceEngine: String, Sendable, Codable, Hashable {
    /// Existing ONNX Runtime backend. This remains the default.
    case onnx

    /// Experimental C++ native inference backend.
    case native
}

/// Native C++ engine model variant.
public enum KittenTTSNativeVariant: String, CaseIterable, Sendable, Codable, Hashable {
    case fp32_15m
    case fp32_40m
    case fp32_80m
    case int8_15m
    case int8_40m
    case int8_80m

    var size: String {
        String(rawValue.split(separator: "_")[1])
    }

    var precision: String {
        String(rawValue.split(separator: "_")[0])
    }

    var archFileName: String {
        "kitten_\(precision)_\(size)_arch"
    }

    var weightsFileName: String {
        "kitten_\(precision)_\(size).bin"
    }

    var voiceDirectoryName: String {
        "voices_kitten_\(size)"
    }
}

/// Local files for the native C++ backend.
public struct KittenTTSNativeModelFiles: Sendable, Equatable {
    /// Native engine architecture JSON.
    public let archURL: URL

    /// Native engine weight binary.
    public let weightsURL: URL

    /// Directory containing one `<voice-id>.bin` file per ``KittenVoice``.
    public let voiceDirectoryURL: URL

    public init(archURL: URL, weightsURL: URL, voiceDirectoryURL: URL) {
        self.archURL = archURL
        self.weightsURL = weightsURL
        self.voiceDirectoryURL = voiceDirectoryURL
    }
}

/// Configuration for the experimental native C++ backend.
public struct KittenTTSNativeConfig: Sendable, Equatable {
    /// Hugging Face repository containing native weights and voice binaries.
    public var modelRepository: String

    /// Optional explicit native variant. When omitted, the SDK maps from ``KittenModel``.
    public var variant: KittenTTSNativeVariant?

    /// Optional local native files. When set, downloads are skipped.
    public var modelFiles: KittenTTSNativeModelFiles?

    public init(
        modelRepository: String = "KittenML/meownn-models",
        variant: KittenTTSNativeVariant? = nil,
        modelFiles: KittenTTSNativeModelFiles? = nil
    ) {
        self.modelRepository = modelRepository
        self.variant = variant
        self.modelFiles = modelFiles
    }
}
