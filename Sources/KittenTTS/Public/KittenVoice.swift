/// The eight voices bundled with the KittenTTS nano model.
///
/// Each voice has a recommended base speed (``defaultSpeed``) that is applied
/// multiplicatively with the caller-supplied speed when generating speech.
///
/// ```swift
/// let result = try await tts.generate("Hello!", options: .init(voice: "luna", speed: 1.2))
/// ```
public enum KittenVoice: String, CaseIterable, Identifiable, Sendable, Codable, Hashable, ExpressibleByStringLiteral {
    // MARK: - Cases

    /// Bella — female, warm and expressive.
    case bella  = "bella"
    /// Jasper — male, clear and conversational.
    case jasper = "jasper"
    /// Luna — female, calm and smooth.
    case luna   = "luna"
    /// Bruno — male, deep and steady.
    case bruno  = "bruno"
    /// Rosie — female, bright and friendly.
    case rosie  = "rosie"
    /// Hugo — male, authoritative.
    case hugo   = "hugo"
    /// Kiki — female, lively and energetic.
    case kiki   = "kiki"
    /// Leo — male, relaxed and natural.
    case leo    = "leo"

    public init(stringLiteral value: String) {
        guard let voice = Self(id: value) else {
            preconditionFailure("Unknown KittenTTS voice: \(value)")
        }
        self = voice
    }

    public init?(id: String) {
        switch id {
        case "bella", "expr-voice-2-f": self = .bella
        case "jasper", "expr-voice-2-m": self = .jasper
        case "luna", "expr-voice-3-f": self = .luna
        case "bruno", "expr-voice-3-m": self = .bruno
        case "rosie", "expr-voice-4-f": self = .rosie
        case "hugo", "expr-voice-4-m": self = .hugo
        case "kiki", "expr-voice-5-f": self = .kiki
        case "leo", "expr-voice-5-m": self = .leo
        default: return nil
        }
    }

    // MARK: - Identifiable

    /// Stable identifier — equal to the underlying voice key used in the model.
    public var id: String { rawValue }

    /// Internal key used in voices.npz embeddings.
    public var embeddingKey: String {
        switch self {
        case .bella: return "expr-voice-2-f"
        case .jasper: return "expr-voice-2-m"
        case .luna: return "expr-voice-3-f"
        case .bruno: return "expr-voice-3-m"
        case .rosie: return "expr-voice-4-f"
        case .hugo: return "expr-voice-4-m"
        case .kiki: return "expr-voice-5-f"
        case .leo: return "expr-voice-5-m"
        }
    }

    // MARK: - Metadata

    /// Human-readable display name (e.g. `"Bella"`).
    public var displayName: String {
        switch self {
        case .bella:  return "Bella"
        case .jasper: return "Jasper"
        case .luna:   return "Luna"
        case .bruno:  return "Bruno"
        case .rosie:  return "Rosie"
        case .hugo:   return "Hugo"
        case .kiki:   return "Kiki"
        case .leo:    return "Leo"
        }
    }

    /// `true` for female voices, `false` for male voices.
    public var isFemale: Bool {
        embeddingKey.hasSuffix("-f")
    }

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer().decode(String.self)
        guard let voice = Self(id: value) else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "Unknown KittenTTS voice: \(value)")
            )
        }
        self = voice
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}
