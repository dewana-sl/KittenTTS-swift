import Foundation

/// Maps IPA phoneme strings to the integer token sequences expected by the KittenTTS model.
///
/// The symbol table is identical to the Python `TextCleaner` in `kittentts/onnx_model.py`.
/// Unknown Unicode scalars are silently skipped
enum TextCleaner {

    // MARK: - Symbol table

    private static let pad: Character = "$"
    private static let punctuation: String = ";:,.!?¡¿—…\"«»\u{201C}\u{201D} "
    private static let lettersUpper: String = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    private static let lettersLower: String = "abcdefghijklmnopqrstuvwxyz"
    /// IPA symbols in the exact order defined by the KittenTTS Python source.
    private static let ipaSymbols: String =
        "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\u{2019}\u{0329}\u{2018}ᵻ"

    /// Token ID used for the start-of-sequence marker (also the pad token).
    static let startTokenID: Int64 = 0

    /// Token ID used for the end-of-sequence marker (index 10 = "…" ellipsis in the symbol table).
    static let endTokenID: Int64 = 10

    /// Token ID used for padding.
    static let padTokenID: Int64 = 0

    /// Symbol → index lookup built once at startup.
    private static let symbolIndex: [Unicode.Scalar: Int] = {
        let all = String(pad) + punctuation + lettersUpper + lettersLower + ipaSymbols
        var map: [Unicode.Scalar: Int] = [:]
        for (i, scalar) in all.unicodeScalars.enumerated() {
            map[scalar] = i
        }
        return map
    }()

    // MARK: - Encoding

    /// Encode an IPA phoneme string into a `[start, …tokens…, end, pad]` Int64 array.
    ///
    /// - Parameter phonemes: An IPA string produced by ``Phonemizer/phonemize(_:)``.
    /// - Returns: The corresponding Int64 token array ready to pass as `input_ids` to the model.
    static func encode(_ phonemes: String) -> [Int64] {
        var tokens: [Int64] = [startTokenID]
        for scalar in phonemes.unicodeScalars {
            if let idx = symbolIndex[scalar] {
                tokens.append(Int64(idx))
            }
        }
        tokens.append(endTokenID)
        tokens.append(padTokenID)
        return tokens
    }

    /// Encode IPA after applying the same basic tokenizer used by the Python SDK.
    static func encodeTokenized(_ phonemes: String) -> [Int64] {
        let tokenized = basicEnglishTokenize(phonemes).joined(separator: " ")
        return encode(tokenized)
    }

    private static func basicEnglishTokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        var currentWord = ""

        func flushWord() {
            guard !currentWord.isEmpty else { return }
            tokens.append(currentWord)
            currentWord.removeAll(keepingCapacity: true)
        }

        for scalar in text.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) || scalar == "_" {
                currentWord.unicodeScalars.append(scalar)
            } else if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                flushWord()
            } else {
                flushWord()
                tokens.append(String(scalar))
            }
        }

        flushWord()
        return tokens
    }
}
