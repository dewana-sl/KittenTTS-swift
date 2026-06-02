import XCTest
@testable import KittenTTS

final class TextCleanerTests: XCTestCase {

    // MARK: - Token structure

    func testEncodeReturnsAtLeastThreeTokens() {
        // [start, ...body..., end, pad]
        let tokens = TextCleaner.encode("a")
        XCTAssertGreaterThanOrEqual(tokens.count, 3)
    }

    func testEncodeStartAndEnd() {
        let tokens = TextCleaner.encode("hello")
        XCTAssertEqual(tokens.first, TextCleaner.startTokenID)
        XCTAssertEqual(tokens[tokens.count - 2], TextCleaner.endTokenID)
        XCTAssertEqual(tokens.last,  TextCleaner.padTokenID)
    }

    func testEmptyStringReturnsThreeTokens() {
        let tokens = TextCleaner.encode("")
        // start + end + pad = 3
        XCTAssertEqual(tokens.count, 3)
        XCTAssertEqual(tokens[0], TextCleaner.startTokenID)
        XCTAssertEqual(tokens[1], TextCleaner.endTokenID)
        XCTAssertEqual(tokens[2], TextCleaner.padTokenID)
    }

    func testKnownIpaSymbols() {
        // Space (index 16 in the symbol table) should encode to 16
        let tokens = TextCleaner.encode(" ")
        XCTAssertEqual(tokens.count, 4)  // start, 16, end, pad
        XCTAssertEqual(tokens[1], 16)
    }

    func testUnknownScalarsSkipped() {
        // A rare Unicode character not in the symbol table
        let tokens = TextCleaner.encode("\u{1F600}") // 😀
        // Should just be start + end + pad with nothing in between
        XCTAssertEqual(tokens.count, 3)
    }

    func testTokenIDsAreNonNegative() {
        let tokens = TextCleaner.encode("Hello world!")
        for token in tokens {
            XCTAssertGreaterThanOrEqual(token, 0)
        }
    }

    func testEncodeTokenizedMatchesPythonQuickBrownIPA() {
        let ipa = "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ."
        let tokens = TextCleaner.encodeTokenized(ipa)
        XCTAssertEqual(tokens, [
            0, 81, 83, 16, 53, 65, 156, 102, 53, 16, 44, 123, 156, 43, 135,
            56, 16, 48, 156, 69, 158, 53, 61, 16, 46, 147, 156, 138, 55, 58,
            61, 16, 157, 57, 135, 64, 85, 16, 81, 83, 16, 54, 156, 47, 102,
            68, 51, 16, 46, 156, 69, 158, 92, 16, 4, 10, 0
        ])
    }

    // MARK: - Constants

    func testEndTokenIsEllipsis() {
        // Index 10 should be "…" — verify by checking the position manually
        // pad($)=0, then punctuation ";:,.!?¡¿—…" → "…" is the 10th character (index 10)
        XCTAssertEqual(TextCleaner.endTokenID, 10)
    }
}
