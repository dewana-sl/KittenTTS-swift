# KittenTTS Public API Guidelines

## Preferred API

Use string-backed model and voice IDs. Swift keeps `.luna` and `.mini` as convenience, but examples should show the shared string shape first:

```swift
let config = KittenTTSConfig(
    model: "mini",
    defaultVoice: "luna",
    speed: 1.1
)

let tts = try await KittenTTS(config)

let result = try await tts.generate(
    "Hello",
    options: KittenTTSGenerateOptions(
        voice: "luna",
        speed: 1.1
    )
)

try await tts.play(result)
try await tts.speak("Hello", voice: "bella", speed: 1.0)

for try await chunk in tts.stream(longText, voice: "luna") {
    try await tts.play(chunk)
}
```

## Compatibility

Existing enum shorthand remains valid:

```swift
try await tts.speak("Hello", voice: .bella, speed: 1.0)
```

Internally, model repository IDs and voice embedding keys stay hidden behind the public IDs.
