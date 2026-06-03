import Foundation

struct NativeAssetBundle {
    let archURL: URL
    let weightsURL: URL
    let voiceDirectoryURL: URL
}

enum NativeAssetDownloader {
    static func isNativeModelCached(for config: KittenTTSConfig) -> Bool {
        if let files = config.nativeConfig.modelFiles {
            return nativeFilesExist(files)
        }

        guard bundledArchURL(for: resolvedVariant(for: config)) != nil else {
            return false
        }

        let dir = nativeDirectory(for: config)
        let variant = resolvedVariant(for: config)
        let weights = dir.appendingPathComponent(variant.weightsFileName)
        let voices = dir.appendingPathComponent(variant.voiceDirectoryName, isDirectory: true)
        return FileManager.default.fileExists(atPath: weights.path) &&
               KittenVoice.allCases.allSatisfy {
                   FileManager.default.fileExists(atPath: voices.appendingPathComponent("\($0.rawValue).bin").path)
               }
    }

    static func downloadNativeAssetsIfNeeded(
        for config: KittenTTSConfig,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> NativeAssetBundle {
        if let files = config.nativeConfig.modelFiles {
            guard nativeFilesExist(files) else {
                throw KittenTTSError.invalidModelData("Native model files are missing or incomplete.")
            }
            progressHandler?(1.0)
            return NativeAssetBundle(
                archURL: files.archURL,
                weightsURL: files.weightsURL,
                voiceDirectoryURL: files.voiceDirectoryURL
            )
        }

        let variant = resolvedVariant(for: config)
        guard let archURL = bundledArchURL(for: variant) else {
            throw KittenTTSError.invalidModelData("Missing bundled native architecture for \(variant.rawValue).")
        }

        let dir = nativeDirectory(for: config)
        let voicesDir = dir.appendingPathComponent(variant.voiceDirectoryName, isDirectory: true)
        let weightsURL = dir.appendingPathComponent(variant.weightsFileName)

        do {
            try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        } catch {
            throw KittenTTSError.downloadFailed("Cannot create native cache directory: \(error.localizedDescription)")
        }

        var completed = 0
        let total = KittenVoice.allCases.count + 1
        func advance() {
            completed += 1
            progressHandler?(Double(completed) / Double(total))
        }

        if !FileManager.default.fileExists(atPath: weightsURL.path) {
            try await ModelDownloader.downloadFile(
                from: remoteURL(
                    repository: config.nativeConfig.modelRepository,
                    path: variant.weightsFileName
                ),
                to: weightsURL,
                progressHandler: { p in
                    progressHandler?(p / Double(total))
                }
            )
        }
        advance()

        for voice in KittenVoice.allCases {
            let voiceURL = voicesDir.appendingPathComponent("\(voice.rawValue).bin")
            if !FileManager.default.fileExists(atPath: voiceURL.path) {
                try await ModelDownloader.downloadFile(
                    from: remoteURL(
                        repository: config.nativeConfig.modelRepository,
                        path: "\(variant.voiceDirectoryName)/\(voice.rawValue).bin"
                    ),
                    to: voiceURL,
                    progressHandler: { _ in }
                )
            }
            advance()
        }

        return NativeAssetBundle(
            archURL: archURL,
            weightsURL: weightsURL,
            voiceDirectoryURL: voicesDir
        )
    }

    private static func nativeFilesExist(_ files: KittenTTSNativeModelFiles) -> Bool {
        FileManager.default.fileExists(atPath: files.archURL.path) &&
        FileManager.default.fileExists(atPath: files.weightsURL.path) &&
        FileManager.default.fileExists(atPath: files.voiceDirectoryURL.path)
    }

    private static func nativeDirectory(for config: KittenTTSConfig) -> URL {
        config.resolvedStorageDirectory
            .appendingPathComponent("native", isDirectory: true)
            .appendingPathComponent(resolvedVariant(for: config).rawValue, isDirectory: true)
    }

    private static func resolvedVariant(for config: KittenTTSConfig) -> KittenTTSNativeVariant {
        config.nativeConfig.variant ?? config.model.nativeVariant
    }

    private static func bundledArchURL(for variant: KittenTTSNativeVariant) -> URL? {
        Bundle.module.url(forResource: variant.archFileName, withExtension: "json") ??
        Bundle.module.url(forResource: variant.archFileName, withExtension: "json", subdirectory: "native_arch")
    }

    private static func remoteURL(repository: String, path: String) -> URL {
        URL(string: "https://huggingface.co/\(repository)/resolve/main/\(path)")!
    }
}
