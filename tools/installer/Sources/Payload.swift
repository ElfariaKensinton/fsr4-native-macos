import Foundation

/// Version of the FSR files bundled in the app (Resources/payload.json).
struct PayloadInfo: Codable, Equatable, Sendable {
    let version: String
    let commit: String
}

struct BundledPayload: Sendable {
    let info: PayloadInfo
    let archive: URL
}

enum PayloadError: LocalizedError {
    case extract(String)

    var errorDescription: String? {
        switch self {
        case .extract(let output):
            return String(localized: "Could not unpack the FSR files: \(output)")
        }
    }
}

/// The FSR files live in a fixed folder outside the app: the bottle points at
/// them, and the hook compiles and caches shaders there while games run.
enum Payload {
    static let marker = ".fsr4-payload.json"
    /// Kept on update: shader cache (build), patched D3DMetal, logs and bottle
    /// backups (dist), the installer's test frames (testcases).
    static let preserved: Set<String> = ["build", "dist", "testcases", marker]

    /// Set FSR4_INSTALLER_ROOT to use another folder (used by tests).
    static var installRoot: URL {
        if let custom = ProcessInfo.processInfo.environment["FSR4_INSTALLER_ROOT"], !custom.isEmpty {
            return URL(fileURLWithPath: custom)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/fsr4-native-macos")
    }

    static func bundled(in bundle: Bundle = .main) -> BundledPayload? {
        guard let json = bundle.url(forResource: "payload", withExtension: "json"),
              let archive = bundle.url(forResource: "payload", withExtension: "tar.gz"),
              let data = try? Data(contentsOf: json),
              let info = try? JSONDecoder().decode(PayloadInfo.self, from: data) else { return nil }
        return BundledPayload(info: info, archive: archive)
    }

    static func installed(at root: URL) -> PayloadInfo? {
        guard let data = try? Data(contentsOf: root.appendingPathComponent(marker)) else { return nil }
        return try? JSONDecoder().decode(PayloadInfo.self, from: data)
    }

    static func script(in root: URL) -> URL {
        root.appendingPathComponent("tools/crossover/crossover_fsr4.py")
    }

    /// Unpacks the bundled files into `root` unless that version is already
    /// there. An older version is replaced; caches in `preserved` stay. Returns
    /// whether files were unpacked.
    @discardableResult
    static func ensure(root: URL, payload: BundledPayload) throws -> Bool {
        let fm = FileManager.default
        let current = installed(at: root)
        if current == payload.info && fm.fileExists(atPath: script(in: root).path) { return false }
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        // Only a folder carrying our marker is cleaned, never an unrelated one.
        if current != nil {
            for item in try fm.contentsOfDirectory(atPath: root.path) where !preserved.contains(item) {
                try fm.removeItem(at: root.appendingPathComponent(item))
            }
        }
        let tar = runCommand("/usr/bin/tar", ["-xzf", payload.archive.path, "-C", root.path])
        guard tar.ok else { throw PayloadError.extract(tar.output) }
        // Files copied out of a downloaded app may inherit quarantine.
        _ = runCommand("/usr/bin/xattr", ["-dr", "com.apple.quarantine", root.path])
        let data = try JSONEncoder().encode(payload.info)
        try data.write(to: root.appendingPathComponent(marker), options: .atomic)
        return true
    }
}
