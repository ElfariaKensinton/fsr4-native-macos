import Foundation

/// A CrossOver bottle and the fsr4-mac keys found in its cxbottle.conf.
struct Bottle: Identifiable, Hashable, Sendable {
    let name: String
    let conf: URL
    /// D3DMETAL_FRAMEWORK_PATH is set: the bottle loads the patched D3DMetal.
    let installed: Bool
    /// FSR4_ROOT: the folder the bottle's FSR files come from.
    let root: String?
    /// FSR4_PROFILE: lossy (FSR 4.1 Lite), exact-fast (FSR 4.1), ...
    let profile: String?
    let panel: Bool

    var id: String { conf.path }
}

enum Bottles {
    /// Set FSR4_INSTALLER_BOTTLES to list another folder (used by tests).
    static var directory: URL {
        if let custom = ProcessInfo.processInfo.environment["FSR4_INSTALLER_BOTTLES"], !custom.isEmpty {
            return URL(fileURLWithPath: custom)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/CrossOver/Bottles")
    }

    /// `"KEY" = "VALUE"` lines of a cxbottle.conf. The last occurrence wins.
    static func values(in text: String) -> [String: String] {
        var result: [String: String] = [:]
        for line in text.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("\"") else { continue }
            let parts = trimmed.split(separator: "=", maxSplits: 1)
            guard parts.count == 2, let key = unquote(parts[0]), let value = unquote(parts[1]) else { continue }
            result[key] = value
        }
        return result
    }

    static func bottle(conf: URL) -> Bottle? {
        guard let data = try? Data(contentsOf: conf) else { return nil }
        let values = values(in: String(decoding: data, as: UTF8.self))
        return Bottle(
            name: conf.deletingLastPathComponent().lastPathComponent,
            conf: conf,
            installed: values["D3DMETAL_FRAMEWORK_PATH"] != nil,
            root: values["FSR4_ROOT"],
            profile: values["FSR4_PROFILE"],
            panel: values["FSR4_PANEL"] == "1")
    }

    static func list(in directory: URL = directory) -> [Bottle] {
        let items = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        return items
            .compactMap { bottle(conf: $0.appendingPathComponent("cxbottle.conf")) }
            .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    private static func unquote(_ text: Substring) -> String? {
        let value = text.trimmingCharacters(in: .whitespaces)
        guard value.count >= 2, value.hasPrefix("\""), value.hasSuffix("\"") else { return nil }
        return String(value.dropFirst().dropLast())
    }
}
