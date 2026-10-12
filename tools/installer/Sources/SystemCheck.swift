import Foundation
import Metal

/// One row of the requirements list.
struct Requirement: Identifiable, Sendable {
    enum Kind: String, Sendable { case xcode, metal, rosetta, crossover }

    /// How the user (or the app) fixes a missing requirement.
    enum Fix: Sendable {
        case open(URL)
        /// Shell command run with administrator privileges.
        case administrator(String)
        /// Command run without privileges, output streamed to the log.
        case stream([String])
    }

    let kind: Kind
    let title: String
    let ok: Bool
    let detail: String
    var fix: Fix? = nil
    var fixTitle: String? = nil

    var id: Kind { kind }
}

enum SystemCheck {
    static let minimumXcode = 26
    static let xcodeStore = URL(string: "macappstore://apps.apple.com/app/xcode/id497799835")!
    static let crossoverSite = URL(string: "https://www.codeweavers.com/crossover")!
    static let d3dmetal = "Contents/SharedSupport/CrossOver/lib/apple_gptk/external/D3DMetal.framework"
    /// Same search order as tools/crossover/crossover_fsr4.py.
    static var crossoverApps: [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return ["/Applications/CrossOver.app", "/Applications/CrossOver Preview.app"].map(URL.init(fileURLWithPath:))
            + ["Applications/CrossOver.app", "Applications/CrossOver Preview.app"].map(home.appendingPathComponent)
    }

    static func all() -> [Requirement] {
        let xcode = checkXcode()
        return [xcode, checkMetal(xcodeReady: xcode.ok), checkRosetta(), checkCrossOver()]
    }

    // MARK: Xcode

    static func installedXcodes() -> [URL] {
        let fm = FileManager.default
        let folders = [URL(fileURLWithPath: "/Applications"), fm.homeDirectoryForCurrentUser.appendingPathComponent("Applications")]
        let apps = folders.flatMap { folder in
            ((try? fm.contentsOfDirectory(atPath: folder.path)) ?? [])
                .filter { $0.hasPrefix("Xcode") && $0.hasSuffix(".app") }
                .map { folder.appendingPathComponent($0) }
        }
        // Prefer a plain Xcode.app over betas.
        return apps.sorted { ($0.lastPathComponent != "Xcode.app" ? 1 : 0, $0.path) < ($1.lastPathComponent != "Xcode.app" ? 1 : 0, $1.path) }
    }

    static func xcodeMajor(_ app: URL) -> Int? {
        let plist = NSDictionary(contentsOf: app.appendingPathComponent("Contents/Info.plist"))
        guard let version = plist?["CFBundleShortVersionString"] as? String else { return nil }
        return Int(version.split(separator: ".").first ?? "")
    }

    /// The Xcode.app that xcode-select points at, if any.
    static func selectedXcode() -> URL? {
        let selected = runCommand("/usr/bin/xcode-select", ["-p"])
        let path = selected.output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard selected.ok, let range = path.range(of: ".app/Contents/Developer") else { return nil }
        let app = URL(fileURLWithPath: String(path[..<range.lowerBound]) + ".app")
        return FileManager.default.fileExists(atPath: app.path) ? app : nil
    }

    static func setupCommand(_ app: URL) -> String {
        let developer = shellQuote(app.appendingPathComponent("Contents/Developer").path)
        let xcodebuild = shellQuote(app.appendingPathComponent("Contents/Developer/usr/bin/xcodebuild").path)
        return "/usr/bin/xcode-select -s \(developer) && \(xcodebuild) -license accept && \(xcodebuild) -runFirstLaunch"
    }

    static func checkXcode() -> Requirement {
        let title = String(localized: "Xcode \(minimumXcode)+")
        let installed = installedXcodes()
        let suitable = installed.filter { (xcodeMajor($0) ?? 0) >= minimumXcode }
        guard let candidate = suitable.first else {
            let detail = installed.isEmpty
                ? String(localized: "Needed to build the hook and FSR shaders. Free in the App Store.")
                : String(localized: "Installed Xcode is too old. Update it in the App Store.")
            return Requirement(kind: .xcode, title: title, ok: false, detail: detail,
                               fix: .open(xcodeStore), fixTitle: String(localized: "Open App Store"))
        }
        let setup = Requirement(kind: .xcode, title: title, ok: false,
                                detail: String(localized: "\(candidate.lastPathComponent) is installed but not set up for command-line tools."),
                                fix: .administrator(setupCommand(candidate)), fixTitle: String(localized: "Set Up…"))
        guard let selected = selectedXcode(), (xcodeMajor(selected) ?? 0) >= minimumXcode else { return setup }
        let xcodebuild = selected.appendingPathComponent("Contents/Developer/usr/bin/xcodebuild").path
        // Functional check: -checkFirstLaunchStatus can report pending extras
        // while the compilers already work.
        guard runCommand(xcodebuild, ["-license", "check"]).ok,
              runCommand("/usr/bin/xcrun", ["--sdk", "macosx", "--show-sdk-path"]).ok,
              runCommand("/usr/bin/xcrun", ["clang", "--version"]).ok else {
            return Requirement(kind: .xcode, title: title, ok: false,
                               detail: String(localized: "Accept the Xcode license and finish its first-launch setup."),
                               fix: .administrator(setupCommand(selected)), fixTitle: String(localized: "Set Up…"))
        }
        let version = (NSDictionary(contentsOf: selected.appendingPathComponent("Contents/Info.plist"))?["CFBundleShortVersionString"] as? String) ?? ""
        return Requirement(kind: .xcode, title: title, ok: true, detail: "\(selected.lastPathComponent) \(version)")
    }

    // MARK: Metal Toolchain

    static func checkMetal(xcodeReady: Bool) -> Requirement {
        let title = String(localized: "Metal Toolchain")
        guard xcodeReady else {
            return Requirement(kind: .metal, title: title, ok: false, detail: String(localized: "Set up Xcode first."))
        }
        let metal = runCommand("/usr/bin/xcrun", ["metal", "--version"])
        if metal.ok, let line = metal.output.split(separator: "\n").first, line.contains("metal version") {
            return Requirement(kind: .metal, title: title, ok: true, detail: String(line))
        }
        return Requirement(kind: .metal, title: title, ok: false,
                           detail: String(localized: "Xcode component that compiles FSR shaders for each game."),
                           fix: .stream(["/usr/bin/xcodebuild", "-downloadComponent", "MetalToolchain"]),
                           fixTitle: String(localized: "Download"))
    }

    // MARK: Rosetta

    static func checkRosetta() -> Requirement {
        let title = String(localized: "Rosetta 2")
        if FileManager.default.fileExists(atPath: "/Library/Apple/usr/libexec/oah/libRosettaRuntime") {
            return Requirement(kind: .rosetta, title: title, ok: true, detail: String(localized: "Installed"))
        }
        return Requirement(kind: .rosetta, title: title, ok: false,
                           detail: String(localized: "CrossOver runs Windows games as Intel code."),
                           fix: .administrator("/usr/sbin/softwareupdate --install-rosetta --agree-to-license"),
                           fixTitle: String(localized: "Install"))
    }

    // MARK: CrossOver

    static func checkCrossOver() -> Requirement {
        let title = String(localized: "CrossOver 24+")
        let fm = FileManager.default
        let apps = crossoverApps.filter { fm.fileExists(atPath: $0.path) }
        if let app = apps.first(where: { fm.fileExists(atPath: $0.appendingPathComponent(d3dmetal).path) }) {
            return Requirement(kind: .crossover, title: title, ok: true, detail: app.deletingPathExtension().lastPathComponent)
        }
        let detail = apps.isEmpty
            ? String(localized: "Runs the Windows game. Install it to /Applications.")
            : String(localized: "This CrossOver has no D3DMetal. Update to version 24 or newer.")
        return Requirement(kind: .crossover, title: title, ok: false, detail: detail,
                           fix: .open(crossoverSite), fixTitle: String(localized: "Download"))
    }

    /// A wineserver runs while any CrossOver bottle is open.
    static func bottleRunning() -> Bool {
        runCommand("/usr/bin/pgrep", ["-x", "wineserver"]).ok
    }
}

/// The Mac's chip and which FSR backend it gets (see src/fsr4_device.h).
enum Chip {
    static var name: String {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else { return "Mac" }
        var bytes = [UInt8](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &bytes, &size, nil, 0) == 0 else { return "Mac" }
        return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }

    /// Apple GPU family 10 (M5 and newer) runs FSR's INT8 network natively.
    static var hasNeuralAccelerators: Bool {
        guard let device = MTLCreateSystemDefaultDevice(),
              let family = MTLGPUFamily(rawValue: 1010) else { return false }
        return device.supportsFamily(family)
    }
}
