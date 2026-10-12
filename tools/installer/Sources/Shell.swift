import Darwin
import Foundation

/// The PATH a GUI app gets from launchd. Commands behave the same whether the
/// app is double-clicked or started from Terminal.
let systemPath = "/usr/bin:/bin:/usr/sbin:/sbin"

struct CommandResult: Sendable {
    let status: Int32
    let output: String
    var ok: Bool { status == 0 }
}

func commandEnvironment() -> [String: String] {
    var environment = ProcessInfo.processInfo.environment
    environment["PATH"] = systemPath
    environment["PYTHONUNBUFFERED"] = "1"  // stream installer output line by line
    return environment
}

/// Single-quotes a value for /bin/sh.
func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

/// Runs a short command and returns its combined stdout and stderr.
func runCommand(_ executable: String, _ arguments: [String]) -> CommandResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    process.environment = commandEnvironment()
    process.standardInput = FileHandle.nullDevice
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    do {
        try process.run()
    } catch {
        return CommandResult(status: -1, output: error.localizedDescription)
    }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return CommandResult(status: process.terminationStatus, output: String(decoding: data, as: UTF8.self))
}

/// Runs a shell command with administrator privileges (macOS password prompt).
/// The command is passed as an argument, so it needs no AppleScript escaping.
func runAdministrator(_ command: String) -> CommandResult {
    runCommand("/usr/bin/osascript", [
        "-e", "on run argv",
        "-e", "do shell script (item 1 of argv) with administrator privileges",
        "-e", "end run",
        command,
    ])
}

/// A long-running command whose output is streamed to the UI.
///
/// The command starts in its own process group (python3 calls setpgrp and then
/// execs it), so cancel() also stops the compilers and test tools it spawns.
final class StreamingCommand: @unchecked Sendable {
    private static let groupLeader = "import os, sys; os.setpgrp(); os.execv(sys.argv[1], sys.argv[1:])"

    private let process = Process()
    private let lock = NSLock()
    private var cancelled = false

    init(_ argv: [String], directory: URL? = nil) {
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-c", Self.groupLeader] + argv
        process.environment = commandEnvironment()
        process.standardInput = FileHandle.nullDevice
        if let directory { process.currentDirectoryURL = directory }
    }

    /// `output` receives whole lines; both callbacks run on the main actor.
    func start(output: @escaping @MainActor (String) -> Void,
               completion: @escaping @MainActor (_ status: Int32, _ cancelled: Bool) -> Void) throws {
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        let finished = DispatchGroup()
        finished.enter()  // end of output
        finished.enter()  // process exit
        let buffer = LineBuffer()
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            let text = data.isEmpty ? buffer.flush() : buffer.append(data)
            if !text.isEmpty {
                DispatchQueue.main.async { MainActor.assumeIsolated { output(text) } }
            }
            if data.isEmpty {
                handle.readabilityHandler = nil
                finished.leave()
            }
        }
        process.terminationHandler = { _ in finished.leave() }
        do {
            try process.run()
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            throw error
        }
        finished.notify(queue: .main) { [self] in
            let status = process.terminationStatus
            let wasCancelled = lock.withLock { cancelled }
            MainActor.assumeIsolated { completion(status, wasCancelled) }
        }
    }

    func cancel() {
        lock.withLock { cancelled = true }
        guard process.isRunning else { return }
        let pid = process.processIdentifier
        // Signal the whole group once the leader has called setpgrp.
        if getpgid(pid) == pid {
            kill(-pid, SIGTERM)
        } else {
            process.terminate()
        }
    }
}

/// Splits a byte stream into complete UTF-8 lines. Readability handlers for one
/// file handle are serial, so no locking is needed.
private final class LineBuffer: @unchecked Sendable {
    private var pending = Data()

    func append(_ data: Data) -> String {
        pending.append(data)
        guard let newline = pending.lastIndex(of: UInt8(ascii: "\n")) else { return "" }
        let lines = pending[...newline]
        pending = Data(pending[pending.index(after: newline)...])
        return String(decoding: lines, as: UTF8.self)
    }

    func flush() -> String {
        defer { pending = Data() }
        return pending.isEmpty ? "" : String(decoding: pending, as: UTF8.self) + "\n"
    }
}
