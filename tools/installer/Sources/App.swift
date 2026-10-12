import AppKit
import SwiftUI

@main
struct FSR4InstallerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var model = InstallerModel()

    var body: some Scene {
        Window("FSR 4 for CrossOver", id: "main") {
            ContentView(model: model)
                .frame(minWidth: 600, minHeight: 560)
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
                    // Pick up Xcode, CrossOver or bottles installed meanwhile.
                    if !model.busy { model.refresh() }
                }
        }
        .defaultSize(width: 640, height: 720)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandGroup(replacing: .help) {
                Link(String(localized: "FSR 4 Guide"), destination: ContentView.guideURL)
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    /// `--snapshot <png>`: render the window once the checks finish, then quit
    /// (README screenshots; same option as the FSR 4 panel app).
    func applicationDidFinishLaunching(_ notification: Notification) {
        let arguments = CommandLine.arguments
        guard let index = arguments.firstIndex(of: "--snapshot"), index + 1 < arguments.count else { return }
        let path = arguments[index + 1]
        DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
            // The frame view includes the window background behind the content.
            guard let content = NSApp.windows.first(where: \.isVisible)?.contentView,
                  let view = content.superview ?? Optional(content),
                  let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { exit(1) }
            view.cacheDisplay(in: view.bounds, to: bitmap)
            let written = (try? bitmap.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))) != nil
            exit(written ? 0 : 1)
        }
    }
}
