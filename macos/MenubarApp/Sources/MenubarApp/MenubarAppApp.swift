import SwiftUI
import AppKit

/// In a SwiftPM SwiftUI app, NSApplication isn't bootstrapped during
/// App.init(), so NSApp is nil there. The reliable place to set the
/// activation policy is the AppDelegate, wired in via the adaptor below.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Equivalent to Info.plist LSUIElement=YES: no Dock icon, menubar only.
        NSApp.setActivationPolicy(.accessory)
    }
}

@main
struct MenubarAppApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var poller = StatusPoller()

    var body: some Scene {
        MenuBarExtra {
            MenubarContent()
                .environmentObject(poller)
        } label: {
            MenubarLabel()
                .environmentObject(poller)
        }
        .menuBarExtraStyle(.window)
    }
}

/// The text shown directly in the menubar. Updates live as states change.
private struct MenubarLabel: View {
    @EnvironmentObject var poller: StatusPoller

    var body: some View {
        // Render "96° 91°" with each host in order. Unreachable hosts show "—".
        // macOS menubar items render in system text color, so per-temp colors
        // here are best-effort; they may be ignored by the menu bar appearance.
        let parts: [String] = poller.hosts.map { host in
            if let temp = poller.states[host.id]?.lastPayload?.vramTempC {
                return "\(temp)°"
            }
            return "—"
        }
        let hottest = poller.states.values
            .compactMap { $0.lastPayload?.vramTempC }
            .max()
        let combined = parts.joined(separator: " ")
        Text(combined.isEmpty ? "GPU" : combined)
            .foregroundColor(hottest.map(colorFor) ?? .primary)
    }
}
