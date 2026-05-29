import SwiftUI
import AppKit

@main
struct MenubarAppApp: App {
    @StateObject private var poller = StatusPoller()

    init() {
        // Equivalent to Info.plist LSUIElement=YES: no Dock icon, menubar only.
        // Setting this in init() runs before the app's main scene is built.
        NSApp.setActivationPolicy(.accessory)
    }

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
