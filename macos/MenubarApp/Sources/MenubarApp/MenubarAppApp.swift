import SwiftUI
import AppKit
import Combine

/// In a SwiftPM SwiftUI app, NSApplication isn't bootstrapped during
/// App.init(), so NSApp is nil there. The reliable place to set the
/// activation policy is the AppDelegate, wired in via the adaptor below.
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var model: AppModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Equivalent to Info.plist LSUIElement=YES: no Dock icon, menubar only.
        NSApp.setActivationPolicy(.accessory)
        model = AppModel()
    }
}

@main
struct MenubarAppApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Settings {
            EmptyView()
        }
    }
}

@MainActor
private final class AppModel {
    let poller = StatusPoller()
    let statusItemController = StatusItemController()

    init() {
        statusItemController.configure(poller: poller)
    }
}

@MainActor
private final class StatusItemController: NSObject {
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    private let popover = NSPopover()
    private weak var poller: StatusPoller?
    private var cancellables: Set<AnyCancellable> = []
    private var windows: [String: NSWindow] = [:]

    func configure(poller: StatusPoller) {
        self.poller = poller

        if let button = statusItem.button {
            button.target = self
            button.action = #selector(togglePopover)
            button.sendAction(on: [.leftMouseUp])
        }

        popover.behavior = .applicationDefined
        popover.animates = true
        popover.contentSize = NSSize(width: 700, height: 430)
        popover.contentViewController = NSHostingController(
            rootView: MenubarContent(
                closePopover: { [weak self] in self?.closePopover() },
                openAddHost: { [weak self] in self?.openAddHost() },
                openHostDashboard: { [weak self] host in self?.openHostDashboard(host: host) },
                openCurveEditor: { [weak self] hostID in self?.openCurveEditor(hostID: hostID) },
                openPowerLimitEditor: { [weak self] hostID in self?.openPowerLimitEditor(hostID: hostID) }
            )
            .environmentObject(poller)
        )

        poller.$hosts
            .combineLatest(poller.$states)
            .sink { [weak self] hosts, states in
                self?.updateStatusTitle(hosts: hosts, states: states)
            }
            .store(in: &cancellables)
        updateStatusTitle(hosts: poller.hosts, states: poller.states)
    }

    @objc private func togglePopover() {
        if popover.isShown {
            closePopover()
            return
        }
        guard let button = statusItem.button else { return }
        popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
        popover.contentViewController?.view.window?.makeKey()
    }

    private func closePopover() {
        popover.performClose(nil)
    }

    private func updateStatusTitle(hosts: [Host], states: [UUID: HostState]) {
        let parts = hosts.map { host in
            if let payload = states[host.id]?.lastPayload {
                let temps = payload.displayGPUs.compactMap { gpu -> String? in
                    guard let temp = gpu.vramTempC else { return nil }
                    return "\(temp)°"
                }
                if !temps.isEmpty {
                    return temps.joined(separator: "/")
                }
            }
            return "-"
        }
        statusItem.button?.title = parts.isEmpty ? "GPU" : parts.joined(separator: " ")
    }

    private func openAddHost() {
        guard let poller else { return }
        openWindow(key: "addHost", title: "Add Host") { [weak self] in
            AddHostWindow {
                self?.windows["addHost"]?.close()
            }
            .environmentObject(poller)
            .frame(width: 380)
        }
    }

    private func openCurveEditor(hostID: UUID) {
        guard let poller else { return }
        openWindow(key: "curve-\(hostID)", title: "Fan Curve") {
            CurveEditor(hostID: hostID)
                .environmentObject(poller)
        }
    }

    private func openHostDashboard(host: Host) {
        guard let url = URL(string: "http://\(host.hostname):\(host.port)/observer") else {
            return
        }
        NSWorkspace.shared.open(url)
    }

    private func openPowerLimitEditor(hostID: UUID) {
        guard let poller else { return }
        openWindow(key: "power-\(hostID)", title: "Power Limit") {
            PowerLimitEditor(hostID: hostID)
                .environmentObject(poller)
        }
    }

    private func openWindow<Content: View>(
        key: String,
        title: String,
        @ViewBuilder content: () -> Content
    ) {
        if let existing = windows[key] {
            existing.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 520, height: 480),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = title
        window.isReleasedWhenClosed = false
        window.contentViewController = NSHostingController(rootView: content())
        window.center()
        windows[key] = window
        NotificationCenter.default.addObserver(
            forName: NSWindow.willCloseNotification,
            object: window,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.windows.removeValue(forKey: key)
            }
        }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
