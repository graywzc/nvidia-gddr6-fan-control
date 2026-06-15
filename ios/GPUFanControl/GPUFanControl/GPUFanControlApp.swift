import SwiftUI

@main
struct GPUFanControlApp: App {
    @StateObject private var poller = StatusPoller()

    var body: some Scene {
        WindowGroup {
            HostListView()
                .environmentObject(poller)
        }
    }
}
