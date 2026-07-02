import SwiftUI

@main
struct GPUFanControlApp: App {
    @StateObject private var poller = StatusPoller()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            HostListView()
                .environmentObject(poller)
                .onChange(of: scenePhase) { _, newPhase in
                    if newPhase == .active {
                        poller.liveActivities.reclaimExistingActivities()
                    }
                }
        }
    }
}
