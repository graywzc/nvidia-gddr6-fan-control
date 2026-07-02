import ActivityKit
import Combine
import Foundation

@MainActor
final class LiveActivityManager: ObservableObject {
    @Published private(set) var pinnedHostID: UUID? {
        didSet {
            if let id = pinnedHostID {
                UserDefaults.standard.set(id.uuidString, forKey: "pinnedLiveActivityHostID")
            } else {
                UserDefaults.standard.removeObject(forKey: "pinnedLiveActivityHostID")
            }
        }
    }

    private var activity: Activity<GPUStatusActivityAttributes>?
    private var lastPushedState: GPUStatusActivityAttributes.ContentState?
    private var lastPushDate: Date?

    init() {
        loadPinnedHost()
    }

    func setPinnedHost(_ host: Host?) {
        Task {
            await activity?.end(nil, dismissalPolicy: .immediate)
            activity = nil
            lastPushedState = nil
            lastPushDate = nil

            pinnedHostID = host?.id

            guard let host else { return }

            startActivity(for: host)
        }
    }

    func update(host: Host, payload: HostStatusPayload) {
        guard host.id == pinnedHostID else { return }

        if activity == nil {
            startActivity(for: host)
        }

        let state = GPUStatusActivityAttributes.ContentState.from(payload: payload, at: Date())
        let now = state.updatedAt

        guard Self.shouldPush(previous: lastPushedState, lastPushDate: lastPushDate, new: state, now: now) else {
            return
        }

        lastPushedState = state
        lastPushDate = now

        Task {
            await activity?.update(
                ActivityContent(
                    state: state,
                    staleDate: Date().addingTimeInterval(15)
                )
            )
        }
    }

    func reclaimExistingActivities() {
        for activity in Activity<GPUStatusActivityAttributes>.activities {
            let attributes = activity.attributes
            if let pinnedHostID, attributes.hostID == pinnedHostID.uuidString {
                self.activity = activity
            } else {
                Task {
                    await activity.end(nil, dismissalPolicy: .immediate)
                }
            }
        }
    }

    // MARK: - Internal (for testing)

    nonisolated static func shouldPush(
        previous: GPUStatusActivityAttributes.ContentState?,
        lastPushDate: Date?,
        new: GPUStatusActivityAttributes.ContentState,
        now: Date
    ) -> Bool {
        guard let previous, let lastPushDate else {
            return true
        }

        let valuesChanged =
            previous.vramTempC != new.vramTempC ||
            previous.fanPct != new.fanPct ||
            previous.gpuUtilPct != new.gpuUtilPct ||
            previous.powerW != new.powerW

        if valuesChanged {
            return true
        }

        let elapsed = now.timeIntervalSince(lastPushDate)
        return elapsed >= 5
    }

    // MARK: - Private

    private func startActivity(for host: Host) {
        guard ActivityAuthorizationInfo().areActivitiesEnabled else { return }
        lastPushedState = nil
        lastPushDate = nil
        do {
            activity = try Activity.request(
                attributes: GPUStatusActivityAttributes(
                    hostID: host.id.uuidString,
                    hostName: host.name
                ),
                content: ActivityContent(
                    state: Self.makeInitialState(),
                    staleDate: Date().addingTimeInterval(15)
                )
            )
        } catch {
            print("LiveActivityManager: failed to request activity: \(error)")
        }
    }

    private func loadPinnedHost() {
        guard let uuidString = UserDefaults.standard.string(forKey: "pinnedLiveActivityHostID") else {
            return
        }
        pinnedHostID = UUID(uuidString: uuidString)
    }

    private static func makeInitialState() -> GPUStatusActivityAttributes.ContentState {
        GPUStatusActivityAttributes.ContentState(
            vramTempC: nil,
            fanPct: nil,
            gpuUtilPct: nil,
            powerW: nil,
            updatedAt: Date()
        )
    }
}

// MARK: - ContentState from HostStatusPayload (app target only)

extension GPUStatusActivityAttributes.ContentState {
    static func from(payload: HostStatusPayload, at date: Date) -> Self {
        let hottest = payload.displayGPUs.compactMap(\.vramTempC).max()
        let maxFan = payload.displayGPUs.compactMap(\.fanPct).max()
        let maxUtil = payload.displayGPUs.compactMap(\.gpuUtilPct).max()
        let maxPower = payload.displayGPUs.compactMap(\.powerW).max()

        return Self(
            vramTempC: hottest,
            fanPct: maxFan,
            gpuUtilPct: maxUtil,
            powerW: maxPower,
            updatedAt: date
        )
    }
}
