import Foundation
import SwiftUI

/// Polls /status on each configured host on a 1 Hz timer.
/// Publishes per-host state for the SwiftUI views to observe.
@MainActor
final class StatusPoller: ObservableObject {
    @Published var hosts: [Host] = [] {
        didSet { saveHosts() }
    }
    @Published var states: [UUID: HostState] = [:]

    private var timer: Timer?
    private let storageKey = "configuredHosts"

    init() {
        loadHosts()
        start()
    }

    func start() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                await self?.tick()
            }
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    func addHost(_ host: Host) {
        hosts.append(host)
    }

    func removeHost(_ host: Host) {
        hosts.removeAll { $0.id == host.id }
        states.removeValue(forKey: host.id)
    }

    private func tick() async {
        // Fire all requests concurrently.
        await withTaskGroup(of: (UUID, Result<HostStatusPayload, Error>).self) { group in
            for host in hosts {
                group.addTask { [host] in
                    do {
                        let payload = try await Self.fetchStatus(host: host)
                        return (host.id, .success(payload))
                    } catch {
                        return (host.id, .failure(error))
                    }
                }
            }
            for await (id, result) in group {
                var state = states[id] ?? HostState()
                switch result {
                case .success(let payload):
                    state.lastPayload = payload
                    state.lastFetchedAt = Date()
                    state.lastError = nil
                case .failure(let err):
                    state.lastError = err.localizedDescription
                    // Keep lastPayload so menubar shows the last known value
                    // while marked stale, rather than empty on a single blip.
                }
                states[id] = state
            }
        }
    }

    private static func fetchStatus(host: Host) async throws -> HostStatusPayload {
        guard let url = host.statusURL else {
            throw NSError(domain: "MenubarApp", code: 1, userInfo: [NSLocalizedDescriptionKey: "bad url"])
        }
        var req = URLRequest(url: url)
        req.timeoutInterval = 2.5
        if !host.token.isEmpty {
            req.setValue("Bearer \(host.token)", forHTTPHeaderField: "Authorization")
        }
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
            throw NSError(domain: "MenubarApp", code: code, userInfo: [NSLocalizedDescriptionKey: "HTTP \(code)"])
        }
        return try JSONDecoder().decode(HostStatusPayload.self, from: data)
    }

    // MARK: - Persistence

    private func loadHosts() {
        guard let data = UserDefaults.standard.data(forKey: storageKey) else { return }
        if let decoded = try? JSONDecoder().decode([Host].self, from: data) {
            hosts = decoded
        }
    }

    private func saveHosts() {
        if let data = try? JSONEncoder().encode(hosts) {
            UserDefaults.standard.set(data, forKey: storageKey)
        }
    }
}
