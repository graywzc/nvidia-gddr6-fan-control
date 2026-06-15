import Foundation
import SwiftUI

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

    func putCurve(host: Host, curve: [[Int]], gpuIndex: Int? = nil) async throws {
        guard let url = URL(string: "http://\(host.hostname):\(host.port)/curve") else {
            throw PollerError("Bad URL")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.timeoutInterval = 5
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !host.token.isEmpty {
            request.setValue("Bearer \(host.token)", forHTTPHeaderField: "Authorization")
        }
        if let gpuIndex {
            request.httpBody = try JSONSerialization.data(withJSONObject: [
                "gpu_index": gpuIndex,
                "curve": curve,
            ])
        } else {
            request.httpBody = try JSONSerialization.data(withJSONObject: curve)
        }
        try await send(request)
    }

    func putPowerLimit(host: Host, watts: Double?, gpuIndex: Int? = nil) async throws {
        guard let url = URL(string: "http://\(host.hostname):\(host.port)/power-limit") else {
            throw PollerError("Bad URL")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.timeoutInterval = 5
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !host.token.isEmpty {
            request.setValue("Bearer \(host.token)", forHTTPHeaderField: "Authorization")
        }
        var body: [String: Any] = ["power_limit_w": watts ?? NSNull()]
        if let gpuIndex {
            body["gpu_index"] = gpuIndex
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        try await send(request)
    }

    private func tick() async {
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
                    let util = payload.gpus?.compactMap(\.gpuUtilPct).max() ?? payload.gpuUtilPct
                    if let util {
                        state.utilHistory.append(util)
                        if state.utilHistory.count > 60 {
                            state.utilHistory.removeFirst(state.utilHistory.count - 60)
                        }
                    }
                case .failure(let error):
                    state.lastError = error.localizedDescription
                }
                states[id] = state
            }
        }
    }

    private static func fetchStatus(host: Host) async throws -> HostStatusPayload {
        guard let url = host.statusURL else {
            throw PollerError("Bad URL")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2.5
        if !host.token.isEmpty {
            request.setValue("Bearer \(host.token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw PollerError("HTTP \(code)")
        }
        return try JSONDecoder().decode(HostStatusPayload.self, from: data)
    }

    private func send(_ request: URLRequest) async throws {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw PollerError("No HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let message = object["error"] as? String {
                throw PollerError(message)
            }
            throw PollerError("HTTP \(http.statusCode)")
        }
    }

    private func loadHosts() {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let decoded = try? JSONDecoder().decode([Host].self, from: data) else {
            return
        }
        hosts = decoded
    }

    private func saveHosts() {
        if let data = try? JSONEncoder().encode(hosts) {
            UserDefaults.standard.set(data, forKey: storageKey)
        }
    }
}

private struct PollerError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}
