import Foundation

/// Mirrors the JSON returned by GET /status on the Linux fan_control.py.
struct HostStatusPayload: Codable {
    let vramTempC: Int?
    let powerW: Double?
    let gpuUtilPct: Int?
    let fanPct: Int?
    let gpuName: String?
    let numFans: Int?
    let curve: [[Double]]?
    let updatedAt: Double?
    let wallTime: Double?
    let dryRun: Bool?

    enum CodingKeys: String, CodingKey {
        case vramTempC = "vram_temp_c"
        case powerW = "power_w"
        case gpuUtilPct = "gpu_util_pct"
        case fanPct = "fan_pct"
        case gpuName = "gpu_name"
        case numFans = "num_fans"
        case curve
        case updatedAt = "updated_at"
        case wallTime = "wall_time"
        case dryRun = "dry_run"
    }
}

/// One row in the user's host list.
struct Host: Identifiable, Codable, Hashable {
    let id: UUID
    var name: String          // friendly label, e.g. "aipc1"
    var hostname: String      // tailnet hostname or IP
    var port: Int
    var token: String         // empty string = no token

    init(id: UUID = UUID(), name: String, hostname: String, port: Int = 8765, token: String = "") {
        self.id = id
        self.name = name
        self.hostname = hostname
        self.port = port
        self.token = token
    }

    var statusURL: URL? {
        URL(string: "http://\(hostname):\(port)/status")
    }
}

/// Live per-host state held by the poller.
struct HostState {
    var lastPayload: HostStatusPayload?
    var lastFetchedAt: Date?
    var lastError: String?

    /// Most-recent-last GPU util samples, capped at 60 (≈ last 60s at 1 Hz).
    var utilHistory: [Int] = []

    var isStale: Bool {
        guard let t = lastFetchedAt else { return true }
        return Date().timeIntervalSince(t) > 5.0
    }
}
