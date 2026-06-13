import Foundation

struct GPUStatusPayload: Codable, Identifiable {
    let index: Int
    let name: String?
    let vramTempC: Int?
    let fanPct: Int?
    let numFans: Int?
    let powerW: Double?
    let gpuUtilPct: Int?
    let curve: [[Double]]?
    let powerLimitW: Double?
    let powerLimitMinW: Double?
    let powerLimitMaxW: Double?
    let powerLimitDefaultW: Double?
    let powerLimitSupported: Bool?

    var id: Int { index }

    enum CodingKeys: String, CodingKey {
        case index
        case name
        case vramTempC = "vram_temp_c"
        case fanPct = "fan_pct"
        case numFans = "num_fans"
        case powerW = "power_w"
        case gpuUtilPct = "gpu_util_pct"
        case curve
        case powerLimitW = "power_limit_w"
        case powerLimitMinW = "power_limit_min_w"
        case powerLimitMaxW = "power_limit_max_w"
        case powerLimitDefaultW = "power_limit_default_w"
        case powerLimitSupported = "power_limit_supported"
    }
}

/// Mirrors the JSON returned by GET /status on the Linux fan_control.py.
struct HostStatusPayload: Codable {
    let vramTempC: Int?
    let powerW: Double?
    let powerLimitW: Double?
    let powerLimitMinW: Double?
    let powerLimitMaxW: Double?
    let powerLimitDefaultW: Double?
    let tdpW: Double?
    let powerLimitSupported: Bool?
    let gpuUtilPct: Int?
    let fanPct: Int?
    let gpuName: String?
    let numFans: Int?
    let gpus: [GPUStatusPayload]?
    let curve: [[Double]]?
    let updatedAt: Double?
    let wallTime: Double?
    let dryRun: Bool?

    enum CodingKeys: String, CodingKey {
        case vramTempC = "vram_temp_c"
        case powerW = "power_w"
        case powerLimitW = "power_limit_w"
        case powerLimitMinW = "power_limit_min_w"
        case powerLimitMaxW = "power_limit_max_w"
        case powerLimitDefaultW = "power_limit_default_w"
        case tdpW = "tdp_w"
        case powerLimitSupported = "power_limit_supported"
        case gpuUtilPct = "gpu_util_pct"
        case fanPct = "fan_pct"
        case gpuName = "gpu_name"
        case numFans = "num_fans"
        case gpus
        case curve
        case updatedAt = "updated_at"
        case wallTime = "wall_time"
        case dryRun = "dry_run"
    }

    var displayGPUs: [GPUStatusPayload] {
        if let gpus, !gpus.isEmpty {
            return gpus.sorted { $0.index < $1.index }
        }
        guard vramTempC != nil || gpuName != nil else { return [] }
        return [
            GPUStatusPayload(
                index: 0,
                name: gpuName,
                vramTempC: vramTempC,
                fanPct: fanPct,
                numFans: numFans,
                powerW: powerW,
                gpuUtilPct: gpuUtilPct,
                curve: curve,
                powerLimitW: powerLimitW,
                powerLimitMinW: powerLimitMinW,
                powerLimitMaxW: powerLimitMaxW,
                powerLimitDefaultW: powerLimitDefaultW,
                powerLimitSupported: powerLimitSupported
            )
        ]
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
