import ActivityKit
import Foundation

public struct GPUStatusActivityAttributes: ActivityAttributes {
    public let hostID: String
    public let hostName: String

    public init(hostID: String, hostName: String) {
        self.hostID = hostID
        self.hostName = hostName
    }

    public struct ContentState: Codable, Hashable {
        var vramTempC: Int?
        var fanPct: Int?
        var gpuUtilPct: Int?
        var powerW: Double?
        var updatedAt: Date
    }
}
