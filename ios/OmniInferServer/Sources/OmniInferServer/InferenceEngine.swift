import Foundation

/// Metrics from an inference run.
public struct EngineMetrics: Sendable {
    public var promptTokens: Int = 0
    public var generatedTokens: Int = 0
    public var prefillSeconds: Double = 0
    public var decodeSeconds: Double = 0

    public var prefillTPS: Double {
        prefillSeconds > 0 ? Double(promptTokens) / prefillSeconds : 0
    }
    public var decodeTPS: Double {
        decodeSeconds > 0 ? Double(generatedTokens) / decodeSeconds : 0
    }

    public var metricsString: String {
        "prefill_tps=\(prefillTPS), decode_tps=\(decodeTPS)"
    }
}

/// Unified inference engine protocol.
/// Both llama.cpp (C bridge) and MLX (Swift) implement this.
public protocol InferenceEngine: AnyObject, Sendable {
    /// Generate a response from a list of messages.
    /// - Parameters:
    ///   - messages: Array of `[role, content]` pairs.
    ///   - thinkingEnabled: Whether to enable thinking mode in the chat template.
    ///   - onToken: Called for each generated token (return value unused).
    ///   - onMetrics: Called once after generation with metrics string.
    /// - Returns: The complete response string.
    func generate(
        messages: [[String: String]],
        thinkingEnabled: Bool,
        onToken: @escaping @Sendable (String) -> Void,
        onMetrics: @escaping @Sendable (String) -> Void
    ) -> String

    /// Reset conversation state.
    func reset()

    /// Free all resources (model, context, etc.).
    func free()
}
