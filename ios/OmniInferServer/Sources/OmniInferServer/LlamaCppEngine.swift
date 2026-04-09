import Foundation
import llama

/// InferenceEngine backed by llama.cpp via the C bridge.
@available(macOS 14, iOS 17, *)
public final class LlamaCppEngine: InferenceEngine, @unchecked Sendable {
    private let handle: Int64

    public init?(modelPath: String, backend: String, nThreads: Int, nCtx: Int, nGpuLayers: Int) {
        let h = OmniInferBridge.shared.initialize(
            modelPath: modelPath,
            backend: backend,
            nThreads: nThreads,
            nCtx: nCtx,
            nGpuLayers: nGpuLayers
        )
        guard h != 0 else { return nil }
        self.handle = h
    }

    public func generate(
        messages: [[String: String]],
        thinkingEnabled: Bool,
        onToken: @escaping @Sendable (String) -> Void,
        onMetrics: @escaping @Sendable (String) -> Void
    ) -> String {
        // Serialize messages for the C++ backend.
        let messagesJSON = (try? JSONSerialization.data(withJSONObject: messages))
            .flatMap { String(data: $0, encoding: .utf8) }

        let lastUser = messages.last(where: { $0["role"] == "user" })?["content"] ?? ""
        return OmniInferBridge.shared.generate(
            handle: handle,
            systemPrompt: nil,
            prompt: lastUser,
            thinkingEnabled: thinkingEnabled,
            messagesJSON: messagesJSON,
            onToken: onToken,
            onMetrics: onMetrics
        )
    }

    public func reset() {
        OmniInferBridge.shared.reset(handle: handle)
    }

    public func free() {
        OmniInferBridge.shared.free(handle: handle)
    }
}
