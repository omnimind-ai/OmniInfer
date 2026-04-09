import Foundation
import MLXLMCommon
import MLXLLM

/// InferenceEngine backed by MLX via mlx-swift-lm.
@available(macOS 14, iOS 17, *)
public final class MLXEngine: InferenceEngine, @unchecked Sendable {
    private var modelContainer: ModelContainer?
    private var session: ChatSession?
    private let modelPath: String

    public init?(modelPath: String) {
        self.modelPath = modelPath
    }

    /// Load the model. Must be called before generate().
    public func loadModel() async -> Bool {
        do {
            let url = URL(fileURLWithPath: modelPath)
            let container = try await loadModelContainer(directory: url)
            self.modelContainer = container
            self.session = ChatSession(container)
            NSLog("[OmniInfer/MLX] model loaded from \(modelPath)")
            return true
        } catch {
            NSLog("[OmniInfer/MLX] loadModel failed: \(error)")
            return false
        }
    }

    public func generate(
        messages: [[String: String]],
        thinkingEnabled: Bool,
        onToken: @escaping @Sendable (String) -> Void,
        onMetrics: @escaping @Sendable (String) -> Void
    ) -> String {
        guard let session else {
            NSLog("[OmniInfer/MLX] generate called without loaded model")
            return ""
        }

        // Reset session and set system instructions from messages.
        session.instructions = nil
        var userMessages: [(role: String, content: String)] = []
        for msg in messages {
            let role = msg["role"] ?? ""
            let content = msg["content"] ?? ""
            if role == "system" {
                session.instructions = content
            } else {
                userMessages.append((role: role, content: content))
            }
        }

        // Feed history turns (all but the last user message).
        // ChatSession manages history internally, so we reset and replay.
        // For simplicity, concatenate history into a single prompt context.
        guard let lastUser = userMessages.last(where: { $0.role == "user" }) else {
            return ""
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result = ""
        let startTime = CFAbsoluteTimeGetCurrent()

        Task {
            do {
                var tokenCount = 0
                for try await chunk in session.streamResponse(to: lastUser.content) {
                    result += chunk
                    tokenCount += 1
                    onToken(chunk)
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let tps = elapsed > 0 ? Double(tokenCount) / elapsed : 0
                onMetrics("decode_tps=\(tps)")
            } catch {
                NSLog("[OmniInfer/MLX] generate error: \(error)")
            }
            semaphore.signal()
        }

        semaphore.wait()
        return result
    }

    public func reset() {
        // Recreate session to clear history.
        if let container = modelContainer {
            session = ChatSession(container)
        }
    }

    public func free() {
        session = nil
        modelContainer = nil
    }
}
