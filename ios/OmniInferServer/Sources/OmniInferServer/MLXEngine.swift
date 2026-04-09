import Foundation
import MLXLLMCommon
import MLXLLM
import Tokenizers

/// InferenceEngine backed by MLX via mlx-swift-lm.
@available(macOS 14, iOS 17, *)
public final class MLXEngine: InferenceEngine, @unchecked Sendable {
    private var modelContainer: ModelContainer?
    private let modelPath: String

    public init?(modelPath: String) {
        self.modelPath = modelPath
        // Model loading is deferred to first generate or explicit load.
    }

    /// Load the model synchronously (blocking).
    public func loadModel() async -> Bool {
        do {
            let url = URL(fileURLWithPath: modelPath)
            let configuration = ModelConfiguration(directory: url)
            let container = try await ModelContainer.load(configuration: configuration)
            self.modelContainer = container
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
        guard let container = modelContainer else {
            NSLog("[OmniInfer/MLX] generate called without loaded model")
            return ""
        }

        // Build chat messages.
        let chatMessages: [[String: String]] = messages

        // Use a semaphore to bridge async → sync for the protocol.
        let semaphore = DispatchSemaphore(value: 0)
        var result = ""

        Task {
            do {
                let output = try await container.perform { (model, tokenizer) in
                    // Apply chat template.
                    let prompt = tokenizer.applyChatTemplate(
                        messages: chatMessages
                    )

                    // Tokenize.
                    let tokens = tokenizer.encode(text: prompt)

                    // Generate with streaming.
                    var fullText = ""
                    let generateParameters = GenerateParameters(temperature: 0.7)

                    for try await generation in try MLXLLMCommon.generate(
                        input: .init(text: LMInput.Text(tokens: .init(tokens))),
                        parameters: generateParameters,
                        model: model
                    ) {
                        let token = generation.token
                        let piece = tokenizer.decode(tokens: [token])
                        fullText += piece
                        onToken(piece)
                    }
                    return fullText
                }
                result = output
            } catch {
                NSLog("[OmniInfer/MLX] generate error: \(error)")
            }
            semaphore.signal()
        }

        semaphore.wait()
        return result
    }

    public func reset() {
        // MLX is stateless per generate call — no KV cache to clear.
    }

    public func free() {
        modelContainer = nil
    }
}
