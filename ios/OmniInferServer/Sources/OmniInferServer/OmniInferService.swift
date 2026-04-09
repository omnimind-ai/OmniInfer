// OmniInferService.swift — mirrors OmniInferService.kt (279 lines).
//
// Embedded Hummingbird HTTP server running on 127.0.0.1:PORT.
// Exposes an OpenAI-compatible API identical to the Android Ktor server.

import Foundation
import Hummingbird
import llama

/// In-process HTTP server providing OpenAI-compatible local inference API.
@available(macOS 14, iOS 17, *)
public final class OmniInferService: Sendable {

    private let port: Int
    private let getHandle: @Sendable () -> Int64
    private let getLoadedModels: @Sendable () -> [String]

    public init(
        port: Int,
        getHandle: @escaping @Sendable () -> Int64,
        getLoadedModels: @escaping @Sendable () -> [String]
    ) {
        self.port = port
        self.getHandle = getHandle
        self.getLoadedModels = getLoadedModels
    }

    /// Start the HTTP server.  Returns when the server has shut down.
    public func run() async throws {
        let router = Router()
        configureRoutes(router)
        let app = Application(
            router: router,
            configuration: .init(address: .hostname("127.0.0.1", port: port))
        )
        try await app.run()
    }

    // MARK: - Routes

    private func configureRoutes(_ router: Router<BasicRequestContext>) {
        router.get("/health") { _, _ in
            return "{\"status\":\"ok\"}"
        }

        router.get("/v1/models") { [getLoadedModels] _, _ in
            let models = getLoadedModels()
            let dataArray = models.map { "{\"id\":\"\($0)\",\"object\":\"model\"}" }
                .joined(separator: ",")
            return "{\"object\":\"list\",\"data\":[\(dataArray)]}"
        }

        router.post("/v1/chat/completions") { [self] request, context in
            return try await handleChatCompletion(request, context: context)
        }
    }

    // MARK: - Chat completion handler

    private func handleChatCompletion(
        _ request: Request,
        context: BasicRequestContext
    ) async throws -> Response {
        let body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        guard let bodyData = body.getData(at: 0, length: body.readableBytes),
              let json = try? JSONSerialization.jsonObject(with: bodyData) as? [String: Any],
              let messages = json["messages"] as? [[String: Any]] else {
            return errorResponse(message: "missing messages", status: .badRequest)
        }

        let stream = json["stream"] as? Bool ?? false
        let enableThinking = json["enable_thinking"] as? Bool ?? false

        // Flatten messages to simple role/content pairs for the C++ backend.
        var flatMessages: [[String: String]] = []
        var lastUserPrompt = ""
        for msg in messages {
            guard let role = msg["role"] as? String else { continue }
            guard let content = extractTextContent(msg["content"]) else { continue }
            flatMessages.append(["role": role, "content": content])
            if role == "user" { lastUserPrompt = content }
        }

        guard !lastUserPrompt.isEmpty else {
            return errorResponse(message: "no user message", status: .badRequest)
        }

        let handle = getHandle()
        guard handle != 0 else {
            return errorResponse(message: "no model loaded", status: .serviceUnavailable)
        }

        // Serialize messages array for the C++ backend.
        let messagesJSON = (try? JSONSerialization.data(withJSONObject: flatMessages))
            .flatMap { String(data: $0, encoding: .utf8) }

        if stream {
            return streamingResponse(handle: handle, messagesJSON: messagesJSON, userPrompt: lastUserPrompt, enableThinking: enableThinking)
        } else {
            return nonStreamingResponse(handle: handle, messagesJSON: messagesJSON, userPrompt: lastUserPrompt, enableThinking: enableThinking)
        }
    }

    // MARK: - Non-streaming response

    private func nonStreamingResponse(handle: Int64, messagesJSON: String?, userPrompt: String, enableThinking: Bool) -> Response {
        var metricsStr: String?
        let result = OmniInferBridge.shared.generate(
            handle: handle, systemPrompt: nil, prompt: userPrompt,
            thinkingEnabled: enableThinking, messagesJSON: messagesJSON,
            onToken: { _ in }, onMetrics: { metricsStr = $0 }
        )

        var resp: [String: Any] = [
            "object": "chat.completion",
            "choices": [[
                "message": ["role": "assistant", "content": result],
                "index": 0, "finish_reason": "stop",
            ] as [String: Any]],
        ]
        if let usage = buildUsageObject(handle: handle, metricsStr: metricsStr) {
            resp["usage"] = usage
        }

        let data = (try? JSONSerialization.data(withJSONObject: resp)) ?? Data()
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(data: data))
        )
    }

    // MARK: - Streaming response (SSE)

    private func streamingResponse(handle: Int64, messagesJSON: String?, userPrompt: String, enableThinking: Bool) -> Response {
        let (stream, continuation) = AsyncStream<String>.makeStream()

        Task.detached { [self] in
            var metricsStr: String?
            _ = OmniInferBridge.shared.generate(
                handle: handle, systemPrompt: nil, prompt: userPrompt,
                thinkingEnabled: enableThinking, messagesJSON: messagesJSON,
                onToken: { token in
                    let chunk: [String: Any] = [
                        "object": "chat.completion.chunk",
                        "choices": [["delta": ["content": token], "index": 0] as [String: Any]],
                    ]
                    if let data = try? JSONSerialization.data(withJSONObject: chunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield("data: \(json)\n\n")
                    }
                },
                onMetrics: { metricsStr = $0 }
            )
            let finalChunk = self.buildUsageChunk(handle: handle, metricsStr: metricsStr)
            if let data = try? JSONSerialization.data(withJSONObject: finalChunk),
               let json = String(data: data, encoding: .utf8) {
                continuation.yield("data: \(json)\n\n")
            }
            continuation.yield("data: [DONE]\n\n")
            continuation.finish()
        }

        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .init("Cache-Control")!: "no-cache"],
            body: .init(asyncSequence: stream.map { .init(string: $0) })
        )
    }

    // MARK: - Helpers

    private func parseMetrics(_ raw: String?) -> [String: Double] {
        guard let raw, !raw.isEmpty else { return [:] }
        var result: [String: Double] = [:]
        for part in raw.split(separator: ",") {
            let kv = part.split(separator: "=", maxSplits: 1)
            if kv.count == 2, let val = Double(kv[1].trimmingCharacters(in: .whitespaces)) {
                result[String(kv[0]).trimmingCharacters(in: .whitespaces)] = val
            }
        }
        return result
    }

    private func buildUsageObject(handle: Int64, metricsStr: String?) -> [String: Any]? {
        let diag = OmniInferBridge.shared.collectDiagnostics(handle: handle)
        let metrics = parseMetrics(metricsStr)
        let promptTokens = Int(diag["prompt_tokens"] ?? "0") ?? 0
        let completionTokens = Int(diag["generated_tokens"] ?? "0") ?? 0
        guard promptTokens > 0 || completionTokens > 0 else { return nil }
        var obj: [String: Any] = [
            "prompt_tokens": promptTokens, "completion_tokens": completionTokens,
            "total_tokens": promptTokens + completionTokens,
        ]
        if let v = metrics["prefill_tps"] { obj["prefill_tokens_per_second"] = v }
        if let v = metrics["decode_tps"] { obj["decode_tokens_per_second"] = v }
        return obj
    }

    private func buildUsageChunk(handle: Int64, metricsStr: String?) -> [String: Any] {
        var chunk: [String: Any] = [
            "object": "chat.completion.chunk",
            "choices": [["delta": [:] as [String: Any], "index": 0, "finish_reason": "stop"] as [String: Any]],
        ]
        if let usage = buildUsageObject(handle: handle, metricsStr: metricsStr) { chunk["usage"] = usage }
        return chunk
    }

    private func extractTextContent(_ content: Any?) -> String? {
        if let str = content as? String { return str.isEmpty ? nil : str }
        if let parts = content as? [[String: Any]] {
            let text = parts.filter { ($0["type"] as? String) == "text" }
                .compactMap { $0["text"] as? String }.joined()
            return text.isEmpty ? nil : text
        }
        return nil
    }

    private func errorResponse(message: String, status: HTTPResponse.Status) -> Response {
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(string: "{\"error\":\"\(message)\"}"))
        )
    }
}
