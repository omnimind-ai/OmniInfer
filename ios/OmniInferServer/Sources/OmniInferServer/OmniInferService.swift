// OmniInferService.swift
//
// Embedded Hummingbird HTTP server on 127.0.0.1:PORT.
// Dispatches to the current InferenceEngine (llama.cpp or MLX).

import Foundation
import Hummingbird

/// In-process HTTP server providing OpenAI-compatible local inference API.
@available(macOS 14, iOS 17, *)
public final class OmniInferService: Sendable {

    private let port: Int
    private let getEngine: @Sendable () -> (any InferenceEngine)?
    private let getLoadedModels: @Sendable () -> [String]

    public init(
        port: Int,
        getEngine: @escaping @Sendable () -> (any InferenceEngine)?,
        getLoadedModels: @escaping @Sendable () -> [String]
    ) {
        self.port = port
        self.getEngine = getEngine
        self.getLoadedModels = getLoadedModels
    }

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
        router.get("/health") { _, _ in "{\"status\":\"ok\"}" }

        router.get("/v1/models") { [getLoadedModels] _, _ in
            let models = getLoadedModels()
            let arr = models.map { "{\"id\":\"\($0)\",\"object\":\"model\"}" }.joined(separator: ",")
            return "{\"object\":\"list\",\"data\":[\(arr)]}"
        }

        router.post("/v1/chat/completions") { [self] request, context in
            return try await handleChatCompletion(request, context: context)
        }
    }

    // MARK: - Chat completion handler

    private func handleChatCompletion(
        _ request: Request, context: BasicRequestContext
    ) async throws -> Response {
        let body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        guard let bodyData = body.getData(at: 0, length: body.readableBytes),
              let json = try? JSONSerialization.jsonObject(with: bodyData) as? [String: Any],
              let messages = json["messages"] as? [[String: Any]] else {
            return errorResponse(message: "missing messages", status: .badRequest)
        }

        let stream = json["stream"] as? Bool ?? false
        let enableThinking = json["enable_thinking"] as? Bool ?? false

        var flatMessages: [[String: String]] = []
        for msg in messages {
            guard let role = msg["role"] as? String,
                  let content = extractTextContent(msg["content"]) else { continue }
            flatMessages.append(["role": role, "content": content])
        }

        guard flatMessages.contains(where: { $0["role"] == "user" }) else {
            return errorResponse(message: "no user message", status: .badRequest)
        }

        guard let engine = getEngine() else {
            return errorResponse(message: "no model loaded", status: .serviceUnavailable)
        }

        if stream {
            return streamingResponse(engine: engine, messages: flatMessages, enableThinking: enableThinking)
        } else {
            return nonStreamingResponse(engine: engine, messages: flatMessages, enableThinking: enableThinking)
        }
    }

    // MARK: - Non-streaming

    private func nonStreamingResponse(
        engine: any InferenceEngine, messages: [[String: String]], enableThinking: Bool
    ) -> Response {
        var metricsStr: String?
        let result = engine.generate(
            messages: messages, thinkingEnabled: enableThinking,
            onToken: { _ in }, onMetrics: { metricsStr = $0 }
        )

        var resp: [String: Any] = [
            "object": "chat.completion",
            "choices": [["message": ["role": "assistant", "content": result],
                         "index": 0, "finish_reason": "stop"] as [String: Any]],
        ]
        if let usage = buildUsage(metricsStr: metricsStr) { resp["usage"] = usage }

        let data = (try? JSONSerialization.data(withJSONObject: resp)) ?? Data()
        return Response(status: .ok, headers: [.contentType: "application/json"],
                        body: .init(byteBuffer: .init(data: data)))
    }

    // MARK: - Streaming (SSE)

    private func streamingResponse(
        engine: any InferenceEngine, messages: [[String: String]], enableThinking: Bool
    ) -> Response {
        let (stream, continuation) = AsyncStream<String>.makeStream()

        Task.detached {
            var metricsStr: String?
            _ = engine.generate(
                messages: messages, thinkingEnabled: enableThinking,
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
            // Final chunk.
            var finalChunk: [String: Any] = [
                "object": "chat.completion.chunk",
                "choices": [["delta": [:] as [String: Any], "index": 0, "finish_reason": "stop"] as [String: Any]],
            ]
            if let usage = Self.buildUsageStatic(metricsStr: metricsStr) { finalChunk["usage"] = usage }
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

    private func buildUsage(metricsStr: String?) -> [String: Any]? {
        Self.buildUsageStatic(metricsStr: metricsStr)
    }

    private static func buildUsageStatic(metricsStr: String?) -> [String: Any]? {
        guard let raw = metricsStr, !raw.isEmpty else { return nil }
        var metrics: [String: Double] = [:]
        for part in raw.split(separator: ",") {
            let kv = part.split(separator: "=", maxSplits: 1)
            if kv.count == 2, let val = Double(kv[1].trimmingCharacters(in: .whitespaces)) {
                metrics[String(kv[0]).trimmingCharacters(in: .whitespaces)] = val
            }
        }
        guard !metrics.isEmpty else { return nil }
        var obj: [String: Any] = [:]
        if let v = metrics["prefill_tps"] { obj["prefill_tokens_per_second"] = v }
        if let v = metrics["decode_tps"] { obj["decode_tokens_per_second"] = v }
        return obj.isEmpty ? nil : obj
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
        Response(status: status, headers: [.contentType: "application/json"],
                 body: .init(byteBuffer: .init(string: "{\"error\":\"\(message)\"}")))
    }
}
