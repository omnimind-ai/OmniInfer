// OmniInferBridge.swift — mirrors OmniInferBridge.kt (91 lines).
//
// Swift-side wrapper for the C bridge API.  Uses Unmanaged to pass
// Swift closures through C function-pointer callbacks.

import Foundation
import llama

/// Swift bridge to the native OmniInfer C inference layer.
public final class OmniInferBridge: @unchecked Sendable {
    public static let shared = OmniInferBridge()
    private init() {}

    // MARK: - Session lifecycle

    /// Initialize a backend and load a model.
    /// - Returns: A session handle (>0) on success, 0 on failure.
    public func initialize(
        modelPath: String,
        backend: String = "llama.cpp",
        nThreads: Int = 0,
        nCtx: Int = 4096,
        nGpuLayers: Int = 99
    ) -> Int64 {
        let config: [String: Any] = [
            "backend": backend,
            "model_path": modelPath,
            "native_lib_dir": "",
            "n_threads": nThreads,
            "n_ctx": nCtx,
            "n_gpu_layers": nGpuLayers,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: config),
              let json = String(data: data, encoding: .utf8) else { return 0 }
        return omniinfer_init(json)
    }

    /// Free a session and release all resources.
    public func free(handle: Int64) {
        omniinfer_free(handle)
    }

    // MARK: - Generation

    /// Run inference, streaming tokens via `onToken`.
    /// - Returns: The complete response string.
    /// Run inference. If `messagesJSON` is provided, it overrides systemPrompt/prompt
    /// and is passed to the C++ backend to build the full conversation.
    public func generate(
        handle: Int64,
        systemPrompt: String?,
        prompt: String,
        thinkingEnabled: Bool = false,
        messagesJSON: String? = nil,
        onToken: @escaping (String) -> Void,
        onMetrics: @escaping (String) -> Void
    ) -> String {
        var reqObj: [String: Any] = ["thinking_enabled": thinkingEnabled]
        if let mj = messagesJSON { reqObj["messages_json"] = mj }
        let requestJSON = (try? JSONSerialization.data(withJSONObject: reqObj))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

        let ctx = CallbackContext(onToken: onToken, onMetrics: onMetrics)
        let ctxPtr = Unmanaged.passRetained(ctx).toOpaque()

        let tokenCb: OmniInferTokenCallback = { token, userdata in
            guard let token, let userdata else { return false }
            let ctx = Unmanaged<CallbackContext>.fromOpaque(userdata)
                .takeUnretainedValue()
            ctx.onToken(String(cString: token))
            return true
        }

        let metricsCb: OmniInferMetricsCallback = { metrics, userdata in
            guard let metrics, let userdata else { return }
            let ctx = Unmanaged<CallbackContext>.fromOpaque(userdata)
                .takeUnretainedValue()
            ctx.onMetrics(String(cString: metrics))
        }

        let resultPtr = omniinfer_generate(
            handle,
            systemPrompt,
            prompt,
            requestJSON,
            tokenCb,
            metricsCb,
            ctxPtr
        )

        // Balance the retain.
        Unmanaged<CallbackContext>.fromOpaque(ctxPtr).release()

        guard let resultPtr else { return "" }
        let result = String(cString: resultPtr)
        omniinfer_free_string(resultPtr)
        return result
    }

    // MARK: - History & control

    public func loadHistory(handle: Int64, messages: [(role: String, content: String)]) -> Bool {
        let count = Int32(messages.count)
        var roles = messages.map { strdup($0.role) }
        var contents = messages.map { strdup($0.content) }
        defer {
            roles.forEach { Darwin.free($0) }
            contents.forEach { Darwin.free($0) }
        }
        return roles.withUnsafeMutableBufferPointer { rolesBuf in
            contents.withUnsafeMutableBufferPointer { contentsBuf in
                let r = rolesBuf.baseAddress!.withMemoryRebound(
                    to: UnsafePointer<CChar>?.self, capacity: messages.count
                ) { rolesPtr in
                    contentsBuf.baseAddress!.withMemoryRebound(
                        to: UnsafePointer<CChar>?.self, capacity: messages.count
                    ) { contentsPtr in
                        omniinfer_load_history(handle, rolesPtr, contentsPtr, count)
                    }
                }
                return r
            }
        }
    }

    public func setThinkMode(handle: Int64, enabled: Bool) {
        omniinfer_set_think_mode(handle, enabled)
    }

    public func reset(handle: Int64) {
        omniinfer_reset(handle)
    }

    public func cancel(handle: Int64) {
        omniinfer_cancel(handle)
    }

    // MARK: - Diagnostics

    public func collectDiagnostics(handle: Int64) -> [String: String] {
        guard let ptr = omniinfer_collect_diagnostics_json(handle) else {
            return [:]
        }
        let json = String(cString: ptr)
        omniinfer_free_string(ptr)
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        return obj.mapValues { "\($0)" }
    }
}

// MARK: - Callback context

private final class CallbackContext {
    let onToken: (String) -> Void
    let onMetrics: (String) -> Void
    init(onToken: @escaping (String) -> Void,
         onMetrics: @escaping (String) -> Void) {
        self.onToken = onToken
        self.onMetrics = onMetrics
    }
}
