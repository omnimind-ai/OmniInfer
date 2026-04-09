// OmniInferServer.swift — unified facade for iOS developers.
//
// Usage:
//   await OmniInferServer.shared.loadModel(modelPath: "/path/to/model.gguf", backend: "llama.cpp")
//   await OmniInferServer.shared.loadModel(modelPath: "/path/to/mlx-model/", backend: "mlx")
//   // Server is now ready at http://127.0.0.1:9099/v1/chat/completions
//   OmniInferServer.shared.unloadModel()
//   await OmniInferServer.shared.stop()

import Foundation
import llama

/// OmniInfer local inference server facade.
@available(macOS 14, iOS 17, *)
public final class OmniInferServer: @unchecked Sendable {
    public static let shared = OmniInferServer()

    private var currentEngine: (any InferenceEngine)?
    private var currentBackend: String = ""
    private var currentModelPath: String = ""
    private var _serverPort: Int = 9099
    private var serverRunning = false
    private var serverTask: Task<Void, Error>?
    private let lock = NSLock()

    private init() {}

    /// The port the HTTP server is listening on.
    public var port: Int {
        lock.lock()
        defer { lock.unlock() }
        return _serverPort
    }

    /// The current inference engine (nil if no model loaded).
    public var engine: (any InferenceEngine)? {
        lock.lock()
        defer { lock.unlock() }
        return currentEngine
    }

    /// Whether a model is loaded and the server is running.
    public var isReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return currentEngine != nil && serverRunning
    }

    // MARK: - Model lifecycle

    /// Load a model and start the HTTP server.
    /// - Parameters:
    ///   - modelPath: Path to model file (.gguf) or directory (MLX).
    ///   - backend: `"llama.cpp"` or `"mlx"`.
    /// - Returns: `true` if the model was loaded and the server started.
    @discardableResult
    public func loadModel(
        modelPath: String,
        backend: String = "llama.cpp",
        port: Int = 9099,
        nThreads: Int = 0,
        nCtx: Int = 2048,
        nGpuLayers: Int = 99
    ) async -> Bool {
        lock.lock()

        // Unload if switching model or backend.
        if currentEngine != nil && (currentModelPath != modelPath || currentBackend != backend) {
            lock.unlock()
            unloadModel()
            lock.lock()
        }

        if currentEngine != nil {
            lock.unlock()
            return true
        }
        lock.unlock()

        // Create the appropriate engine.
        let engine: (any InferenceEngine)?
        switch backend {
        case "mlx":
            let mlx = MLXEngine(modelPath: modelPath)
            if let mlx, await mlx.loadModel() {
                engine = mlx
            } else {
                engine = nil
            }
        default: // "llama.cpp"
            engine = LlamaCppEngine(
                modelPath: modelPath, backend: backend,
                nThreads: nThreads, nCtx: nCtx, nGpuLayers: nGpuLayers
            )
        }

        guard let engine else { return false }

        lock.lock()
        currentEngine = engine
        currentBackend = backend
        currentModelPath = modelPath
        _serverPort = port
        lock.unlock()

        // Start HTTP server if not already running.
        if !serverRunning {
            let service = OmniInferService(
                port: port,
                getEngine: { [weak self] in self?.engine },
                getLoadedModels: { [weak self] in self?.getLoadedModels() ?? [] }
            )
            serverTask = Task {
                try await service.run()
            }
            lock.lock()
            serverRunning = true
            lock.unlock()
        }

        return true
    }

    /// Unload the current model (server keeps running for the next load).
    public func unloadModel() {
        lock.lock()
        let eng = currentEngine
        currentEngine = nil
        currentBackend = ""
        currentModelPath = ""
        lock.unlock()

        eng?.free()
    }

    /// Unload model and stop the HTTP server.
    public func stop() async {
        unloadModel()
        serverTask?.cancel()
        serverTask = nil
        lock.lock()
        serverRunning = false
        lock.unlock()
    }

    // MARK: - Queries

    public func getLoadedModels() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        guard currentEngine != nil else { return [] }
        let name = (currentModelPath as NSString).lastPathComponent
        return [name]
    }
}
