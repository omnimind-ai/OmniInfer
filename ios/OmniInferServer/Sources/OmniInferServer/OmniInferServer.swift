// OmniInferServer.swift — mirrors OmniInferServer.kt (131 lines).
//
// Public facade for iOS developers.  Usage:
//
//   await OmniInferServer.shared.loadModel(modelPath: "/path/to/model.gguf")
//   // Server is now ready at http://127.0.0.1:9099/v1/chat/completions
//   OmniInferServer.shared.unloadModel()
//   await OmniInferServer.shared.stop()

import Foundation
import llama

/// OmniInfer local inference server facade.
@available(macOS 14, iOS 17, *)
public final class OmniInferServer: @unchecked Sendable {
    public static let shared = OmniInferServer()

    private var _currentHandle: Int64 = 0
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

    /// The current session handle (0 if no model loaded).
    public var currentHandle: Int64 {
        lock.lock()
        defer { lock.unlock() }
        return _currentHandle
    }

    /// Whether a model is loaded and the server is running.
    public var isReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _currentHandle != 0 && serverRunning
    }

    // MARK: - Model lifecycle

    /// Load a model and start the HTTP server.
    /// - Returns: `true` if the model was loaded and the server started.
    @discardableResult
    public func loadModel(
        modelPath: String,
        backend: String = "llama.cpp",
        port: Int = 9099,
        nThreads: Int = 0,
        nCtx: Int = 2048
    ) async -> Bool {
        lock.lock()

        // Unload if switching model or backend.
        if _currentHandle != 0 && (currentModelPath != modelPath || currentBackend != backend) {
            lock.unlock()
            unloadModel()
            lock.lock()
        }

        if _currentHandle != 0 {
            lock.unlock()
            return true
        }
        lock.unlock()

        let handle = OmniInferBridge.shared.initialize(
            modelPath: modelPath,
            backend: backend,
            nThreads: nThreads,
            nCtx: nCtx
        )

        guard handle != 0 else { return false }

        lock.lock()
        _currentHandle = handle
        currentBackend = backend
        currentModelPath = modelPath
        _serverPort = port
        lock.unlock()

        // Start HTTP server if not already running.
        if !serverRunning {
            let service = OmniInferService(
                port: port,
                getHandle: { [weak self] in self?.currentHandle ?? 0 },
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
        let handle = _currentHandle
        _currentHandle = 0
        currentBackend = ""
        currentModelPath = ""
        lock.unlock()

        if handle != 0 {
            OmniInferBridge.shared.free(handle: handle)
        }
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
        guard _currentHandle != 0 else { return [] }
        let name = (currentModelPath as NSString).lastPathComponent
        return [name]
    }

    public func getDiagnostics() -> [String: String] {
        lock.lock()
        let handle = _currentHandle
        lock.unlock()
        guard handle != 0 else { return [:] }
        return OmniInferBridge.shared.collectDiagnostics(handle: handle)
    }
}
