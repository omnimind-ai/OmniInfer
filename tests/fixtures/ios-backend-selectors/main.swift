for (legacy, selector) in [("llama.cpp", "llama.cpp-metal"), ("mlx", "mlx-metal")] {
    precondition(BackendSelector(selector: legacy)?.rawValue == legacy)
    precondition(BackendSelector(selector: selector)?.rawValue == legacy)
    precondition(BackendSelector(selector: legacy) == BackendSelector(selector: selector))
}
for unsupported in ["llama.cpp-cuda", "llama.cpp-linux", "unknown", ""] {
    precondition(BackendSelector(selector: unsupported) == nil)
}
print("Swift backend selector contracts passed")
