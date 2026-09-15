/// Public selectors map to stable engine identities before comparing or loading runtimes.
internal enum BackendSelector: String {
    case llamaCpp = "llama.cpp"
    case mlx = "mlx"

    init?(selector: String) {
        switch selector {
        case "llama.cpp", "llama.cpp-metal": self = .llamaCpp
        case "mlx", "mlx-metal": self = .mlx
        default: return nil
        }
    }
}
