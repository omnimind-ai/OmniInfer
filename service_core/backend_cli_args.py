from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from service_core.backends import BackendSpec


@dataclass
class ParsedBackendExtraArgs:
    ctx_size: int | None = None
    message: str | None = None
    image: str | None = None
    launch_args: list[str] = field(default_factory=list)
    request_overrides: dict[str, Any] = field(default_factory=dict)


def parse_backend_load_extra_args(backend: BackendSpec, tokens: list[str]) -> ParsedBackendExtraArgs:
    if not tokens:
        return ParsedBackendExtraArgs()
    if backend.family in {"llama.cpp", "turboquant"}:
        return _parse_llama_cpp_load_args(tokens)
    if backend.family == "mlx-lm":
        return _parse_mlx_load_args(tokens)
    return _parse_generic_load_args(tokens)


def parse_backend_chat_extra_args(backend: BackendSpec, tokens: list[str]) -> ParsedBackendExtraArgs:
    if not tokens:
        return ParsedBackendExtraArgs()
    if backend.family in {"llama.cpp", "turboquant"}:
        return _parse_llama_cpp_chat_args(tokens)
    if backend.family == "mlx-lm":
        return _parse_mlx_chat_args(tokens)
    return _parse_generic_chat_args(tokens)


def _parse_llama_cpp_load_args(tokens: list[str]) -> ParsedBackendExtraArgs:
    parsed = ParsedBackendExtraArgs()
    reserved = {"-m", "--model", "-mm", "--mmproj", "--message", "-p", "--prompt", "--image"}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        flag, inline_value = _split_flag_value(token)
        if flag in reserved:
            raise ValueError(f"{flag} is a basic OmniInfer parameter and should be passed with the main CLI options")
        if flag in {"-c", "--ctx-size"}:
            parsed.ctx_size = _parse_int(_take_option_value(tokens, i, inline_value, flag))
            i += 1 if inline_value is not None else 2
            continue
        parsed.launch_args.append(token)
        i += 1
    return parsed


def _parse_mlx_load_args(tokens: list[str]) -> ParsedBackendExtraArgs:
    raise ValueError(
        "mlx-mac does not currently expose backend-native load extra args in OmniInfer; "
        "use the stable CLI options only"
    )


def _parse_generic_load_args(tokens: list[str]) -> ParsedBackendExtraArgs:
    return ParsedBackendExtraArgs(launch_args=list(tokens))


def _parse_llama_cpp_chat_args(tokens: list[str]) -> ParsedBackendExtraArgs:
    parsed = ParsedBackendExtraArgs()
    reserved = {"-m", "--model", "-mm", "--mmproj"}
    aliases: dict[str, tuple[str, str]] = {
        "-n": ("max_tokens", "int"),
        "--n-predict": ("max_tokens", "int"),
        "--max-tokens": ("max_tokens", "int"),
        "--temp": ("temperature", "float"),
        "--temperature": ("temperature", "float"),
        "--top-k": ("top_k", "int"),
        "--top-p": ("top_p", "float"),
        "--min-p": ("min_p", "float"),
        "--typical-p": ("typical_p", "float"),
        "--seed": ("seed", "int"),
        "--repeat-penalty": ("repeat_penalty", "float"),
        "--repeat-last-n": ("repeat_last_n", "int"),
        "--presence-penalty": ("presence_penalty", "float"),
        "--frequency-penalty": ("frequency_penalty", "float"),
        "--min-keep": ("min_keep", "int"),
        "--mirostat": ("mirostat", "int"),
        "--mirostat-tau": ("mirostat_tau", "float"),
        "--mirostat-ent": ("mirostat_tau", "float"),
        "--mirostat-eta": ("mirostat_eta", "float"),
        "--mirostat-lr": ("mirostat_eta", "float"),
        "--dynatemp-range": ("dynatemp_range", "float"),
        "--dynatemp-exp": ("dynatemp_exponent", "float"),
        "--json-schema": ("json_schema", "str"),
        "--grammar": ("grammar", "str"),
        "--samplers": ("samplers", "str"),
        "--cache-prompt": ("cache_prompt", "bool"),
        "--ignore-eos": ("ignore_eos", "bool"),
        "-e": ("ignore_eos", "bool"),
        "--stream": ("stream", "bool"),
        "--think": ("think", "bool"),
    }
    list_aliases = {
        "--stop": "stop",
        "--dry-sequence-breaker": "dry_sequence_breaker",
    }
    i = 0
    while i < len(tokens):
        token = tokens[i]
        flag, inline_value = _split_flag_value(token)
        if flag in reserved:
            raise ValueError(f"{flag} is a basic OmniInfer parameter and should be passed with the main CLI options")
        if flag in {"-p", "--prompt", "--message"}:
            parsed.message = _take_option_value(tokens, i, inline_value, flag)
            i += 1 if inline_value is not None else 2
            continue
        if flag == "--image":
            parsed.image = _take_option_value(tokens, i, inline_value, flag)
            i += 1 if inline_value is not None else 2
            continue
        if flag in {"-c", "--ctx-size"}:
            parsed.ctx_size = _parse_int(_take_option_value(tokens, i, inline_value, flag))
            i += 1 if inline_value is not None else 2
            continue
        if flag in aliases:
            key, value_type = aliases[flag]
            if value_type == "bool":
                value = _take_bool_or_true(tokens, i, inline_value)
                i += 1 if inline_value is not None else (2 if _next_is_value(tokens, i) else 1)
            else:
                value = _coerce_value(_take_option_value(tokens, i, inline_value, flag), value_type)
                i += 1 if inline_value is not None else 2
            parsed.request_overrides[key] = value
            continue
        if flag in list_aliases:
            key = list_aliases[flag]
            value = _take_option_value(tokens, i, inline_value, flag)
            existing = parsed.request_overrides.get(key)
            if not isinstance(existing, list):
                existing = [] if existing is None else [existing]
            existing.append(value)
            parsed.request_overrides[key] = existing
            i += 1 if inline_value is not None else 2
            continue
        if flag.startswith("--no-"):
            parsed.request_overrides[flag[5:].replace("-", "_")] = False
            i += 1
            continue
        if flag.startswith("--"):
            key = flag[2:].replace("-", "_")
            if key in {"messages", "model", "backend", "mmproj", "stream_options"}:
                raise ValueError(f"{flag} is managed by OmniInfer and should not be passed as a backend-native extra arg")
            if inline_value is not None:
                parsed.request_overrides[key] = _auto_scalar(inline_value)
                i += 1
                continue
            if _next_is_value(tokens, i):
                parsed.request_overrides[key] = _auto_scalar(tokens[i + 1])
                i += 2
                continue
            parsed.request_overrides[key] = True
            i += 1
            continue
        raise ValueError(f"unsupported llama.cpp chat extra arg: {flag}")
    return parsed


def _parse_mlx_chat_args(tokens: list[str]) -> ParsedBackendExtraArgs:
    parsed = ParsedBackendExtraArgs()
    aliases: dict[str, tuple[str, str]] = {
        "-n": ("max_tokens", "int"),
        "--max-tokens": ("max_tokens", "int"),
        "--temp": ("temperature", "float"),
        "--temperature": ("temperature", "float"),
        "--top-k": ("top_k", "int"),
        "--top-p": ("top_p", "float"),
        "--min-p": ("min_p", "float"),
        "--seed": ("seed", "int"),
        "--stream": ("stream", "bool"),
        "--think": ("think", "bool"),
    }
    list_aliases = {"--stop": "stop"}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        flag, inline_value = _split_flag_value(token)
        if flag in {"-p", "--prompt", "--message"}:
            parsed.message = _take_option_value(tokens, i, inline_value, flag)
            i += 1 if inline_value is not None else 2
            continue
        if flag == "--image":
            parsed.image = _take_option_value(tokens, i, inline_value, flag)
            i += 1 if inline_value is not None else 2
            continue
        if flag in list_aliases:
            key = list_aliases[flag]
            value = _take_option_value(tokens, i, inline_value, flag)
            existing = parsed.request_overrides.get(key)
            if not isinstance(existing, list):
                existing = [] if existing is None else [existing]
            existing.append(value)
            parsed.request_overrides[key] = existing
            i += 1 if inline_value is not None else 2
            continue
        if flag in aliases:
            key, value_type = aliases[flag]
            if value_type == "bool":
                value = _take_bool_or_true(tokens, i, inline_value)
                i += 1 if inline_value is not None else (2 if _next_is_value(tokens, i) else 1)
            else:
                value = _coerce_value(_take_option_value(tokens, i, inline_value, flag), value_type)
                i += 1 if inline_value is not None else 2
            parsed.request_overrides[key] = value
            continue
        if flag.startswith("--no-"):
            parsed.request_overrides[flag[5:].replace("-", "_")] = False
            i += 1
            continue
        if flag.startswith("--"):
            key = flag[2:].replace("-", "_")
            if key in {"messages", "model", "backend", "mmproj", "stream_options"}:
                raise ValueError(f"{flag} is managed by OmniInfer and should not be passed as a backend-native extra arg")
            if inline_value is not None:
                parsed.request_overrides[key] = _auto_scalar(inline_value)
                i += 1
                continue
            if _next_is_value(tokens, i):
                parsed.request_overrides[key] = _auto_scalar(tokens[i + 1])
                i += 2
                continue
            parsed.request_overrides[key] = True
            i += 1
            continue
        raise ValueError(f"unsupported mlx-mac chat extra arg: {flag}")
    return parsed


def _parse_generic_chat_args(tokens: list[str]) -> ParsedBackendExtraArgs:
    parsed = ParsedBackendExtraArgs()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        flag, inline_value = _split_flag_value(token)
        if flag in {"-p", "--prompt", "--message"}:
            parsed.message = _take_option_value(tokens, i, inline_value, flag)
            i += 1 if inline_value is not None else 2
            continue
        if flag == "--image":
            parsed.image = _take_option_value(tokens, i, inline_value, flag)
            i += 1 if inline_value is not None else 2
            continue
        if flag in {"-c", "--ctx-size"}:
            parsed.ctx_size = _parse_int(_take_option_value(tokens, i, inline_value, flag))
            i += 1 if inline_value is not None else 2
            continue
        if flag.startswith("--no-"):
            parsed.request_overrides[flag[5:].replace("-", "_")] = False
            i += 1
            continue
        if flag.startswith("--"):
            key = flag[2:].replace("-", "_")
            if inline_value is not None:
                parsed.request_overrides[key] = _auto_scalar(inline_value)
                i += 1
                continue
            if _next_is_value(tokens, i):
                parsed.request_overrides[key] = _auto_scalar(tokens[i + 1])
                i += 2
                continue
            parsed.request_overrides[key] = True
            i += 1
            continue
        raise ValueError(f"unsupported backend-native chat extra arg: {flag}")
    return parsed


def _split_flag_value(token: str) -> tuple[str, str | None]:
    if token.startswith("--") and "=" in token:
        flag, value = token.split("=", 1)
        return flag, value
    return token, None


def _take_option_value(tokens: list[str], index: int, inline_value: str | None, flag: str) -> str:
    if inline_value is not None:
        return inline_value
    if index + 1 >= len(tokens):
        raise ValueError(f"{flag} requires a value")
    return tokens[index + 1]


def _next_is_value(tokens: list[str], index: int) -> bool:
    if index + 1 >= len(tokens):
        return False
    candidate = tokens[index + 1]
    return not candidate.startswith("-")


def _take_bool_or_true(tokens: list[str], index: int, inline_value: str | None) -> bool:
    if inline_value is not None:
        return _parse_bool(inline_value)
    if _next_is_value(tokens, index):
        return _parse_bool(tokens[index + 1])
    return True


def _parse_bool(value: str) -> bool:
    low = value.strip().lower()
    if low in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if low in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer value: {value!r}") from exc


def _coerce_value(value: str, value_type: str) -> Any:
    if value_type == "int":
        return _parse_int(value)
    if value_type == "float":
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid float value: {value!r}") from exc
    if value_type == "bool":
        return _parse_bool(value)
    return value


def _auto_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return text
    low = text.lower()
    if low in {"true", "false", "yes", "no", "on", "off"}:
        return _parse_bool(text)
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text
