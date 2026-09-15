//! Public selectors are aliases, never replacements for persisted runtime IDs.
use serde_json::Value;
use thiserror::Error;

pub fn selector(id: &str) -> &str {
    match id {
        "llama.cpp-linux" => "llama.cpp-cpu",
        "llama.cpp-linux-cuda" => "llama.cpp-cuda",
        "llama.cpp-linux-rocm" => "llama.cpp-rocm",
        "llama.cpp-linux-vulkan" => "llama.cpp-vulkan",
        "llama.cpp-linux-s390x" => "llama.cpp-cpu-s390x",
        "llama.cpp-linux-openvino" => "llama.cpp-openvino",
        "llama.cpp-windows-arm64" => "llama.cpp-cpu-arm64",
        "llama.cpp-mac" | "llama.cpp-ios" => "llama.cpp-metal",
        "llama.cpp-mac-intel" => "llama.cpp-cpu",
        "ik_llama.cpp-linux" => "ik_llama.cpp-cpu",
        "ik_llama.cpp-linux-cuda" => "ik_llama.cpp-cuda",
        "stable-diffusion.cpp-linux-vulkan" => "stable-diffusion.cpp-vulkan",
        "omniinfer-native-linux" => "omniinfer-native-eagle3",
        "mnn-linux" => "mnn-cpu",
        "vllm-linux-cuda" => "vllm-cuda",
        "freetoken-linux-cuda" => "freetoken-cuda",
        "vla.cpp-linux" => "vla.cpp-cpu",
        "vla.cpp-linux-cuda" => "vla.cpp-cuda",
        "turboquant-mac" => "turboquant-metal",
        "mlx-mac" | "mlx-ios" => "mlx-metal",
        // WSL2 is an explicit execution environment; Android standalone placement
        // is configurable, unlike the AAR's CPU/HTP selectors. Keep both explicit.
        _ => id,
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ResolveError {
    #[error("Unsupported backend: {name}. Available selectors: {available}")]
    Unknown { name: String, available: String },
    #[error("Backend {0} is incompatible with this host architecture")]
    Architecture(String),
    #[error("Ambiguous backend {name}; use an explicit runtime ID: {candidates}")]
    Ambiguous { name: String, candidates: String },
}

/// Exact IDs take precedence. No prefix stripping, hardware probing, or fallback
/// to a different engine/accelerator is permitted during name resolution.
pub fn resolve_id<'a>(
    name: &str,
    entries: impl IntoIterator<Item = (&'a str, &'a str, bool)>,
) -> Result<&'a str, ResolveError> {
    let entries = entries.into_iter().collect::<Vec<_>>();
    if let Some((id, _, compatible)) = entries.iter().find(|(id, _, _)| *id == name) {
        return if *compatible {
            Ok(id)
        } else {
            Err(ResolveError::Architecture(name.into()))
        };
    }
    let candidates = entries
        .iter()
        .filter(|(_, alias, compatible)| *alias == name && *compatible)
        .map(|(id, _, _)| *id)
        .collect::<Vec<_>>();
    match candidates.as_slice() {
        [id] => Ok(id),
        [] => Err(ResolveError::Unknown {
            name: name.into(),
            available: entries
                .iter()
                .filter(|(_, _, compatible)| *compatible)
                .map(|(_, alias, _)| *alias)
                .collect::<Vec<_>>()
                .join(", "),
        }),
        _ => Err(ResolveError::Ambiguous {
            name: name.into(),
            candidates: candidates.join(", "),
        }),
    }
}

pub fn resolve_rows<'a>(rows: &'a [Value], name: &str) -> Result<&'a str, ResolveError> {
    resolve_id(
        name,
        rows.iter().filter_map(|row| {
            let id = row.get("id")?.as_str()?;
            Some((
                id,
                row.get("selector").and_then(Value::as_str).unwrap_or(id),
                row.get("architecture_compatible")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            ))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ambiguous_alias_fails_closed_and_exact_id_wins() {
        let entries = [("one", "shared", true), ("two", "shared", true)];
        assert!(matches!(
            resolve_id("shared", entries),
            Err(ResolveError::Ambiguous { .. })
        ));
        assert_eq!(resolve_id("one", entries).unwrap(), "one");
        assert_eq!(
            resolve_id(
                "shared",
                [("one", "shared", false), ("two", "shared", true)]
            )
            .unwrap(),
            "two"
        );
        assert!(resolve_id("one", [("one", "shared", false)]).is_err());
    }
}
