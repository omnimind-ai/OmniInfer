use serde_json::json;

use super::*;

#[test]
fn builds_llama_cpp_cuda_command() {
    let backend = json!({
        "id": "llama.cpp-linux-cuda",
        "launcher_path": "/runtime/llama.cpp-linux-cuda/bin/llama-server",
        "runtime_dir": "/runtime/llama.cpp-linux-cuda",
        "default_args": ["-ngl", "999"],
        "external_server_protocol": "llama.cpp-server",
        "log_file_name": "runtime.log"
    });
    let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
        backend,
        model_path: "/models/qwen.gguf".to_string(),
        mmproj_path: Some("/models/mmproj.gguf".to_string()),
        host: "127.0.0.1".to_string(),
        port: 12345,
        ctx_size: Some(8192),
        launch_args: None,
    })
    .unwrap();
    let log_dir = PathBuf::from("/runtime/llama.cpp-linux-cuda")
        .join("logs")
        .display()
        .to_string();
    assert_eq!(plan.ctx_size, Some(8192));
    assert_eq!(plan.cwd, PathBuf::from("/runtime/llama.cpp-linux-cuda/bin"));
    assert_eq!(plan.protocol, ExternalServerProtocol::LlamaCppServer);
    assert_eq!(plan.client_endpoint, "http://127.0.0.1:12345");
    assert!(plan.protocol.is_openai_compatible());
    assert_eq!(plan.readiness_probe, RuntimeReadinessProbe::HttpHealth);
    assert_eq!(
        plan.command,
        vec![
            "/runtime/llama.cpp-linux-cuda/bin/llama-server".to_string(),
            "-m".to_string(),
            "/models/qwen.gguf".to_string(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            "12345".to_string(),
            "--no-webui".to_string(),
            "--slot-save-path".to_string(),
            log_dir,
            "-ngl".to_string(),
            "999".to_string(),
            "-c".to_string(),
            "8192".to_string(),
            "--mmproj".to_string(),
            "/models/mmproj.gguf".to_string()
        ]
    );
}

#[test]
fn ik_uses_webui_none() {
    let backend = json!({
        "id": "ik_llama.cpp-linux-cuda",
        "launcher_path": "/runtime/ik/bin/llama-server",
        "runtime_dir": "/runtime/ik",
        "default_args": ["--jinja", "-ngl", "999"],
        "external_server_protocol": "llama.cpp-server"
    });
    let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
        backend,
        model_path: "/models/qwen.gguf".to_string(),
        mmproj_path: None,
        host: "127.0.0.1".to_string(),
        port: 12345,
        ctx_size: None,
        launch_args: None,
    })
    .unwrap();
    assert!(
        plan.command
            .windows(2)
            .any(|items| items == ["--webui", "none"])
    );
    assert!(!plan.command.iter().any(|item| item == "--no-webui"));
}

#[test]
fn builds_vla_cpp_zmq_server_shape() {
    let backend = json!({
        "id": "vla.cpp-linux-cuda",
        "launcher_path": "/runtime/vla.cpp-linux-cuda/bin/vla-server",
        "runtime_dir": "/runtime/vla.cpp-linux-cuda",
        "default_args": ["--timing-detail", "phase"],
        "external_server_protocol": "vla.cpp-zmq-server",
        "log_file_name": "vla-server.log"
    });
    let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
        backend,
        model_path: "/models/smolvla.gguf".to_string(),
        mmproj_path: Some("/models/mmproj.gguf".to_string()),
        host: "127.0.0.1".to_string(),
        port: 15555,
        ctx_size: Some(8192),
        launch_args: None,
    })
    .unwrap();
    assert_eq!(plan.ctx_size, None);
    assert_eq!(plan.protocol, ExternalServerProtocol::VlaCppZmqServer);
    assert_eq!(plan.client_endpoint, "tcp://127.0.0.1:15555");
    assert!(!plan.protocol.is_openai_compatible());
    assert_eq!(
        plan.readiness_probe,
        RuntimeReadinessProbe::TcpConnectAndLog {
            marker: "vla-server: bound to tcp://127.0.0.1:15555. ready.".to_string(),
        }
    );
    assert_eq!(
        plan.command,
        vec![
            "/runtime/vla.cpp-linux-cuda/bin/vla-server".to_string(),
            "--bind".to_string(),
            "tcp://127.0.0.1:15555".to_string(),
            "--timing-detail".to_string(),
            "phase".to_string(),
            "/models/mmproj.gguf".to_string(),
            "/models/smolvla.gguf".to_string(),
        ]
    );
}

#[test]
fn builds_omniinfer_vla_zmq_server_shape() {
    let backend = json!({
        "id": "omniinfer-vla-linux-cuda",
        "launcher_path": "/runtime/omniinfer-vla-linux-cuda/bin/omniinfer-vla-server",
        "runtime_dir": "/runtime/omniinfer-vla-linux-cuda",
        "default_args": ["--arch", "pi05"],
        "external_server_protocol": "omniinfer-vla-zmq-server",
        "log_file_name": "omniinfer-vla-server.log"
    });
    let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
        backend,
        model_path: "/models/pi05_libero_finetuned_v044".to_string(),
        mmproj_path: None,
        host: "127.0.0.1".to_string(),
        port: 15556,
        ctx_size: Some(8192),
        launch_args: None,
    })
    .unwrap();
    assert_eq!(plan.ctx_size, None);
    assert_eq!(plan.protocol, ExternalServerProtocol::OmniInferVlaZmqServer);
    assert_eq!(plan.client_endpoint, "tcp://127.0.0.1:15556");
    assert!(!plan.protocol.is_openai_compatible());
    assert_eq!(
        plan.readiness_probe,
        RuntimeReadinessProbe::TcpConnectAndLog {
            marker: "omniinfer-vla-server: bound to tcp://127.0.0.1:15556. ready.".to_string(),
        }
    );
    assert_eq!(
        plan.command,
        vec![
            "/runtime/omniinfer-vla-linux-cuda/bin/omniinfer-vla-server".to_string(),
            "--bind".to_string(),
            "tcp://127.0.0.1:15556".to_string(),
            "--checkpoint".to_string(),
            "/models/pi05_libero_finetuned_v044".to_string(),
            "--arch".to_string(),
            "pi05".to_string(),
        ]
    );
}

#[test]
fn rejects_vla_cpp_context_launch_arg() {
    let backend = json!({
        "id": "vla.cpp-linux",
        "launcher_path": "/runtime/vla.cpp-linux/bin/vla-server",
        "runtime_dir": "/runtime/vla.cpp-linux",
        "default_args": ["-c", "4096"],
        "external_server_protocol": "vla.cpp-zmq-server"
    });
    let error = build_external_runtime_plan(&ExternalRuntimeRequest {
        backend,
        model_path: "/models/smolvla.gguf".to_string(),
        mmproj_path: None,
        host: "127.0.0.1".to_string(),
        port: 15555,
        ctx_size: None,
        launch_args: None,
    })
    .unwrap_err();
    assert_eq!(error, RuntimePlanError::ReservedLaunchArg("-c".to_string()));
}

#[test]
fn rejects_unauthenticated_non_loopback_vla_bind() {
    let backend = json!({
        "id": "vla.cpp-linux",
        "launcher_path": "/runtime/vla.cpp-linux/bin/vla-server",
        "runtime_dir": "/runtime/vla.cpp-linux",
        "external_server_protocol": "vla.cpp-zmq-server"
    });
    for host in ["0.0.0.0", "192.0.2.10", "::"] {
        let error = build_external_runtime_plan(&ExternalRuntimeRequest {
            backend: backend.clone(),
            model_path: "/models/smolvla.gguf".to_string(),
            mmproj_path: None,
            host: host.to_string(),
            port: 15555,
            ctx_size: None,
            launch_args: None,
        })
        .unwrap_err();
        assert_eq!(
            error,
            RuntimePlanError::NonLoopbackVlaBind(host.to_string())
        );
    }
}

#[test]
fn rejects_non_loopback_omniinfer_vla_bind() {
    let backend = json!({
        "id": "omniinfer-vla-linux-cuda",
        "launcher_path": "/runtime/omniinfer-vla-linux-cuda/bin/omniinfer-vla-server",
        "runtime_dir": "/runtime/omniinfer-vla-linux-cuda",
        "external_server_protocol": "omniinfer-vla-zmq-server"
    });
    let error = build_external_runtime_plan(&ExternalRuntimeRequest {
        backend,
        model_path: "/models/pi05_libero_finetuned_v044".to_string(),
        mmproj_path: None,
        host: "0.0.0.0".to_string(),
        port: 15556,
        ctx_size: None,
        launch_args: None,
    })
    .unwrap_err();
    assert_eq!(
        error,
        RuntimePlanError::NonLoopbackOmniInferVlaBind("0.0.0.0".to_string())
    );
}

#[test]
fn formats_ipv6_loopback_client_endpoint() {
    assert_eq!(
        ExternalServerProtocol::VlaCppZmqServer.client_endpoint("::1", 15555),
        "tcp://[::1]:15555"
    );
}

#[test]
fn vllm_uses_openai_server_shape() {
    let backend = json!({
        "id": "vllm-linux-cuda",
        "launcher_path": "/runtime/vllm/bin/vllm",
        "runtime_dir": "/runtime/vllm",
        "default_args": ["--max-model-len", "4096"],
        "external_server_protocol": "vllm-openai-server",
        "log_file_name": "vllm-server.log"
    });
    let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
        backend,
        model_path: "Qwen/Qwen3.5-4B".to_string(),
        mmproj_path: None,
        host: "127.0.0.1".to_string(),
        port: 23456,
        ctx_size: None,
        launch_args: None,
    })
    .unwrap();
    assert_eq!(plan.ctx_size, Some(4096));
    assert_eq!(plan.proxy_model_ref.as_deref(), Some("local"));
    assert_eq!(plan.protocol, ExternalServerProtocol::VllmOpenAiServer);
    assert_eq!(plan.client_endpoint, "http://127.0.0.1:23456");
    assert!(plan.protocol.is_openai_compatible());
    assert_eq!(
        plan.command,
        vec![
            "/runtime/vllm/bin/vllm",
            "serve",
            "Qwen/Qwen3.5-4B",
            "--host",
            "127.0.0.1",
            "--port",
            "23456",
            "--served-model-name",
            "local",
            "--max-model-len",
            "4096"
        ]
    );
}

#[test]
fn freetoken_uses_managed_openai_server_shape_and_ready_marker() {
    let backend = json!({
        "id": "freetoken-linux-cuda",
        "launcher_path": "/runtime/freetoken/bin/ft",
        "runtime_dir": "/runtime/freetoken",
        "default_args": ["--memory-ratio", "0.8"],
        "external_server_protocol": "freetoken-openai-server",
        "log_file_name": "freetoken-server.log"
    });
    let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
        backend,
        model_path: "Qwen/Qwen3.6-35B-A3B".to_string(),
        mmproj_path: None,
        host: "127.0.0.1".to_string(),
        port: 19190,
        ctx_size: Some(8192),
        launch_args: None,
    })
    .unwrap();
    assert_eq!(plan.ctx_size, Some(8192));
    assert_eq!(plan.proxy_model_ref.as_deref(), Some("local"));
    assert_eq!(plan.protocol, ExternalServerProtocol::FreeTokenOpenAiServer);
    assert_eq!(plan.client_endpoint, "http://127.0.0.1:19190");
    assert!(plan.protocol.is_openai_compatible());
    assert_eq!(
        plan.readiness_probe,
        RuntimeReadinessProbe::TcpConnectAndLog {
            marker: "API server is ready to serve on 127.0.0.1:19190".to_string(),
        }
    );
    assert_eq!(
        plan.command,
        vec![
            "/runtime/freetoken/bin/ft",
            "serve",
            "--model",
            "Qwen/Qwen3.6-35B-A3B",
            "--host",
            "127.0.0.1",
            "--port",
            "19190",
            "--served-model-name",
            "local",
            "--memory-ratio",
            "0.8",
            "--max-seq-len-override",
            "8192",
        ]
    );
}

#[test]
fn wsl_vllm_uses_manifest_and_translates_windows_model_path() {
    let root = std::env::temp_dir().join(format!("omniinfer-wsl-plan-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let manifest = root.join("vllm-wsl2.json");
    std::fs::write(
        &manifest,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "backend": "vllm-wsl2-cuda",
            "distribution": "Ubuntu-24.04",
            "linux_launcher": "/home/test/runtime/current/venv/bin/vllm",
            "linux_runner": "/home/test/runtime/current/bin/omniinfer-vllm-run",
            "linux_stopper": "/home/test/runtime/current/bin/omniinfer-vllm-stop",
            "linux_pid_dir": "/home/test/runtime/current/run",
            "automount_root": "/mnt",
        }))
        .unwrap(),
    )
    .unwrap();
    let backend = json!({
        "id": "vllm-wsl2-cuda",
        "launcher_path": manifest,
        "runtime_dir": root,
        "default_args": [],
        "external_server_protocol": "vllm-wsl2-openai-server",
        "log_file_name": "vllm-wsl2-server.log"
    });
    let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
        backend,
        model_path: r"D:\models\Qwen 4B".to_string(),
        mmproj_path: None,
        host: "127.0.0.1".to_string(),
        port: 24567,
        ctx_size: Some(8192),
        launch_args: None,
    })
    .unwrap();
    assert_eq!(plan.proxy_model_ref.as_deref(), Some("local"));
    assert!(
        plan.command
            .iter()
            .any(|arg| arg == "/mnt/d/models/Qwen 4B")
    );
    assert!(
        plan.command
            .windows(2)
            .any(|args| args == ["--max-model-len", "8192"])
    );
    assert_eq!(
        plan.stop_command.as_ref().unwrap().last().unwrap(),
        "/home/test/runtime/current/run/24567.pid"
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn wsl_vllm_rejects_unc_model_path() {
    let error = translate_wsl_model_ref(r"\\server\share\model", "/mnt").unwrap_err();
    assert_eq!(
        error,
        RuntimePlanError::UnsupportedWslModelPath(r"\\server\share\model".to_string())
    );
}

#[test]
fn wsl_vllm_keeps_huggingface_references_and_rejects_malformed_manifest() {
    assert_eq!(
        translate_wsl_model_ref("Qwen/Qwen2.5-7B-Instruct", "/mnt").unwrap(),
        "Qwen/Qwen2.5-7B-Instruct"
    );
    let manifest = WslVllmLauncherManifest {
        schema_version: 1,
        backend: "vllm-wsl2-cuda".to_string(),
        distribution: "Ubuntu-24.04".to_string(),
        linux_launcher: "/home/test/runtime/../other/vllm".to_string(),
        linux_runner: "/home/test/runtime/run".to_string(),
        linux_stopper: "/home/test/runtime/stop".to_string(),
        linux_pid_dir: "/home/test/runtime/pids".to_string(),
        automount_root: "/mnt".to_string(),
    };
    let error = validate_wsl_manifest("vllm-wsl2-cuda", Path::new("vllm-wsl2.json"), &manifest)
        .unwrap_err();
    assert!(matches!(
        error,
        RuntimePlanError::InvalidWslLauncherManifest { .. }
    ));
}

#[test]
fn ctx_size_replaces_existing_flag() {
    let backend = json!({
        "id": "llama.cpp-linux-cuda",
        "launcher_path": "/runtime/bin/llama-server",
        "runtime_dir": "/runtime",
        "default_args": ["-ngl", "999", "--ctx-size", "2048"],
        "external_server_protocol": "llama.cpp-server"
    });
    let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
        backend,
        model_path: "/models/qwen.gguf".to_string(),
        mmproj_path: None,
        host: "127.0.0.1".to_string(),
        port: 12345,
        ctx_size: Some(8192),
        launch_args: None,
    })
    .unwrap();
    assert_eq!(plan.ctx_size, Some(8192));
    assert!(plan.command.windows(2).any(|items| items == ["-c", "8192"]));
    assert!(
        !plan
            .command
            .windows(2)
            .any(|items| items == ["--ctx-size", "2048"])
    );
}

#[test]
fn rejects_reserved_managed_args() {
    let backend = json!({
        "id": "llama.cpp-linux-cuda",
        "launcher_path": "/runtime/bin/llama-server",
        "runtime_dir": "/runtime",
        "default_args": ["--host", "0.0.0.0"],
        "external_server_protocol": "llama.cpp-server"
    });
    let error = build_external_runtime_plan(&ExternalRuntimeRequest {
        backend,
        model_path: "/models/qwen.gguf".to_string(),
        mmproj_path: None,
        host: "127.0.0.1".to_string(),
        port: 12345,
        ctx_size: None,
        launch_args: None,
    })
    .unwrap_err();
    assert_eq!(
        error,
        RuntimePlanError::ReservedLaunchArg("--host".to_string())
    );
}
