#!/usr/bin/env python3
"""RuntimeManager: path resolution, embedded backend lifecycle."""

from __future__ import annotations

import tempfile
import unittest
import io
import os
import pty
import threading
import time
from pathlib import Path
from unittest.mock import patch

from service_core import commands, tui
from service_core.platforms.linux import LinuxPlatform
from service_core.platforms.mac import MacPlatform
from service_core.platforms.windows import WindowsPlatform
from service_core.platforms.common import display_path_reference
from service_core.platforms.registry import default_backend_for_current_host
from service_core.backends.base import BackendSpec
from service_core.cli import build_parser
from service_core.local_state import (
    legacy_state_file,
    load_selected_backend,
    load_selected_model,
    save_selected_backend,
    save_selected_model,
    state_file,
)
from service_core.runtime import RuntimeManager
from service_core.tui import _consume_visible_text


# ---------------------------------------------------------------------------
# Runtime root discovery
# ---------------------------------------------------------------------------


class RuntimeRootResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repo_root = self.root / "repo"
        self.app_root = self.root / "app"
        self.repo_root.mkdir()
        self.app_root.mkdir()
        self.platform = LinuxPlatform()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prefers_existing_requested_runtime_root(self) -> None:
        requested_root = self.app_root / "custom-runtime"
        requested_root.mkdir()

        resolved = self.platform.discover_runtime_root(
            repo_root=self.repo_root,
            app_root=self.app_root,
            requested_runtime_root=str(requested_root),
        )

        self.assertEqual(resolved, requested_root.resolve())

    def test_prefers_portable_runtime_root_under_app(self) -> None:
        portable_root = self.app_root / "runtime"
        portable_root.mkdir()

        resolved = self.platform.discover_runtime_root(
            repo_root=self.repo_root,
            app_root=self.app_root,
        )

        self.assertEqual(resolved, portable_root.resolve())

    def test_prefers_existing_local_runtime_root_under_repo(self) -> None:
        local_root = self.repo_root / ".local" / "runtime" / "linux"
        local_root.mkdir(parents=True)

        resolved = self.platform.discover_runtime_root(
            repo_root=self.repo_root,
            app_root=self.app_root,
        )

        self.assertEqual(resolved, local_root.resolve())

    def test_returns_canonical_local_runtime_root_when_missing(self) -> None:
        legacy_root = self.repo_root / "platform" / "Linux"
        legacy_root.mkdir(parents=True)

        resolved = self.platform.discover_runtime_root(
            repo_root=self.repo_root,
            app_root=self.app_root,
        )

        self.assertEqual(resolved, (self.repo_root / ".local" / "runtime" / "linux").resolve())

    def test_each_platform_uses_own_folder_name(self) -> None:
        for platform_cls, expected_name in [
            (LinuxPlatform, "linux"),
            (MacPlatform, "macos"),
            (WindowsPlatform, "windows"),
        ]:
            platform = platform_cls()
            resolved = platform.discover_runtime_root(
                repo_root=self.repo_root,
                app_root=self.app_root,
            )
            self.assertEqual(resolved.name, expected_name, f"{platform_cls.__name__} folder")


# ---------------------------------------------------------------------------
# Local state
# ---------------------------------------------------------------------------


class LocalStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app_root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_state_file_lives_under_project_local_config(self) -> None:
        self.assertEqual(state_file(self.app_root), self.app_root / ".local" / "config" / "state.json")

    def test_selected_backend_round_trips_through_local_state(self) -> None:
        save_selected_backend("ik_llama.cpp-linux-cuda", self.app_root)
        self.assertEqual(load_selected_backend(self.app_root), "ik_llama.cpp-linux-cuda")

    def test_selected_model_round_trips_through_local_state(self) -> None:
        save_selected_model(
            "/models/demo.gguf",
            self.app_root,
            mmproj="/models/mmproj.gguf",
            ctx_size=4096,
        )

        self.assertEqual(
            load_selected_model(self.app_root),
            {
                "model": "/models/demo.gguf",
                "mmproj": "/models/mmproj.gguf",
                "ctx_size": 4096,
            },
        )

    def test_legacy_cli_state_migrates_to_shared_state_file(self) -> None:
        legacy = legacy_state_file(self.app_root)
        legacy.parent.mkdir(parents=True)
        legacy.write_text('{"selected_backend": "llama.cpp-linux-cuda"}\n', encoding="utf-8")

        self.assertEqual(load_selected_backend(self.app_root), "llama.cpp-linux-cuda")
        self.assertTrue(state_file(self.app_root).is_file())
        self.assertFalse(legacy.is_file())

    def test_runtime_manager_prefers_persisted_backend(self) -> None:
        persisted_backend = default_backend_for_current_host()
        save_selected_backend(persisted_backend, self.app_root)

        manager = RuntimeManager(
            repo_root=str(self.app_root),
            app_root=str(self.app_root),
            backend_host="127.0.0.1",
            backend_port=0,
            startup_timeout_s=10,
            default_backend_id="definitely-not-a-backend",
        )

        self.assertEqual(manager.snapshot()["backend"], persisted_backend)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class CliParserTests(unittest.TestCase):
    def test_backend_list_defaults_to_compatible_scope(self) -> None:
        args = build_parser().parse_args(["backend", "list"])
        self.assertEqual(args.scope, "compatible")

    def test_top_level_load_alias_parses_like_model_load(self) -> None:
        args = build_parser().parse_args(["load", "-m", "model.gguf"])
        self.assertEqual(args.command, "load")
        self.assertEqual(args.model, "model.gguf")

    def test_chat_accepts_positional_prompt(self) -> None:
        args = build_parser().parse_args(["chat", "hello"])
        self.assertEqual(args.command, "chat")
        self.assertEqual(args.prompt, "hello")

    def test_top_level_select_is_not_a_command(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["select", "llama.cpp-linux"])


# ---------------------------------------------------------------------------
# Shared command helpers
# ---------------------------------------------------------------------------


class CommandHelperTests(unittest.TestCase):
    def test_backend_models_dir_defaults_to_shared_local_models_on_all_desktop_platforms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for platform_cls, runtime_name in [
                (LinuxPlatform, "linux"),
                (MacPlatform, "macos"),
                (WindowsPlatform, "windows"),
            ]:
                platform = platform_cls()
                backends = platform.build_backends(
                    app_root=root,
                    runtime_root=root / ".local" / "runtime" / runtime_name,
                    backend_overrides=None,
                )

                for backend in backends.values():
                    self.assertEqual(
                        Path(backend.models_dir or "").resolve(),
                        (root / ".local" / "models").resolve(),
                        f"{platform_cls.__name__} {backend.id}",
                    )

    def test_discovers_models_in_local_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "qwen" / "model.gguf"
            model.parent.mkdir()
            model.write_bytes(b"gguf")
            (model.parent / "ignore.txt").write_text("not a model", encoding="utf-8")

            rows = commands.discover_models_in_roots([root])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].label, "qwen/model.gguf")

    def test_links_manual_model_into_managed_models_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            external = root / "downloads" / "model.gguf"
            external.parent.mkdir()
            external.write_bytes(b"gguf")

            with patch("service_core.commands.APP_ROOT", root):
                linked = commands.link_model_into_managed_models(external)
                rows = commands.discover_models_in_roots([commands.managed_models_dir()])

            self.assertEqual(linked.parent.resolve(), (root / ".local" / "models" / "downloads").resolve())
            self.assertTrue(linked.is_symlink())
            self.assertEqual(linked.resolve(), external.resolve())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].path.resolve(), linked.resolve())
            self.assertEqual(rows[0].label, "downloads/model.gguf")

    def test_detects_model_files_in_manual_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "Qwen3.5-4B"
            model_dir.mkdir()
            (model_dir / "Qwen3.5-4B-Q8_0.gguf").write_bytes(b"gguf")
            (model_dir / "Qwen3.5-4B-Q4_K_M.gguf").write_bytes(b"gguf")
            (model_dir / "mmproj-F16.gguf").write_bytes(b"gguf")

            rows = commands.detect_model_files_in_directory(model_dir)

        self.assertEqual([row.label for row in rows], ["Qwen3.5-4B-Q4_K_M.gguf", "Qwen3.5-4B-Q8_0.gguf"])

    def test_links_directory_model_under_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            external_root = root / "Qwen3.5-4B"
            external = external_root / "snapshots" / "Qwen3.5-4B-Q4_K_M.gguf"
            external.parent.mkdir(parents=True)
            external.write_bytes(b"gguf")

            with patch("service_core.commands.APP_ROOT", root):
                linked = commands.link_model_into_managed_models(external, model_root=external_root)

            self.assertEqual(
                linked,
                root / ".local" / "models" / "Qwen3.5-4B" / "snapshots" / "Qwen3.5-4B-Q4_K_M.gguf",
            )
            self.assertTrue(linked.is_symlink())
            self.assertEqual(linked.resolve(), external.resolve())

    def test_model_reference_preserves_managed_symlink_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            external = root / "downloads" / "model.gguf"
            external.parent.mkdir()
            external.write_bytes(b"gguf")

            with patch("service_core.commands.APP_ROOT", root):
                linked = commands.link_model_into_managed_models(external)
                resolved = commands.resolve_model_reference(str(linked))

            self.assertEqual(resolved, linked)
            self.assertEqual(resolved.resolve(), external.resolve())

    def test_display_path_reference_preserves_symlink_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            external = root / "downloads" / "model.gguf"
            managed = root / ".local" / "models" / "downloads" / "model.gguf"
            external.parent.mkdir()
            managed.parent.mkdir(parents=True)
            external.write_bytes(b"gguf")
            managed.symlink_to(external)

            ref = display_path_reference(str(managed), str(root / ".local" / "models"))

            self.assertEqual(ref, "downloads/model.gguf")

    def test_remembered_model_load_options_require_existing_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.gguf"
            model.write_bytes(b"gguf")
            save_selected_model(str(model), root, ctx_size=2048)

            with patch("service_core.commands.APP_ROOT", root):
                options = commands.remembered_model_load_options()

        self.assertIsNotNone(options)
        self.assertEqual(options.model, str(model))
        self.assertEqual(options.ctx_size, 2048)

    def test_remembered_model_load_options_ignore_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_selected_model(str(root / "missing.gguf"), root)

            with patch("service_core.commands.APP_ROOT", root):
                options = commands.remembered_model_load_options()

        self.assertIsNone(options)

    def test_remembered_model_load_options_drop_missing_mmproj(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.gguf"
            model.write_bytes(b"gguf")
            save_selected_model(str(model), root, mmproj=str(root / "missing-mmproj.gguf"))

            with patch("service_core.commands.APP_ROOT", root):
                options = commands.remembered_model_load_options()

        self.assertIsNotNone(options)
        self.assertEqual(options.mmproj, None)

    def test_shutdown_service_posts_shutdown_when_running(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_is_running() -> bool:
            return len(calls) == 0

        def fake_request_json(method: str, endpoint: str, **_kwargs):
            calls.append((method, endpoint))
            return 200, {"ok": True}, b"{}"

        with (
            patch("service_core.commands.is_service_running", side_effect=fake_is_running),
            patch("service_core.commands.request_json", side_effect=fake_request_json),
        ):
            stopped = commands.shutdown_service(wait_timeout_s=0.1)

        self.assertTrue(stopped)
        self.assertEqual(calls, [("POST", "/omni/shutdown")])

    def test_shutdown_service_skips_when_not_running(self) -> None:
        with (
            patch("service_core.commands.is_service_running", return_value=False),
            patch("service_core.commands.request_json") as request,
        ):
            stopped = commands.shutdown_service(wait_timeout_s=0.1)

        self.assertFalse(stopped)
        request.assert_not_called()

    def test_tui_suppresses_initial_thinking_block(self) -> None:
        output, buffer, visible = _consume_visible_text("<think>hidden", visible_started=False)
        self.assertEqual(output, "")
        self.assertEqual(buffer, "<think>hidden")
        self.assertFalse(visible)

        output, buffer, visible = _consume_visible_text("<think>hidden</think>\nanswer", visible_started=False)
        self.assertEqual(output, "answer")
        self.assertEqual(buffer, "")
        self.assertTrue(visible)

    def test_tui_chat_prompt_records_readline_history(self) -> None:
        class FakeReadline:
            def __init__(self) -> None:
                self.items: list[str] = []

            def clear_history(self) -> None:
                self.items.clear()

            def add_history(self, value: str) -> None:
                self.items.append(value)

        fake = FakeReadline()
        with (
            patch("service_core.tui._readline", fake),
            patch("service_core.tui._CHAT_HISTORY", []),
            patch("builtins.input", return_value="你好啊你是谁"),
        ):
            result = tui._prompt("You", history=True)

        self.assertEqual(result, "你好啊你是谁")
        self.assertEqual(fake.items, ["你好啊你是谁"])

    def test_tui_menu_prompt_does_not_record_readline_history(self) -> None:
        class FakeReadline:
            def __init__(self) -> None:
                self.items: list[str] = []

            def clear_history(self) -> None:
                self.items.clear()

            def add_history(self, value: str) -> None:
                self.items.append(value)

        fake = FakeReadline()
        with (
            patch("service_core.tui._readline", fake),
            patch("service_core.tui._CHAT_HISTORY", ["previous chat"]),
            patch("builtins.input", return_value=""),
        ):
            result = tui._prompt("Select backend", default="1")

        self.assertEqual(result, "1")
        self.assertEqual(fake.items, [])

    def test_tui_menu_prompt_accepts_cancel(self) -> None:
        with (
            patch("service_core.tui._can_use_interactive_menu", return_value=False),
            patch("builtins.input", return_value="esc"),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            result = tui._select_menu(
                title="Models",
                subtitle="Pick a model",
                items=[tui._MenuItem(label="demo.gguf")],
            )

        self.assertIsNone(result)

    def test_tui_load_progress_hides_backend_start_detail(self) -> None:
        self.assertEqual(
            tui._model_load_progress_text("Starting backend llama.cpp-linux-cuda and loading model..."),
            "Loading model...",
        )
        self.assertEqual(tui._model_load_progress_text("Waiting for OmniInfer gateway..."), "Waiting for OmniInfer gateway...")

    def test_tui_decodes_common_arrow_escape_sequences(self) -> None:
        self.assertEqual(tui._decode_escape_sequence("[A"), "up")
        self.assertEqual(tui._decode_escape_sequence("OA"), "up")
        self.assertEqual(tui._decode_escape_sequence("[B"), "down")
        self.assertEqual(tui._decode_escape_sequence("OB"), "down")
        self.assertEqual(tui._decode_escape_sequence("[Z"), "")

    @unittest.skipIf(os.name == "nt", "PTY test is POSIX-only")
    def test_tui_reads_delayed_arrow_sequence_from_pty(self) -> None:
        master_fd, slave_fd = pty.openpty()
        slave = os.fdopen(slave_fd, "rb", buffering=0)

        class FakeStdin:
            def fileno(self) -> int:
                return slave.fileno()

        result: list[str] = []

        def read_key() -> None:
            with patch("sys.stdin", FakeStdin()):
                result.append(tui._read_menu_key_posix())

        reader = threading.Thread(target=read_key)
        reader.start()
        try:
            time.sleep(0.05)
            os.write(master_fd, b"\x1b")
            time.sleep(0.1)
            os.write(master_fd, b"[B")
            reader.join(timeout=1)
            self.assertEqual(result, ["down"])
        finally:
            reader.join(timeout=1)
            slave.close()
            os.close(master_fd)

    def test_tui_formats_last_context_usage(self) -> None:
        result = tui._format_context_usage(
            {"prompt_tokens": 13, "completion_tokens": 12, "total_tokens": 25},
            4096,
        )

        self.assertEqual(result, "prompt=13, completion=12, context=25/4096 (0.6%)")

    def test_tui_context_usage_requires_usage_payload(self) -> None:
        self.assertEqual(tui._format_context_usage(None, 4096), "not available yet")

    def test_tui_extracts_context_size_from_runtime_props(self) -> None:
        self.assertEqual(tui._context_size_from_runtime_props({"n_ctx": 8192}), 8192)
        self.assertEqual(
            tui._context_size_from_runtime_props({"default_generation_settings": {"n_ctx": 4096}}),
            4096,
        )
        self.assertEqual(
            tui._context_size_from_runtime_props({"default_generation_settings": {"params": {"n_ctx": 2048}}}),
            2048,
        )

    def test_tui_formats_missing_context_size_explicitly(self) -> None:
        self.assertEqual(tui._format_context_size(None, loaded=True), "backend default (unreported)")
        self.assertEqual(tui._format_context_size(None, loaded=False), "not loaded")

    def test_tui_builds_conversation_payload_with_history(self) -> None:
        with (
            patch("service_core.commands.ensure_service_running"),
            patch(
                "service_core.commands.current_runtime_state",
                return_value={"model": "demo.gguf", "request_defaults": {"temperature": 0.1}},
            ),
        ):
            payload = tui._build_conversation_payload(
                "next",
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            )

        self.assertEqual(
            payload["messages"],
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "next"},
            ],
        )
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(payload["temperature"], 0.1)

    def test_chat_payload_defaults_to_longer_completion_budget(self) -> None:
        with (
            patch("service_core.commands.ensure_service_running"),
            patch(
                "service_core.commands.current_runtime_state",
                return_value={"model": "demo.gguf", "request_defaults": {}},
            ),
        ):
            payload = commands.build_chat_payload(commands.ChatOptions(message="hello"))

        self.assertEqual(payload["max_tokens"], 2048)


# ---------------------------------------------------------------------------
# Embedded backend lifecycle
# ---------------------------------------------------------------------------


class _FakeEmbeddedDriver:
    def __init__(self) -> None:
        self.unloaded = False

    def load_model(
        self,
        *,
        model_path: str,
        model_ref: str,
        mmproj_path: str | None,
        ctx_size: int | None,
        load_options=None,
    ):
        return {
            "model_path": model_path,
            "model_ref": model_ref,
            "mmproj_path": mmproj_path,
            "ctx_size": ctx_size,
            "load_options": dict(load_options or {}),
        }

    def unload_model(self, state):
        self.unloaded = True
        self.state = state

    def chat_completion(self, state, payload):
        return {
            "object": "chat.completion",
            "model": payload.get("model") or state["model_ref"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "embedded hello"}, "finish_reason": "stop"}],
        }

    def stream_chat_completion(self, state, payload):
        yield {
            "object": "chat.completion.chunk",
            "model": payload.get("model") or state["model_ref"],
            "choices": [{"index": 0, "delta": {"content": "embedded "}, "finish_reason": None}],
        }
        yield {
            "object": "chat.completion.chunk",
            "model": payload.get("model") or state["model_ref"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }


class EmbeddedRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        repo_root = Path(self.temp_dir.name)
        self.model_dir = repo_root / "models" / "demo-model"
        self.model_dir.mkdir(parents=True)
        self.fake_driver = _FakeEmbeddedDriver()

        self._platform_patch = patch("service_core.runtime.current_host_platform", return_value=MacPlatform())
        self._driver_patch = patch("service_core.runtime.get_embedded_backend_driver", return_value=self.fake_driver)
        self._platform_patch.start()
        self._driver_patch.start()

        self.manager = RuntimeManager(
            repo_root=str(repo_root),
            app_root=str(repo_root),
            backend_host="127.0.0.1",
            backend_port=0,
            startup_timeout_s=10,
            default_backend_id="mlx-mac",
        )
        self.manager.backends["mlx-mac"].python_modules = ()

    def tearDown(self) -> None:
        self._driver_patch.stop()
        self._platform_patch.stop()
        self.temp_dir.cleanup()

    def test_select_model(self) -> None:
        result = self.manager.select_model(model=str(self.model_dir), backend_id="mlx-mac")
        self.assertEqual(result["selected_backend"], "mlx-mac")
        self.assertEqual(self.manager.current_runtime_mode(), "embedded")

    def test_chat_completion(self) -> None:
        self.manager.select_model(model=str(self.model_dir), backend_id="mlx-mac")
        response = self.manager.chat_completion({"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(response["choices"][0]["message"]["content"], "embedded hello")

    def test_stream_chat_completion(self) -> None:
        self.manager.select_model(model=str(self.model_dir), backend_id="mlx-mac")
        events = self.manager.stream_chat_completion({"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(events[0]["choices"][0]["delta"]["content"], "embedded ")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")

    def test_snapshot_with_loaded_model(self) -> None:
        self.manager.select_model(model=str(self.model_dir), backend_id="mlx-mac")
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["backend"], "mlx-mac")
        self.assertTrue(snapshot["backend_ready"])

    def test_stop_runtime(self) -> None:
        self.manager.select_model(model=str(self.model_dir), backend_id="mlx-mac")
        self.manager.stop_runtime()
        self.assertTrue(self.fake_driver.unloaded)

    def test_select_model_invalid_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.select_model(model=str(self.model_dir), backend_id="nonexistent-backend")

    def test_select_model_missing_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.manager.select_model(model="/tmp/does-not-exist-model-dir", backend_id="mlx-mac")

    def test_snapshot_without_loaded_model(self) -> None:
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["backend"], "mlx-mac")
        self.assertIsNone(snapshot["model"])
        self.assertFalse(snapshot["backend_ready"])


class ExternalRuntimeLaunchArgsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.launcher = root / "llama-server"
        self.launcher.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
        self.manager = RuntimeManager(
            repo_root=str(root),
            app_root=str(root),
            backend_host="127.0.0.1",
            backend_port=12345,
            startup_timeout_s=10,
            default_backend_id="llama.cpp-linux",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _backend(self, backend_id: str) -> BackendSpec:
        return BackendSpec(
            id=backend_id,
            label=backend_id,
            family="llama.cpp",
            runtime_dir=str(self.launcher.parent),
            launcher_path=str(self.launcher),
            models_dir=None,
            catalog_url=None,
            description=backend_id,
            capabilities=["chat"],
        )

    def test_ik_launch_uses_supported_webui_disable_arg(self) -> None:
        launch = self.manager._prepare_external_runtime_launch(
            self._backend("ik_llama.cpp-linux-cuda"),
            model_path="/tmp/model.gguf",
            mmproj_path=None,
        )
        self.assertIn("--webui", launch.cmd)
        self.assertIn("none", launch.cmd)
        self.assertNotIn("--no-webui", launch.cmd)

    def test_ik_launch_keeps_reasoning_jinja_defaults(self) -> None:
        backend = self._backend("ik_llama.cpp-linux-cuda")
        backend.default_args = ["--jinja", "--reasoning-format", "deepseek", "-ngl", "999"]

        launch = self.manager._prepare_external_runtime_launch(
            backend,
            model_path="/tmp/model.gguf",
            mmproj_path=None,
        )

        self.assertIn("--jinja", launch.cmd)
        self.assertIn("--reasoning-format", launch.cmd)
        self.assertIn("deepseek", launch.cmd)

    def test_llama_launch_keeps_no_webui_arg(self) -> None:
        launch = self.manager._prepare_external_runtime_launch(
            self._backend("llama.cpp-linux-cuda"),
            model_path="/tmp/model.gguf",
            mmproj_path=None,
        )
        self.assertIn("--no-webui", launch.cmd)
        self.assertNotIn("--webui", launch.cmd)


if __name__ == "__main__":
    unittest.main()
