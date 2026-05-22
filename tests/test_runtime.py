#!/usr/bin/env python3
"""RuntimeManager: path resolution, embedded backend lifecycle."""

from __future__ import annotations

import tempfile
import types
import unittest
import io
import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

from service_core import cli, commands, tui
from service_core.platforms.linux import LinuxPlatform
from service_core.platforms.mac import MacPlatform
from service_core.platforms.windows import WindowsPlatform
from service_core.platforms.common import display_path_reference
from service_core.platforms.registry import default_backend_for_current_host
from service_core.backends.base import BackendSpec
from service_core.cli import build_parser
from service_core.local_state import (
    legacy_state_file,
    load_default_thinking,
    load_selected_backend,
    load_selected_model,
    load_tui_show_reasoning,
    save_default_thinking,
    save_selected_backend,
    save_selected_model,
    save_tui_show_reasoning,
    state_file,
)
from service_core.runtime import LoadedRuntime, RuntimeManager, _summarize_backend_startup_failure
from service_core.service import load_app_config
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

    def test_default_thinking_round_trips_through_local_state(self) -> None:
        save_default_thinking(True, self.app_root)
        self.assertEqual(load_default_thinking(self.app_root), True)

        save_default_thinking(False, self.app_root)
        self.assertEqual(load_default_thinking(self.app_root), False)

    def test_tui_reasoning_display_defaults_hidden_and_round_trips(self) -> None:
        self.assertFalse(load_tui_show_reasoning(self.app_root))

        save_tui_show_reasoning(True, self.app_root)
        self.assertTrue(load_tui_show_reasoning(self.app_root))

        save_tui_show_reasoning(False, self.app_root)
        self.assertFalse(load_tui_show_reasoning(self.app_root))

    def test_app_config_uses_persisted_default_thinking(self) -> None:
        save_default_thinking(True, self.app_root)

        config = load_app_config(self.app_root)

        self.assertEqual(config["default_thinking"], "on")

    def test_app_config_defaults_window_mode_hidden(self) -> None:
        config = load_app_config(self.app_root)

        self.assertEqual(config["window_mode"], "hidden")

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

    def test_build_command_parses_in_source_checkout(self) -> None:
        with patch("service_core.commands.is_backend_build_supported", return_value=True):
            args = build_parser().parse_args(["build", "llama.cpp-cpu"])
        self.assertEqual(args.command, "build")
        self.assertEqual(args.backend_name, "llama.cpp-cpu")

    def test_build_command_is_hidden_in_packaged_release(self) -> None:
        with patch("service_core.commands.is_backend_build_supported", return_value=False):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["build", "llama.cpp-cpu"])

    def test_build_help_text_is_source_only(self) -> None:
        with patch("service_core.commands.is_backend_build_supported", return_value=True):
            self.assertIn("omniinfer build <backend>", build_parser().format_help())
        with patch("service_core.commands.is_backend_build_supported", return_value=False):
            self.assertNotIn("omniinfer build <backend>", build_parser().format_help())

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

    def test_requested_window_mode_defaults_hidden(self) -> None:
        self.assertEqual(cli._requested_window_mode([]), "hidden")
        self.assertEqual(cli._requested_window_mode(["--window-mode", "visible"]), "visible")

    def test_detached_health_target_uses_cli_bind_over_config(self) -> None:
        config = {"host": "127.0.0.1", "port": 9000, "startup_timeout": 60}
        with patch("service_core.cli.load_app_config", return_value=config):
            target = cli._detached_health_target(
                ["--port", "9157", "--host", "127.0.0.1", "--startup-timeout", "20"]
            )

        self.assertEqual(target, ("127.0.0.1", 9157, 20))

    def test_detached_health_target_checks_loopback_for_lan_bind(self) -> None:
        config = {"host": "127.0.0.1", "port": 9000, "startup_timeout": 60}
        with patch("service_core.cli.load_app_config", return_value=config):
            target = cli._detached_health_target(["--lan", "--port", "9158"])

        self.assertEqual(target, ("127.0.0.1", 9158, 60))

    def test_hidden_window_relaunch_redirects_child_output_and_waits_for_health(self) -> None:
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        with tempfile.TemporaryDirectory() as temp_dir:
            app_root = Path(temp_dir)
            log_path = app_root / ".local" / "logs" / "serve-child.log"
            fake_process = Mock()
            fake_process.poll.return_value = None
            popen = Mock(return_value=fake_process)
            with (
                patch("service_core.cli.os.name", "nt"),
                patch("service_core.cli.APP_ROOT", app_root),
                patch("service_core.cli.REPO_ROOT", app_root),
                patch(
                    "service_core.cli.load_app_config",
                    return_value={"host": "127.0.0.1", "port": 9000, "startup_timeout": 60},
                ),
                patch("service_core.cli._has_console", return_value=True),
                patch("service_core.cli._detached_child_log_path", return_value=log_path),
                patch("service_core.cli.sys.frozen", True, create=True),
                patch("service_core.cli.sys.executable", "omniinfer.exe"),
                patch("service_core.cli.sys.argv", ["omniinfer.exe", "serve", "--port", "9157", "--host", "127.0.0.1"]),
                patch("service_core.cli.subprocess.Popen", popen),
                patch("service_core.cli.urllib.request.urlopen", return_value=Response()) as urlopen,
                patch.dict(os.environ, {}, clear=True),
            ):
                with self.assertRaises(SystemExit) as raised:
                    cli._ensure_window_mode(["--port", "9157", "--host", "127.0.0.1"])

            self.assertEqual(raised.exception.code, 0)
            kwargs = popen.call_args.kwargs
            self.assertEqual(kwargs["cwd"], str(app_root))
            self.assertIsNotNone(kwargs["stdout"])
            self.assertEqual(kwargs["stderr"], cli.subprocess.STDOUT)
            urlopen.assert_called_with("http://127.0.0.1:9157/health", timeout=1)
            self.assertIn("launching detached OmniInfer", log_path.read_text(encoding="utf-8"))

    def test_wait_for_detached_health_reports_early_child_exit_with_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "serve-child.log"
            log_path.write_text("bind failed on requested port", encoding="utf-8")
            fake_process = Mock()
            fake_process.poll.return_value = 1

            with self.assertRaises(SystemExit) as raised:
                cli._wait_for_detached_health(
                    fake_process,
                    host="127.0.0.1",
                    port=9157,
                    timeout_s=20,
                    log_path=log_path,
                )

        message = str(raised.exception)
        self.assertIn("exited early", message)
        self.assertIn("127.0.0.1:9157", message)
        self.assertIn("bind failed on requested port", message)

    def test_serve_forwards_service_arguments(self) -> None:
        try:
            with patch("service_core.service.main", return_value=0) as service_main:
                result = cli.main(
                    ["serve", "--host", "0.0.0.0", "--window-mode", "visible", "--default-backend", "llama.cpp-linux"]
                )
        finally:
            cli._cli_port_override = None
            commands.set_port_override(None)

        self.assertEqual(result, 0)
        service_main.assert_called_once_with(
            ["--host", "0.0.0.0", "--window-mode", "visible", "--default-backend", "llama.cpp-linux"]
        )

    def test_interactive_serve_uses_server_tui(self) -> None:
        try:
            with (
                patch("sys.stdin.isatty", return_value=True),
                patch("sys.stdout.isatty", return_value=True),
                patch("service_core.tui.run_server_tui", return_value=0) as run_server_tui,
            ):
                result = cli.main(["serve", "--lan"])
        finally:
            cli._cli_port_override = None
            commands.set_port_override(None)

        self.assertEqual(result, 0)
        run_server_tui.assert_called_once_with(["--lan"])

    def test_direct_serve_env_skips_server_tui(self) -> None:
        try:
            with (
                patch.dict(os.environ, {"OMNIINFER_SERVE_DIRECT": "1"}),
                patch("sys.stdin.isatty", return_value=True),
                patch("sys.stdout.isatty", return_value=True),
                patch("service_core.tui.run_server_tui") as run_server_tui,
                patch("service_core.service.main", return_value=0) as service_main,
            ):
                result = cli.main(["serve", "--lan"])
        finally:
            cli._cli_port_override = None
            commands.set_port_override(None)

        self.assertEqual(result, 0)
        run_server_tui.assert_not_called()
        service_main.assert_called_once_with(["--lan"])

    def test_serve_lan_forwards_service_argument(self) -> None:
        try:
            with patch("service_core.service.main", return_value=0) as service_main:
                result = cli.main(["serve", "--lan", "--window-mode", "visible"])
        finally:
            cli._cli_port_override = None
            commands.set_port_override(None)

        self.assertEqual(result, 0)
        service_main.assert_called_once_with(["--lan", "--window-mode", "visible"])

    def test_serve_cloudflare_forwards_service_arguments(self) -> None:
        try:
            with patch("service_core.service.main", return_value=0) as service_main:
                result = cli.main(
                    [
                        "serve",
                        "--cloudflare",
                        "--cloudflared-path",
                        "/opt/bin/cloudflared",
                        "--cloudflare-no-print-key",
                    ]
                )
        finally:
            cli._cli_port_override = None
            commands.set_port_override(None)

        self.assertEqual(result, 0)
        service_main.assert_called_once_with(
            ["--cloudflare", "--cloudflared-path", "/opt/bin/cloudflared", "--cloudflare-no-print-key"]
        )

    def test_serve_help_forwards_to_service_without_hidden_window_restart(self) -> None:
        try:
            with patch("service_core.service.main", return_value=0) as service_main:
                result = cli.main(["serve", "-h"])
        finally:
            cli._cli_port_override = None
            commands.set_port_override(None)

        self.assertEqual(result, 0)
        service_main.assert_called_once_with(["-h"])

    def test_server_alias_forwards_to_service(self) -> None:
        try:
            with patch("service_core.service.main", return_value=0) as service_main:
                result = cli.main(["server", "--lan", "--window-mode", "visible"])
        finally:
            cli._cli_port_override = None
            commands.set_port_override(None)

        self.assertEqual(result, 0)
        service_main.assert_called_once_with(["--lan", "--window-mode", "visible"])

    def test_global_port_override_is_forwarded_to_serve(self) -> None:
        try:
            with patch("service_core.service.main", return_value=0) as service_main:
                result = cli.main(["--port", "9010", "serve", "--host", "127.0.0.1", "--window-mode", "visible"])
        finally:
            cli._cli_port_override = None
            commands.set_port_override(None)

        self.assertEqual(result, 0)
        service_main.assert_called_once_with(["--host", "127.0.0.1", "--window-mode", "visible", "--port", "9010"])

    def test_hidden_completion_includes_build_only_when_supported(self) -> None:
        with (
            patch.dict(os.environ, {"COMP_CWORD": "1"}, clear=False),
            patch("service_core.commands.is_backend_build_supported", return_value=True),
            patch("builtins.print") as print_mock,
        ):
            cli.handle_hidden_completion([""])
        suggestions = print_mock.call_args.args[0]
        self.assertIn("build", suggestions.split())

        with (
            patch.dict(os.environ, {"COMP_CWORD": "1"}, clear=False),
            patch("service_core.commands.is_backend_build_supported", return_value=False),
            patch("builtins.print") as print_mock,
        ):
            cli.handle_hidden_completion([""])
        suggestions = print_mock.call_args.args[0]
        self.assertNotIn("build", suggestions.split())


# ---------------------------------------------------------------------------
# Shared command helpers
# ---------------------------------------------------------------------------


class CommandHelperTests(unittest.TestCase):
    def assertSameFile(self, left: Path, right: Path) -> None:
        self.assertTrue(left.exists(), f"{left} should exist")
        self.assertTrue(right.exists(), f"{right} should exist")
        self.assertTrue(left.samefile(right), f"{left} should reference the same file as {right}")

    def backend_spec(self, backend_id: str, launcher_path: Path, capabilities: list[str] | None = None) -> BackendSpec:
        return BackendSpec(
            id=backend_id,
            label=backend_id,
            family="test",
            runtime_dir=str(launcher_path.parent.parent),
            launcher_path=str(launcher_path),
            models_dir=None,
            catalog_url=None,
            description="",
            capabilities=capabilities or [],
        )

    def test_source_gateway_launch_uses_cli_serve_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            with patch("service_core.commands.REPO_ROOT", repo_root):
                command = commands.gateway_launch_command(
                    host="127.0.0.1",
                    port=9000,
                    startup_timeout=60,
                    window_mode="hidden",
                    default_thinking="off",
                    default_backend="llama.cpp-linux",
                )

        self.assertEqual(command[1:3], [str(repo_root / "omniinfer.py"), "serve"])
        self.assertIn("--default-backend", command)
        self.assertIn("llama.cpp-linux", command)

    def test_packaged_gateway_launch_uses_cli_binary_serve(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            exe_path = Path(temp_dir) / "omniinfer"
            exe_path.write_text("", encoding="utf-8")
            with (
                patch("service_core.commands.sys.executable", str(exe_path)),
                patch("service_core.commands.sys.frozen", True, create=True),
            ):
                command = commands.gateway_launch_command(
                    host="127.0.0.1",
                    port=9000,
                    startup_timeout=60,
                    window_mode="hidden",
                    default_thinking="off",
                    default_backend="",
                )

            self.assertTrue(Path(command[0]).samefile(exe_path))
            self.assertEqual(command[1], "serve")

    def test_packaged_gateway_launch_prefers_omniinfer_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            new_exe = Path(temp_dir) / "omniinfer"
            old_exe = Path(temp_dir) / "omniinfer-cli"
            current_exe = Path(temp_dir) / "python"
            new_exe.write_text("", encoding="utf-8")
            old_exe.write_text("", encoding="utf-8")
            current_exe.write_text("", encoding="utf-8")
            with (
                patch("service_core.commands.sys.executable", str(current_exe)),
                patch("service_core.commands.sys.frozen", True, create=True),
            ):
                command = commands.gateway_launch_command(
                    host="127.0.0.1",
                    port=9000,
                    startup_timeout=60,
                    window_mode="hidden",
                    default_thinking="off",
                    default_backend="",
                )

            self.assertTrue(Path(command[0]).samefile(new_exe))
            self.assertEqual(command[1], "serve")

    def test_start_service_background_defaults_window_mode_hidden(self) -> None:
        config = {
            "host": "127.0.0.1",
            "port": 9000,
            "startup_timeout": 60,
            "default_thinking": "off",
            "default_backend": "",
        }
        with (
            patch("service_core.commands.get_service_config", return_value=config),
            patch("service_core.commands.subprocess.Popen") as popen,
        ):
            commands.start_service_background()

        command = popen.call_args.args[0]
        index = command.index("--window-mode")
        self.assertEqual(command[index + 1], "hidden")

    def test_server_backend_choices_include_installed_and_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            installed_launcher = root / "installed" / "bin" / "backend"
            installed_launcher.parent.mkdir(parents=True)
            installed_launcher.write_text("", encoding="utf-8")
            compatible_launcher = root / "compatible" / "bin" / "backend"
            incompatible_launcher = root / "incompatible" / "bin" / "backend"
            backends = {
                "installed": self.backend_spec("installed", installed_launcher, ["cpu"]),
                "compatible": self.backend_spec("compatible", compatible_launcher, ["cuda"]),
                "incompatible": self.backend_spec("incompatible", incompatible_launcher, ["rocm"]),
            }
            platform = types.SimpleNamespace(
                is_hardware_compatible=lambda backend: backend.id == "compatible",
            )
            with (
                patch("service_core.tui.commands.local_backends", return_value=backends),
                patch("service_core.tui.commands.is_backend_build_supported", return_value=True),
                patch("service_core.tui.current_host_platform", return_value=platform),
            ):
                choices = tui._server_backend_choices()

        self.assertEqual([backend.id for backend in choices], ["installed", "compatible"])

    def test_server_backend_choices_hide_uninstalled_without_build_support(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            installed_launcher = root / "installed" / "bin" / "backend"
            installed_launcher.parent.mkdir(parents=True)
            installed_launcher.write_text("", encoding="utf-8")
            compatible_launcher = root / "compatible" / "bin" / "backend"
            backends = {
                "installed": self.backend_spec("installed", installed_launcher, ["cpu"]),
                "compatible": self.backend_spec("compatible", compatible_launcher, ["cuda"]),
            }
            platform = types.SimpleNamespace(is_hardware_compatible=lambda _backend: True)
            with (
                patch("service_core.tui.commands.local_backends", return_value=backends),
                patch("service_core.tui.commands.is_backend_build_supported", return_value=False),
                patch("service_core.tui.current_host_platform", return_value=platform),
            ):
                choices = tui._server_backend_choices()

        self.assertEqual([backend.id for backend in choices], ["installed"])

    def test_command_service_base_url_uses_loopback_for_all_interfaces(self) -> None:
        with patch(
            "service_core.commands.load_app_config",
            return_value={"host": "0.0.0.0", "port": 9000},
        ):
            self.assertEqual(commands.service_base_url(), "http://127.0.0.1:9000")

    def test_backend_build_command_resolves_windows_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "scripts" / "platforms" / "windows" / "llama.cpp-cpu" / "build.ps1"
            script.parent.mkdir(parents=True)
            script.write_text("", encoding="utf-8")
            with (
                patch("service_core.commands.REPO_ROOT", root),
                patch("service_core.commands.sys.frozen", False, create=True),
                patch("service_core.commands.detect_system_name", return_value="windows"),
                patch("service_core.commands._is_windows", return_value=True),
                patch("service_core.commands.local_backends", return_value={"llama.cpp-cpu": object()}),
            ):
                command, script_path = commands.backend_build_command(
                    commands.BackendBuildOptions(backend="llama.cpp-cpu")
                )

        self.assertEqual(script_path, script)
        self.assertEqual(command[:5], ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"])
        self.assertEqual(command[-2:], ["-BuildType", "Release"])

    def test_backend_build_command_resolves_macos_script_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "scripts" / "platforms" / "macos" / "llama.cpp-mac" / "build.sh"
            script.parent.mkdir(parents=True)
            script.write_text("", encoding="utf-8")
            with (
                patch("service_core.commands.REPO_ROOT", root),
                patch("service_core.commands.sys.frozen", False, create=True),
                patch("service_core.commands.detect_system_name", return_value="mac"),
                patch("service_core.commands._is_windows", return_value=False),
                patch("service_core.commands.local_backends", return_value={"llama.cpp-mac": object()}),
            ):
                command, script_path = commands.backend_build_command(
                    commands.BackendBuildOptions(backend="llama.cpp-mac")
                )

        self.assertEqual(script_path, script)
        self.assertEqual(command, ["bash", str(script), "--build-type", "Release"])

    def test_backend_build_rejected_when_frozen(self) -> None:
        with patch("service_core.commands.sys.frozen", True, create=True):
            with self.assertRaises(SystemExit):
                commands.backend_build_command(commands.BackendBuildOptions(backend="llama.cpp-cpu"))

    def test_backend_build_runs_script_and_checks_output_binary(self) -> None:
        fake_backend = type(
            "FakeBackend",
            (),
            {
                "launcher_path": str(Path("runtime") / "bin" / "omniinfer-backend.cmd"),
                "binary_exists": True,
            },
        )()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "scripts" / "platforms" / "linux" / "llama.cpp-linux" / "build.sh"
            script.parent.mkdir(parents=True)
            script.write_text("", encoding="utf-8")
            with (
                patch("service_core.commands.REPO_ROOT", root),
                patch("service_core.commands.sys.frozen", False, create=True),
                patch("service_core.commands.detect_system_name", return_value="linux"),
                patch("service_core.commands._is_windows", return_value=False),
                patch("service_core.commands.local_backends", return_value={"llama.cpp-linux": fake_backend}),
                patch("service_core.commands.is_service_running", return_value=False),
                patch("service_core.commands.subprocess.run") as run_mock,
            ):
                run_mock.return_value.returncode = 0
                result = commands.build_backend(
                    commands.BackendBuildOptions(backend="llama.cpp-linux")
                )

        self.assertEqual(result.returncode, 0)
        run_mock.assert_called_once_with(["bash", str(script), "--build-type", "Release"], cwd=str(root), check=False)

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
            (model.parent / "mmproj-F16.gguf").write_bytes(b"gguf")

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
            self.assertSameFile(linked, external)
            self.assertEqual(len(rows), 1)
            self.assertSameFile(rows[0].path, external)
            self.assertEqual(rows[0].label, "downloads/model.gguf")

    def test_windows_symlink_privilege_error_falls_back_to_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            external = root / "downloads" / "model.gguf"
            external.parent.mkdir()
            external.write_bytes(b"gguf")
            symlink_error = OSError("symbolic link privilege not held")
            symlink_error.winerror = 1314

            with (
                patch("service_core.commands.APP_ROOT", root),
                patch("service_core.commands._is_windows", return_value=True),
                patch("service_core.commands.os.symlink", side_effect=symlink_error),
            ):
                linked = commands.link_model_into_managed_models(external)
                linked_again = commands.link_model_into_managed_models(external)
                rows = commands.discover_models_in_roots([commands.managed_models_dir()])

            self.assertEqual(linked, linked_again)
            self.assertFalse(linked.is_symlink())
            self.assertSameFile(linked, external)
            self.assertEqual(len(rows), 1)
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

            expected = (
                root
                / ".local"
                / "models"
                / "Qwen3.5-4B"
                / "snapshots"
                / "Qwen3.5-4B-Q4_K_M.gguf"
            )
            self.assertEqual(linked.parent.resolve(), expected.parent.resolve())
            self.assertEqual(linked.name, expected.name)
            self.assertSameFile(linked, external)

    def test_manual_directory_model_link_uses_detected_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manual_root = root / "models"
            external = manual_root / "gguf" / "qwen" / "qwen3.5-9b" / "Qwen3.5-9B-Q4_K_M.gguf"
            external.parent.mkdir(parents=True)
            external.write_bytes(b"gguf")

            with patch("service_core.commands.APP_ROOT", root):
                model_root = commands.infer_managed_model_root(external, manual_root)
                linked = commands.link_model_into_managed_models(
                    external,
                    model_root=model_root,
                    preserve_relative_path=False,
                )

            self.assertEqual(model_root, external.parent.resolve())
            expected = root / ".local" / "models" / "qwen3.5-9b" / external.name
            self.assertEqual(linked.parent.resolve(), expected.parent.resolve())
            self.assertEqual(linked.name, expected.name)
            self.assertSameFile(linked, external)

    def test_manual_directory_model_root_skips_snapshot_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manual_root = root / "models"
            external = (
                manual_root
                / "Qwen3.5-4B"
                / "snapshots"
                / "0123456789abcdef"
                / "Qwen3.5-4B-Q4_K_M.gguf"
            )
            external.parent.mkdir(parents=True)
            external.write_bytes(b"gguf")

            model_root = commands.infer_managed_model_root(external, manual_root)

            self.assertEqual(model_root, (manual_root / "Qwen3.5-4B").resolve())

    def test_model_reference_preserves_managed_link_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            external = root / "downloads" / "model.gguf"
            external.parent.mkdir()
            external.write_bytes(b"gguf")

            with patch("service_core.commands.APP_ROOT", root):
                linked = commands.link_model_into_managed_models(external)
                resolved = commands.resolve_model_reference(str(linked))

            self.assertEqual(resolved, linked)
            self.assertSameFile(resolved, external)

    def test_model_reference_accepts_quoted_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "Qwen3.5-4B-Q4_K_M.gguf"
            model.write_bytes(b"gguf")

            resolved = commands.resolve_model_reference(f'"{model}"')

            self.assertEqual(resolved.resolve(), model.resolve())

    def test_tui_manual_model_directory_accepts_quoted_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "Qwen3.5-4B-GGUF"
            model = model_dir / "Qwen3.5-4B-Q4_K_M.gguf"
            model_dir.mkdir()
            model.write_bytes(b"gguf")

            with patch("service_core.commands.APP_ROOT", root), patch(
                "service_core.tui._prompt",
                return_value=f'"{model_dir}"',
            ), patch("service_core.tui._print_notice"), patch(
                "service_core.commands.link_model_into_managed_models",
                side_effect=lambda path, **_kwargs: path,
            ) as link_model:
                linked = tui._prompt_model_path()

            self.assertIsNotNone(linked)
            assert linked is not None
            self.assertEqual(linked.resolve(), model.resolve())
            link_model.assert_called_once()
            self.assertEqual(link_model.call_args.args[0].resolve(), model.resolve())

    def test_display_path_reference_preserves_managed_link_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            external = root / "downloads" / "model.gguf"
            external.parent.mkdir()
            external.write_bytes(b"gguf")
            with patch("service_core.commands.APP_ROOT", root):
                managed = commands.link_model_into_managed_models(external)

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

    def test_tui_extracts_hidden_thinking_for_optional_display(self) -> None:
        output, buffer, visible, reasoning = tui._consume_visible_text_parts(
            "<think>hidden</think>\nanswer",
            visible_started=False,
        )

        self.assertEqual(output, "answer")
        self.assertEqual(buffer, "")
        self.assertTrue(visible)
        self.assertEqual(reasoning, "hidden")

    def test_tui_detects_hidden_thinking_wait_state(self) -> None:
        self.assertTrue(tui._is_hidden_thinking_pending("<think>hidden", visible_started=False))
        self.assertFalse(tui._is_hidden_thinking_pending("answer", visible_started=False))
        self.assertFalse(tui._is_hidden_thinking_pending("<think>hidden", visible_started=True))

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

    def test_tui_menu_filter_matches_label_and_details(self) -> None:
        items = [
            tui._MenuItem(label="llama.cpp-linux", details=["cpu"]),
            tui._MenuItem(label="llama.cpp-linux-cuda", details=["gpu", "cuda"]),
            tui._MenuItem(label="ik_llama.cpp-linux-cuda", details=["gpu", "cuda"]),
        ]

        self.assertEqual(tui._filter_menu_indices(items, "ik cuda"), [2])
        self.assertEqual(tui._filter_menu_indices(items, "gpu"), [1, 2])

    def test_tui_menu_moves_within_filtered_items(self) -> None:
        visible = [1, 3, 4]

        self.assertEqual(tui._move_visible_selection(visible, 1, 1), 3)
        self.assertEqual(tui._move_visible_selection(visible, 1, -1), 4)
        self.assertEqual(tui._move_visible_selection(visible, 0, 1), 3)

    def test_tui_menu_render_includes_search_prompt(self) -> None:
        rendered = tui._render_menu(
            "Backends",
            "Choose backend",
            [tui._MenuItem(label="llama.cpp-linux-cuda", details=["installed"])],
            0,
            query="cuda",
            visible_indices=[0],
        )

        self.assertIn("Search: cuda", rendered)
        self.assertIn("type to filter", rendered)

    def test_tui_load_progress_hides_backend_start_detail(self) -> None:
        self.assertEqual(tui._model_load_progress_text("Starting OmniInfer gateway..."), "Starting local service...")
        self.assertEqual(tui._model_load_progress_text("Waiting for OmniInfer gateway..."), "Starting local service...")
        self.assertEqual(tui._model_load_progress_text("Preparing model request..."), "Preparing model...")
        self.assertEqual(tui._model_load_progress_text("Sending model load request..."), "Preparing model...")
        self.assertEqual(tui._model_load_progress_text("Checking model artifact..."), "Checking model file...")
        self.assertEqual(tui._model_load_progress_text("Selecting backend for this model..."), "Selecting backend...")
        self.assertEqual(
            tui._model_load_progress_text("Loading model with llama.cpp-linux-cuda..."),
            "Loading model with llama.cpp-linux-cuda...",
        )
        self.assertEqual(
            tui._model_load_progress_text("Starting backend llama.cpp-linux-cuda and loading model..."),
            "Starting backend and loading model...",
        )

    def test_tui_decodes_common_arrow_escape_sequences(self) -> None:
        self.assertEqual(tui._decode_escape_sequence("[A"), "up")
        self.assertEqual(tui._decode_escape_sequence("OA"), "up")
        self.assertEqual(tui._decode_escape_sequence("[B"), "down")
        self.assertEqual(tui._decode_escape_sequence("OB"), "down")
        self.assertEqual(tui._decode_escape_sequence("[C"), "right")
        self.assertEqual(tui._decode_escape_sequence("[D"), "left")
        self.assertEqual(tui._decode_escape_sequence("[Z"), "")

    def test_tui_renders_input_cursor_inline(self) -> None:
        rendered = tui._render_input_value("你好", 1, 20)
        self.assertIn("▌", rendered)
        self.assertTrue(rendered.startswith("你"))

    def test_tui_renders_long_input_with_visible_cursor(self) -> None:
        rendered = tui._render_input_value("abcdefghijklmnopqrstuvwxyz", 2, 10)

        self.assertIn("▌", rendered)
        self.assertIn("…", rendered)
        self.assertLessEqual(tui._display_width(rendered), 10)
        self.assertLess(tui._input_cursor_prefix_width(rendered), tui._display_width(rendered))

    def test_tui_input_state_handles_unicode_backspace(self) -> None:
        state = tui._InputBoxState(history=[])
        for char in "你好啊你是谁":
            state.insert(char)

        for _ in range(3):
            state.backspace()

        self.assertEqual(state.text, "你好啊")
        self.assertEqual(state.cursor, 3)

    def test_tui_input_state_walks_chat_history(self) -> None:
        state = tui._InputBoxState(history=["first", "second"])
        for char in "draft":
            state.insert(char)

        state.history_previous()
        self.assertEqual(state.text, "second")
        state.history_previous()
        self.assertEqual(state.text, "first")
        state.history_next()
        self.assertEqual(state.text, "second")
        state.history_next()
        self.assertEqual(state.text, "draft")

    def test_tui_input_box_key_handler_edits_text_and_history(self) -> None:
        state = tui._InputBoxState(history=["older"])
        for key in ["h", "i", "left", "!"]:
            self.assertIsNone(tui._apply_input_box_key(state, key))

        self.assertEqual(state.text, "h!i")
        self.assertEqual(state.cursor, 2)
        self.assertIsNone(tui._apply_input_box_key(state, "right"))
        self.assertIsNone(tui._apply_input_box_key(state, "backspace"))
        self.assertEqual(state.text, "h!")
        self.assertIsNone(tui._apply_input_box_key(state, "up"))
        self.assertEqual(state.text, "older")
        self.assertIsNone(tui._apply_input_box_key(state, "down"))
        self.assertEqual(state.text, "h!")
        self.assertEqual(tui._apply_input_box_key(state, "enter"), "submit")

    def test_tui_input_box_ctrl_z_exits_only_when_empty(self) -> None:
        state = tui._InputBoxState(history=[])
        state.insert("x")
        state.cursor = 0

        self.assertIsNone(tui._apply_input_box_key(state, "ctrl_z"))
        self.assertEqual(state.text, "")
        self.assertEqual(tui._apply_input_box_key(state, "ctrl_z"), "exit")

    def test_tui_can_use_fixed_input_box_on_windows_with_vt(self) -> None:
        class FakeTTY:
            def isatty(self) -> bool:
                return True

        with (
            patch("service_core.tui.os.name", "nt"),
            patch("service_core.tui._enable_windows_virtual_terminal", return_value=True),
            patch("sys.stdin", FakeTTY()),
            patch("sys.stdout", FakeTTY()),
        ):
            self.assertTrue(tui._can_use_fixed_input_box())

    def test_tui_disables_fixed_input_box_on_windows_without_vt(self) -> None:
        class FakeTTY:
            def isatty(self) -> bool:
                return True

        with (
            patch("service_core.tui.os.name", "nt"),
            patch("service_core.tui._enable_windows_virtual_terminal", return_value=False),
            patch("sys.stdin", FakeTTY()),
            patch("sys.stdout", FakeTTY()),
        ):
            self.assertFalse(tui._can_use_fixed_input_box())

    def test_tui_reads_windows_input_box_extended_keys(self) -> None:
        keys = iter(["\xe0", "K", "\xe0", "M", "\xe0", "S", "\r"])
        fake_msvcrt = types.SimpleNamespace(getwch=lambda: next(keys))

        with patch.dict("sys.modules", {"msvcrt": fake_msvcrt}):
            self.assertEqual(tui._read_input_box_key_windows(), "left")
            self.assertEqual(tui._read_input_box_key_windows(), "right")
            self.assertEqual(tui._read_input_box_key_windows(), "delete")
            self.assertEqual(tui._read_input_box_key_windows(), "enter")

    @unittest.skipIf(os.name == "nt", "PTY test is POSIX-only")
    def test_tui_reads_arrow_sequence_from_pipe(self) -> None:
        read_fd, write_fd = os.pipe()

        class FakeStdin:
            def fileno(self) -> int:
                return read_fd

        try:
            os.write(write_fd, b"\x1b[B")
            with (
                patch("sys.stdin", FakeStdin()),
                patch("termios.tcgetattr", return_value=[]),
                patch("termios.tcsetattr"),
                patch("tty.setraw"),
            ):
                self.assertEqual(tui._read_menu_key_posix(), "down")
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_tui_formats_last_context_usage(self) -> None:
        result = tui._format_context_usage(
            {"prompt_tokens": 13, "completion_tokens": 12, "total_tokens": 25},
            4096,
        )

        self.assertEqual(result, "input=13, output=12, total=25/4096 (0.6%)")

    def test_tui_formats_prompt_status_usage(self) -> None:
        result = tui._format_status_usage(
            {"prompt_tokens": 13, "completion_tokens": 12, "total_tokens": 25},
            4096,
        )

        self.assertEqual(result, "in 13  out 12  ctx 25/4096 0.6%")

    def test_tui_formats_status_runtime_device_and_threads(self) -> None:
        state = {
            "backend_ready": True,
            "runtime_mode": "external_server",
            "backend_port": 12345,
            "ctx_size": 4096,
        }

        self.assertEqual(tui._format_status_runtime(state), "ready port 12345")
        self.assertEqual(tui._format_status_context_size(state), "ctx 4096")
        with patch.dict(os.environ, {"OMNIINFER_CUDA_VISIBLE_DEVICES": "7"}, clear=False):
            self.assertEqual(tui._format_status_device("llama.cpp-linux-cuda"), "cuda 7")
        self.assertEqual(tui._format_status_threads(["-t", "16", "--threads-batch=8"]), "t 16 tb 8")

    def test_tui_prompt_status_uses_model_basename(self) -> None:
        self.assertEqual(tui._format_status_model("/tmp/models/demo.gguf"), "demo.gguf")
        self.assertEqual(tui._format_status_model(None), "-")

    def test_tui_prompt_status_shows_slash_command_hints(self) -> None:
        self.assertIn("/backend", tui._prompt_status_text("/", "backend demo"))
        self.assertIn("load a different managed model", tui._prompt_status_text("/mo", "backend demo"))
        self.assertIn("no match", tui._prompt_status_text("/unknown", "backend demo"))
        self.assertEqual(tui._prompt_status_text("hello", "backend demo"), "backend demo")

    def test_tui_build_command_is_source_only(self) -> None:
        with patch("service_core.commands.is_backend_build_supported", return_value=True):
            self.assertIn(("/build", "build a compatible backend from this source checkout"), tui._command_table())
            self.assertIn("/build", tui._prompt_status_text("/", "backend demo"))

        with patch("service_core.commands.is_backend_build_supported", return_value=False):
            self.assertNotIn("/build", [name for name, _description in tui._command_table()])
            self.assertNotIn("/build", tui._prompt_status_text("/", "backend demo"))

    def test_tui_status_line_combines_prompt_context(self) -> None:
        session = tui._ChatSessionState(
            backend="llama.cpp-linux-cuda",
            last_usage={"prompt_tokens": 13, "completion_tokens": 12, "total_tokens": 25},
        )

        with (
            patch(
                "service_core.commands.current_runtime_state",
                return_value={
                    "model": "/tmp/demo.gguf",
                    "ctx_size": 4096,
                    "backend_ready": True,
                    "runtime_mode": "external_server",
                    "backend_port": 12345,
                    "launch_args": ["-t", "16"],
                },
            ),
            patch("service_core.commands.get_default_thinking", return_value=True),
        ):
            result = tui._StatusLine().text(session)

        self.assertIn("backend llama.cpp-linux-cuda", result)
        self.assertIn("model demo.gguf", result)
        self.assertIn("think on", result)
        self.assertIn("ready port 12345", result)
        self.assertIn("t 16", result)
        self.assertIn("ctx 25/4096 0.6%", result)

    def test_tui_status_view_groups_runtime_and_conversation_details(self) -> None:
        session = tui._ChatSessionState(
            backend="llama.cpp-linux-cuda",
            reasoning_visible=True,
            messages=[{"role": "user", "content": "hi"}],
            last_usage={"prompt_tokens": 13, "completion_tokens": 12, "total_tokens": 25},
        )

        with (
            patch(
                "service_core.commands.current_runtime_state",
                return_value={
                    "backend": "llama.cpp-linux-cuda",
                    "backend_ready": True,
                    "runtime_mode": "external_server",
                    "backend_port": 12345,
                    "model": "/tmp/demo.gguf",
                    "ctx_size": 4096,
                    "launch_args": ["-t", "16"],
                    "request_defaults": {"max_tokens": 2048},
                    "thinking": {"default_enabled": True},
                },
            ),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            tui._print_status(session=session)

        rendered = output.getvalue()
        self.assertIn("Runtime", rendered)
        self.assertIn("Model", rendered)
        self.assertIn("Generation", rendered)
        self.assertIn("Conversation", rendered)
        self.assertIn("Reasoning display", rendered)
        self.assertIn("show", rendered)
        self.assertIn("input=13, output=12, total=25/4096 (0.6%)", rendered)

    def test_tui_assistant_header_starts_at_line_beginning(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            tui._TranscriptView().render_assistant_header()

        self.assertEqual(output.getvalue(), "\rAssistant:\n")

    def test_tui_message_writer_keeps_multiline_output_in_block(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            writer = tui._MessageBlockWriter(kind="assistant")
            writer.write("hello\nworld")
            writer.finish()

        self.assertEqual(output.getvalue(), "  │ hello\n  │ world\n")

    def test_tui_fixed_prompt_during_output_uses_scroll_region(self) -> None:
        with (
            patch("service_core.tui._can_use_fixed_input_box", return_value=True),
            patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            with tui._FixedPromptDuringOutput("You", "backend demo"):
                print("Assistant: hello")

        rendered = output.getvalue()
        self.assertIn("\033[s\033[1;20r\033[u", rendered)
        self.assertIn("backend demo", rendered)
        self.assertIn("\033[20;1H", rendered)
        self.assertIn("\033[s\033[r\033[u", rendered)

    def test_tui_input_box_clears_previous_rows_after_resize(self) -> None:
        sizes = iter([os.terminal_size((80, 24)), os.terminal_size((80, 30))])

        with (
            patch("shutil.get_terminal_size", side_effect=lambda fallback=(80, 24): next(sizes)),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            tui._PROMPT_BOX_LAST_TOP = None
            tui._draw_input_box("You", tui._InputBoxState(history=[]), status="ready")
            tui._draw_input_box("You", tui._InputBoxState(history=[]), status="ready")
            tui._PROMPT_BOX_LAST_TOP = None

        rendered = output.getvalue()
        self.assertIn("\033[21;1H\033[2K", rendered)
        self.assertIn("\033[24;1H\033[2K", rendered)
        self.assertIn("\033[27;1H", rendered)

    def test_tui_notice_capture_routes_notices_to_status(self) -> None:
        center = tui._NoticeCenter()

        with patch("sys.stdout", new_callable=io.StringIO) as output:
            with center.capture():
                tui._print_notice("Model loaded", kind="success")
                tui._print_kv("Backend", "llama.cpp-linux-cuda")

        self.assertEqual(output.getvalue(), "")
        self.assertEqual(center.status_text(), "info: Backend: llama.cpp-linux-cuda")

    def test_tui_command_menu_sets_selected_command_notice(self) -> None:
        session = tui._ChatSessionState(backend="llama.cpp-linux-cuda")
        model_index = [name for name, _description in tui._command_table()].index("/model")

        with (
            patch("service_core.tui._can_use_interactive_menu", return_value=True),
            patch("service_core.tui._select_menu", return_value=model_index) as select_menu,
        ):
            tui._show_command_menu(session)

        select_menu.assert_called_once()
        self.assertIn("/model", session.notices.status_text())

    def test_tui_context_usage_requires_usage_payload(self) -> None:
        self.assertEqual(tui._format_context_usage(None, 4096), "not available yet")

    def test_tui_formats_speed_with_two_decimals(self) -> None:
        self.assertEqual(tui._format_speed(141.4599104824004), "141.46")
        self.assertEqual(tui._format_speed("8"), "8.00")
        self.assertEqual(tui._format_speed(None), "-")

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
        self.assertNotIn("think", payload)

    def test_chat_payload_includes_explicit_thinking_override(self) -> None:
        with (
            patch("service_core.commands.ensure_service_running"),
            patch(
                "service_core.commands.current_runtime_state",
                return_value={"model": "demo.gguf", "request_defaults": {}},
            ),
        ):
            payload = commands.build_chat_payload(commands.ChatOptions(message="hello", think="on"))

        self.assertTrue(payload["think"])

    def test_stream_parser_yields_reasoning_content(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def __iter__(self):
                events = [
                    {"choices": [{"delta": {"reasoning_content": "plan"}}]},
                    {"choices": [{"delta": {"content": "answer"}}]},
                    {"usage": {"total_tokens": 3}},
                ]
                for event in events:
                    yield f"data: {json.dumps(event)}\n".encode("utf-8")
                yield b"data: [DONE]\n"

        with patch("service_core.commands.urllib.request.urlopen", return_value=FakeResponse()):
            chunks = list(commands.iter_chat_stream_payload({"messages": [], "stream": True}))

        self.assertEqual(chunks[0].reasoning_text, "plan")
        self.assertEqual(chunks[1].text, "answer")
        self.assertEqual(chunks[2].final_payload, {"usage": {"total_tokens": 3}})

    def test_tui_chat_loop_starts_spinner_for_reasoning_chunks(self) -> None:
        spinners: list[Any] = []

        class FakeSpinner:
            def __init__(self, text: str) -> None:
                self.text = text
                self.active = False
                self.starts = 0
                self.stops = 0
                spinners.append(self)

            def start(self) -> None:
                self.active = True
                self.starts += 1

            def stop(self) -> None:
                self.active = False
                self.stops += 1

        def fake_stream(_payload: dict[str, Any]):
            yield commands.ChatStreamChunk(reasoning_text="plan")
            yield commands.ChatStreamChunk(text="answer")
            yield commands.ChatStreamChunk(final_payload={"usage": {"total_tokens": 2}})

        with (
            patch("service_core.tui._LoadingSpinner", FakeSpinner),
            patch("service_core.tui._print_chat_header"),
            patch("service_core.tui._prompt", side_effect=["hello", "/exit"]),
            patch("service_core.commands.get_tui_show_reasoning", return_value=False),
            patch("service_core.commands.get_default_thinking", return_value=False),
            patch("service_core.commands.build_chat_payload", return_value={"messages": []}),
            patch("service_core.commands.iter_chat_stream_payload", side_effect=fake_stream),
            patch("sys.stdout", io.StringIO()),
        ):
            tui._chat_loop("llama.cpp-linux-cuda")

        self.assertEqual(len(spinners), 1)
        self.assertEqual(spinners[0].starts, 1)
        self.assertGreaterEqual(spinners[0].stops, 1)

    def test_tui_chat_loop_can_render_visible_reasoning_chunks(self) -> None:
        def fake_stream(_payload: dict[str, Any]):
            yield commands.ChatStreamChunk(reasoning_text="plan")
            yield commands.ChatStreamChunk(text="answer")

        with (
            patch("service_core.tui._print_chat_header"),
            patch("service_core.tui._prompt", side_effect=["hello", "/exit"]),
            patch("service_core.commands.get_tui_show_reasoning", return_value=True),
            patch("service_core.commands.get_default_thinking", return_value=False),
            patch("service_core.commands.build_chat_payload", return_value={"messages": []}),
            patch("service_core.commands.iter_chat_stream_payload", side_effect=fake_stream),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            tui._chat_loop("llama.cpp-linux-cuda")

        rendered = output.getvalue()
        self.assertIn("Reasoning:", rendered)
        self.assertIn("  │ plan", rendered)
        self.assertIn("Assistant:", rendered)

    def test_tui_chat_loop_renders_user_before_assistant_with_fixed_prompt(self) -> None:
        def fake_stream(_payload: dict[str, Any]):
            yield commands.ChatStreamChunk(text="answer")

        with (
            patch("service_core.tui._print_chat_header"),
            patch("service_core.tui._prompt", side_effect=["hello", "/exit"]),
            patch("service_core.tui._can_use_fixed_input_box", return_value=True),
            patch("service_core.commands.get_tui_show_reasoning", return_value=False),
            patch("service_core.commands.get_default_thinking", return_value=False),
            patch("service_core.commands.build_chat_payload", return_value={"messages": []}),
            patch("service_core.commands.iter_chat_stream_payload", side_effect=fake_stream),
            patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            tui._chat_loop("llama.cpp-linux-cuda")

        rendered = output.getvalue()
        user_index = rendered.find("You:")
        assistant_index = rendered.find("Assistant:")
        self.assertGreaterEqual(user_index, 0)
        self.assertGreater(assistant_index, user_index)
        self.assertIn("  │ hello", rendered)
        self.assertIn("  │ answer", rendered)

    def test_tui_chat_loop_routes_stream_errors_to_prompt_status(self) -> None:
        prompts: list[str | None] = []
        inputs = iter(["hello", "/exit"])

        def fake_prompt(_label: str, **kwargs: Any) -> str:
            prompts.append(kwargs.get("status"))
            return next(inputs)

        def fake_stream(_payload: dict[str, Any]):
            raise SystemExit("No model is currently loaded.")
            yield

        with (
            patch("service_core.tui._prompt", side_effect=fake_prompt),
            patch("service_core.tui._print_chat_header"),
            patch("service_core.commands.get_tui_show_reasoning", return_value=False),
            patch("service_core.commands.current_runtime_state", return_value={"model": "demo.gguf"}),
            patch("service_core.commands.get_default_thinking", return_value=False),
            patch("service_core.commands.build_chat_payload", return_value={"messages": []}),
            patch("service_core.commands.iter_chat_stream_payload", side_effect=fake_stream),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            tui._chat_loop("llama.cpp-linux-cuda")

        self.assertNotIn("Assistant", output.getvalue())
        self.assertIn("warn: No model is currently loaded.", prompts[-1] or "")

    def test_tui_thinking_command_toggles_default(self) -> None:
        with (
            patch("service_core.commands.get_default_thinking", return_value=False),
            patch("service_core.commands.set_default_thinking", return_value=True) as set_default,
            patch("sys.stdout", io.StringIO()),
        ):
            tui._handle_thinking_command("/think")

        set_default.assert_called_once_with(True)

    def test_tui_thinking_command_sets_explicit_value(self) -> None:
        with (
            patch("service_core.commands.set_default_thinking", return_value=False) as set_default,
            patch("sys.stdout", io.StringIO()),
        ):
            tui._handle_thinking_command("/think off")

        set_default.assert_called_once_with(False)

    def test_tui_reasoning_command_sets_visible_display(self) -> None:
        session = tui._ChatSessionState(backend="llama.cpp-linux-cuda", reasoning_visible=False)

        with (
            patch("service_core.commands.set_tui_show_reasoning", return_value=True) as set_visible,
            patch("sys.stdout", io.StringIO()),
        ):
            tui._handle_reasoning_command(session, "/reasoning on")

        set_visible.assert_called_once_with(True)
        self.assertTrue(session.reasoning_visible)

    def test_set_default_thinking_persists_local_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch("service_core.commands.APP_ROOT", root),
                patch("service_core.commands.ensure_service_running"),
                patch(
                    "service_core.commands.request_json",
                    return_value=(200, {"default_enabled": True}, b"{}"),
                ),
            ):
                enabled = commands.set_default_thinking(True)

            self.assertTrue(enabled)
            self.assertEqual(load_default_thinking(root), True)

    def test_tui_backend_switch_reloads_remembered_model(self) -> None:
        options = commands.ModelLoadOptions(model="/models/demo.gguf")
        with (
            patch("service_core.commands.remembered_model_load_options", return_value=options),
            patch("service_core.tui._load_model", return_value="ik_llama.cpp-linux-cuda") as load_model,
            patch("service_core.tui._choose_model") as choose_model,
        ):
            backend = tui._load_model_after_backend_switch()

        self.assertEqual(backend, "ik_llama.cpp-linux-cuda")
        load_model.assert_called_once_with(options)
        choose_model.assert_not_called()

    def test_tui_backend_switch_prompts_when_no_remembered_model(self) -> None:
        model = Path("/models/demo.gguf")
        with (
            patch("service_core.commands.remembered_model_load_options", return_value=None),
            patch("service_core.tui._choose_model", return_value=model),
            patch("service_core.tui._load_model", return_value="llama.cpp-linux-cuda") as load_model,
        ):
            backend = tui._load_model_after_backend_switch()

        self.assertEqual(backend, "llama.cpp-linux-cuda")
        load_model.assert_called_once()
        self.assertEqual(load_model.call_args.args[0].model, str(model))


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
        self.assertEqual(snapshot["runtime_mode"], "embedded")
        self.assertEqual(snapshot["launch_args"], [])

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


class _AutoSelectPlatform(WindowsPlatform):
    def __init__(self) -> None:
        self.cuda_memory_available = False

    def available_memory_gib_for_backend(self, backend_name: str) -> float:
        if backend_name == "llama.cpp-cuda":
            return 999.0 if self.cuda_memory_available else 0.0
        return 999.0

    def safety_margin_gib_for_backend(self, backend_name: str) -> float:
        return 0.0


class AutoSelectRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model_path = self.root / "Qwen3.5-2B-Q4_K_M.gguf"
        self.model_path.write_text("model", encoding="utf-8")
        self.platform = _AutoSelectPlatform()
        self._platform_patch = patch("service_core.runtime.current_host_platform", return_value=self.platform)
        self._platform_patch.start()

        self.manager = RuntimeManager(
            repo_root=str(self.root),
            app_root=str(self.root),
            backend_host="127.0.0.1",
            backend_port=0,
            startup_timeout_s=10,
            default_backend_id="llama.cpp-cpu",
        )
        for backend_id in ("llama.cpp-cpu", "llama.cpp-cuda"):
            backend = self.manager.backends[backend_id]
            bin_dir = Path(backend.runtime_dir) / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            launcher = bin_dir / "llama-server.exe"
            launcher.write_text("", encoding="utf-8")
            backend.launcher_path = str(launcher)
        self.manager.catalog.available_backend_ids = {"llama.cpp-cpu", "llama.cpp-cuda"}
        self.started_backends: list[str] = []

    def tearDown(self) -> None:
        self._platform_patch.stop()
        self.temp_dir.cleanup()

    def _loaded_runtime(self, backend_id: str, model_path: str) -> LoadedRuntime:
        return LoadedRuntime(
            backend_id=backend_id,
            model_path=model_path,
            model_ref=Path(model_path).name,
            mmproj_path=None,
            mmproj_ref=None,
            ctx_size=None,
            runtime_mode="external_server",
            host="127.0.0.1",
            port=12345,
            process=None,
            launch_args=list(self.manager.backends[backend_id].default_args),
            request_defaults={},
        )

    def test_auto_select_releases_previous_runtime_before_memory_check(self) -> None:
        old_model = self.root / "old.gguf"
        old_model.write_text("old", encoding="utf-8")
        self.manager.loaded_runtime = self._loaded_runtime("llama.cpp-cuda", str(old_model))
        self.manager._is_runtime_running_locked = lambda: self.manager.loaded_runtime is not None  # type: ignore[method-assign]

        def stop_runtime() -> None:
            self.platform.cuda_memory_available = True
            self.manager.loaded_runtime = None

        def start_runtime(backend, model_path, mmproj_path, **kwargs):
            self.started_backends.append(backend.id)
            runtime = self._loaded_runtime(backend.id, model_path)
            self.manager.loaded_runtime = runtime
            return runtime

        self.manager._stop_runtime_locked = stop_runtime  # type: ignore[method-assign]
        self.manager._start_runtime_locked = start_runtime  # type: ignore[method-assign]

        result = self.manager.select_model(model=str(self.model_path))

        self.assertEqual(result["selected_backend"], "llama.cpp-cuda")
        self.assertEqual(self.started_backends, ["llama.cpp-cuda"])


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

    def test_ik_launch_uses_jinja_without_unsupported_reasoning_flag(self) -> None:
        backend = self._backend("ik_llama.cpp-linux-cuda")
        backend.default_args = ["--jinja", "-ngl", "999"]

        launch = self.manager._prepare_external_runtime_launch(
            backend,
            model_path="/tmp/model.gguf",
            mmproj_path=None,
        )

        self.assertIn("--jinja", launch.cmd)
        self.assertNotIn("--reasoning-format", launch.cmd)

    def test_llama_launch_keeps_no_webui_arg(self) -> None:
        launch = self.manager._prepare_external_runtime_launch(
            self._backend("llama.cpp-linux-cuda"),
            model_path="/tmp/model.gguf",
            mmproj_path=None,
        )
        self.assertIn("--no-webui", launch.cmd)
        self.assertNotIn("--webui", launch.cmd)

    def test_startup_failure_summary_prefers_unknown_argument(self) -> None:
        summary = _summarize_backend_startup_failure(
            [
                "error: unknown argument: -mm",
                "usage: llama-server [options]",
                "export-lora:",
                "  -m,    --model                  model path from which to load base model",
            ]
        )

        self.assertEqual(summary, "error: unknown argument: -mm")


if __name__ == "__main__":
    unittest.main()
