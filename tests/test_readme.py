import re
import unittest
import urllib.parse
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPOSITORY_ROOT / "README.md"
DEMO_ASSET_DIR = REPOSITORY_ROOT / "docs" / "assets" / "demo"
DEMO_POSTER_MAX_BYTES = 400 * 1024
TERMINAL_ANIMATION_MAX_BYTES = 3 * 1024 * 1024
VLA_ANIMATION_MAX_BYTES = 6 * 1024 * 1024
DEMO_POSTER_TOTAL_MAX_BYTES = 2 * 1024 * 1024
DEMO_ANIMATION_TOTAL_MAX_BYTES = 10 * 1024 * 1024
RETIRED_DEMO_ATTACHMENT = "4ac5329e-8c54-4ea9-8a51-02306c0607e9"
RETIRED_VLA_DEMO_ATTACHMENT = "83eb563d-60fc-42f6-9032-a9c7b7eedb8c"


class RootReadmeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = README_PATH.read_text(encoding="utf-8")

    def test_product_sections_are_concise_and_ordered(self):
        sections = (
            "## Quick Start",
            "## Demo",
            "## News",
            "## About",
            "## Platform Support",
            "## Documentation",
            "## Architecture",
            "## Contributing",
            "## Citation",
            "## License",
        )
        offsets = [self.readme.index(section) for section in sections]
        self.assertEqual(offsets, sorted(offsets))

    def test_primary_entry_points_and_status_badges_are_visible(self):
        for marker in (
            'href="#quick-start"',
            'href="#documentation"',
            "https://github.com/omnimind-ai/OmniInfer/releases",
            "actions/workflows/main-platform-ci.yml/badge.svg",
            "img.shields.io/github/v/release/omnimind-ai/OmniInfer",
            "img.shields.io/github/license/omnimind-ai/OmniInfer",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.readme)

    def test_three_release_installers_and_first_run_are_copyable(self):
        for platform in ("Linux x64", "macOS arm64", "Windows x64 PowerShell"):
            with self.subTest(platform=platform):
                self.assertIn(f"<th>{platform}</th>", self.readme)
        self.assertGreaterEqual(self.readme.count("scripts/install.sh | bash"), 2)
        self.assertIn("scripts/install.ps1 | iex", self.readme)
        self.assertIn("1. Run `omniinfer` in a terminal.", self.readme)
        self.assertIn("2. Choose a compatible backend.", self.readme)
        self.assertIn("3. Select a local model and start chatting.", self.readme)

    def test_implementation_details_stay_in_subdocuments(self):
        for detail in (
            "install-from-source.sh",
            "install-from-source.ps1",
            "--state-root",
            "--runtime-root",
            "cloudflared",
            "vllm-wsl2-cuda",
            "vllm-wsl2-rocm",
        ):
            with self.subTest(detail=detail):
                self.assertNotIn(detail, self.readme)

        cli = (REPOSITORY_ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
        remote = (REPOSITORY_ROOT / "docs" / "remote-access.md").read_text(
            encoding="utf-8"
        )
        installation = (
            REPOSITORY_ROOT / "docs" / "installation.md"
        ).read_text(encoding="utf-8")
        build = (REPOSITORY_ROOT / "docs" / "build.md").read_text(encoding="utf-8")
        self.assertIn("#### Desktop application integration", cli)
        self.assertIn("#### Windows vLLM through WSL2", cli)
        self.assertIn("## Managed cloudflared", remote)
        self.assertIn("## Complete Source Setup", installation)
        self.assertIn("## Manual Installation", installation)
        self.assertIn("## Remove the Release CLI", installation)
        self.assertIn("## Windows", build)
        self.assertIn("## Linux", build)
        self.assertIn("## macOS", build)

    def test_news_copy_is_preserved(self):
        self.assertIn(
            "- **2026-08-14** — 🚀 **Day-0 support for Qwen3.8-27B.** "
            "OmniInfer is ready for Qwen's latest 27B vision-language model "
            "from day one.",
            self.readme,
        )

    def test_demo_section_presents_three_focused_demos(self):
        demos = (
            (
                "### Terminal UI — choose a backend, load a model, chat locally",
                "docs/assets/demo/tui-chat-poster.webp",
            ),
            (
                "### Local API — OpenAI-compatible endpoint",
                "docs/assets/demo/local-api-poster.webp",
            ),
            (
                "### Browser VLA demo — SmolVLA on LIBERO",
                "docs/assets/demo/vla-libero-poster.webp",
            ),
        )
        for title, poster in demos:
            with self.subTest(title=title):
                self.assertIn(title, self.readme)
                self.assertIn(f'href="{poster}"', self.readme)
        self.assertNotIn("<video", self.readme)
        self.assertIn(
            'Static preview: <a href="docs/assets/demo/vla-libero-poster.webp">'
            "SmolVLA LIBERO dashboard screenshot</a>",
            self.readme,
        )
        self.assertEqual(self.readme.count('<img src="docs/assets/demo/'), 3)
        self.assertEqual(self.readme.count('width="720" alt="'), 3)

    def test_demo_animations_are_placeholder_free_and_described(self):
        for placeholder in (
            "OMNIINFER_DEMO_VIDEO_TUI",
            "OMNIINFER_DEMO_VIDEO_API",
            "OMNIINFER_DEMO_VIDEO_VLA",
        ):
            with self.subTest(placeholder=placeholder):
                self.assertNotIn(placeholder, self.readme)
        animations = (
            ("docs/assets/demo/tui-chat.webp", "Terminal UI"),
            ("docs/assets/demo/local-api.webp", "OpenAI-compatible"),
            ("docs/assets/demo/vla-libero.webp", "SmolVLA LIBERO"),
        )
        for path, label in animations:
            with self.subTest(path=path):
                self.assertIn(f'src="{path}"', self.readme)
                self.assertIn(f'alt="{label}', self.readme)

    def test_old_provisional_demo_media_is_retired(self):
        self.assertNotIn(RETIRED_DEMO_ATTACHMENT, self.readme)
        self.assertNotIn("docs/assets/demo/vla-libero.gif", self.readme)
        self.assertNotIn(RETIRED_VLA_DEMO_ATTACHMENT, self.readme)

    def test_demo_posters_exist_within_size_budget(self):
        posters = ("tui-chat-poster.webp", "local-api-poster.webp", "vla-libero-poster.webp")
        total = 0
        for name in posters:
            path = DEMO_ASSET_DIR / name
            with self.subTest(poster=name):
                self.assertTrue(path.is_file(), path)
                size = path.stat().st_size
                self.assertLessEqual(size, DEMO_POSTER_MAX_BYTES)
                total += size
        self.assertLessEqual(total, DEMO_POSTER_TOTAL_MAX_BYTES)

    def test_demo_animations_exist_within_size_budget(self):
        animations = (
            ("tui-chat.webp", TERMINAL_ANIMATION_MAX_BYTES),
            ("local-api.webp", TERMINAL_ANIMATION_MAX_BYTES),
            ("vla-libero.webp", VLA_ANIMATION_MAX_BYTES),
        )
        total = 0
        for name, max_bytes in animations:
            path = DEMO_ASSET_DIR / name
            with self.subTest(animation=name):
                self.assertTrue(path.is_file(), path)
                size = path.stat().st_size
                self.assertLessEqual(size, max_bytes)
                total += size
        self.assertLessEqual(total, DEMO_ANIMATION_TOTAL_MAX_BYTES)

    def test_readme_has_no_local_machine_details(self):
        leaks = (
            "/home/",
            ":\\Users",
            "zhangguanhuai",
            "zzw@",
            "time-crystal",
            "yutong",
        )
        for leak in leaks:
            with self.subTest(leak=leak):
                self.assertNotIn(leak, self.readme)
        self.assertEqual(
            re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", self.readme),
            [],
        )

    def test_local_readme_links_resolve(self):
        markdown_targets = re.findall(r"!?(?:\[[^\]]*\])\(([^)]+)\)", self.readme)
        html_targets = re.findall(r"(?:href|src)=\"([^\"]+)\"", self.readme)
        missing = []
        for raw_target in markdown_targets + html_targets:
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            parsed = urllib.parse.urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            relative_path = urllib.parse.unquote(parsed.path)
            if relative_path and not (REPOSITORY_ROOT / relative_path).exists():
                missing.append(target)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
