import os
import stat
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
HELPER = SCRIPTS_DIR / "jekyll-env.sh"
BUILD = SCRIPTS_DIR / "jekyll-build.sh"
SERVE = SCRIPTS_DIR / "jekyll-serve.sh"


class JekyllScriptsTest(unittest.TestCase):
    def test_scripts_exist_and_are_executable(self):
        for script in (HELPER, BUILD, SERVE):
            with self.subTest(script=script.name):
                self.assertTrue(script.exists(), f"missing script: {script}")
                mode = script.stat().st_mode
                self.assertTrue(mode & stat.S_IXUSR, f"script is not executable: {script}")

    def test_build_script_help(self):
        result = subprocess.run(
            [str(BUILD), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bundle exec jekyll build", result.stdout)

    def test_serve_script_help(self):
        result = subprocess.run(
            [str(SERVE), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bundle exec jekyll serve", result.stdout)
        self.assertIn("PORT", result.stdout)

    def test_serve_script_passes_default_host_and_port(self):
        env = os.environ.copy()
        env["JEKYLL_BIN"] = "/bin/echo"
        result = subprocess.run(
            [str(SERVE)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("serve --host 127.0.0.1 --port 4001", result.stdout)

    def test_env_script_does_not_force_legacy_user_gem_path(self):
        command = (
            "unset BUNDLER_USER_BIN; "
            "source scripts/jekyll-env.sh; "
            'printf "%s" "$PATH"'
        )
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(".gem/ruby/2.6.0/bin", result.stdout)

    def test_env_script_does_not_pin_bundler_version_by_default(self):
        command = (
            "unset JEKYLL_BUNDLER_VERSION; "
            "source scripts/jekyll-env.sh; "
            'printf "%s" "${JEKYLL_BUNDLER_VERSION:-}"'
        )
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual("", result.stdout)

    def test_env_script_initializes_rbenv_when_available(self):
        command = (
            'export PATH="/usr/bin:/bin"; '
            "source scripts/jekyll-env.sh; "
            'printf "%s" "$PATH"'
        )
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(".rbenv/shims", result.stdout)


if __name__ == "__main__":
    unittest.main()
