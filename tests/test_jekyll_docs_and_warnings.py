import unittest
from pathlib import Path


class JekyllDocsAndWarningsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = Path("README.md").read_text(encoding="utf-8")
        cls.config = Path("_config.yml").read_text(encoding="utf-8")
        cls.github_actions_post = Path(
            "_posts/2022-07-18-python-project.md"
        ).read_text(encoding="utf-8")

    def test_readme_documents_local_jekyll_scripts(self):
        expected_tokens = [
            "scripts/jekyll-build.sh",
            "scripts/jekyll-serve.sh",
            "http://127.0.0.1:4001/",
        ]
        for token in expected_tokens:
            with self.subTest(token=token):
                self.assertIn(token, self.readme)

    def test_config_uses_plugins_key_instead_of_legacy_gems_key(self):
        self.assertNotIn("\ngems:\n", self.config)
        self.assertIn("\nplugins:\n", self.config)
        self.assertIn("  - jekyll-paginate", self.config)
        self.assertIn("  - jekyll-redirect-from", self.config)

    def test_github_actions_examples_are_wrapped_in_raw_blocks(self):
        raw_start = "{% raw %}\n```yaml"
        raw_end = "```\n{% endraw %}"
        self.assertIn(raw_start, self.github_actions_post)
        self.assertIn(raw_end, self.github_actions_post)
        self.assertIn("${{ github.event_name == 'pull_request' && github.event.action == 'unassigned' }}", self.github_actions_post)
        self.assertIn("${{ failure() }}", self.github_actions_post)


if __name__ == "__main__":
    unittest.main()
