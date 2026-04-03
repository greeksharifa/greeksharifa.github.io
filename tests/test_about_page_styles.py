import unittest
from pathlib import Path


class AboutPageStyleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = Path("about.md").read_text(encoding="utf-8")

    def test_about_page_uses_more_compact_typography(self):
        expected_tokens = [
            "line-height: 1.4;",
            "max-width: 820px;",
            "padding: 28px 20px;",
            "margin-bottom: 24px;",
            "width: 128px;",
            "font-size: 1.7em;",
            "font-size: 0.84em;",
            "font-size: 1.12em;",
            "margin-bottom: 12px;",
            "font-size: 0.9em;",
        ]
        for token in expected_tokens:
            with self.subTest(token=token):
                self.assertIn(token, self.text)


if __name__ == "__main__":
    unittest.main()
