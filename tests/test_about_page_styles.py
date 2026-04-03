import unittest
from pathlib import Path


class AboutPageStyleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = Path("about.md").read_text(encoding="utf-8")

    def test_about_page_uses_more_compact_typography(self):
        expected_tokens = [
            "font-size: 0.94rem;",
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

    def test_about_page_title_is_centered(self):
        self.assertIn(".page-title {\ntext-align: center;\n}", self.text)

    def test_publication_sections_use_smaller_text_hierarchy(self):
        expected_tokens = [
            ".publication-section .entry-title {\nfont-size: 0.84em;\n}",
            ".publication-section .pub-authors,\n.publication-section .entry-subtitle {\nfont-size: 0.71em;\n}",
        ]
        for token in expected_tokens:
            with self.subTest(token=token):
                self.assertIn(token, self.text)

    def test_publication_section_class_is_applied_to_papers_block(self):
        self.assertIn("<!-- Education -->\n<div class=\"section compact-section\">", self.text)
        self.assertIn("<!-- =                        International Papers                       = -->\n    <!-- ===================================================================== -->\n    <h2>International Papers</h2>", self.text)
        self.assertIn("</div>\n\n\n<div class=\"section publication-section\">\n    <!-- ===================================================================== -->", self.text)

    def test_education_and_internships_use_compact_text_hierarchy(self):
        expected_tokens = [
            ".compact-section .entry-title {\nfont-size: 0.84em;\n}",
            ".compact-section .entry-meta {\nfont-size: 0.71em;\n}",
            ".compact-section .entry-subtitle,\n.compact-section .entry-description {\nfont-size: 0.71em;\n}",
            "<!-- Education -->\n<div class=\"section compact-section\">",
            "<!-- Internship -->\n<div class=\"section compact-section\">",
        ]
        for token in expected_tokens:
            with self.subTest(token=token):
                self.assertIn(token, self.text)


if __name__ == "__main__":
    unittest.main()
