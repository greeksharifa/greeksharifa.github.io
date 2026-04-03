import unittest
from pathlib import Path


class SidebarContactLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sidebar_html = Path("_includes/sidebar.html").read_text(encoding="utf-8")
        cls.sidebar_scss = Path("_scss/component/_sidebar.scss").read_text(encoding="utf-8")

    def test_follow_me_uses_single_inline_paragraph(self):
        self.assertIn('<p class="sidebar-contact-inline">', self.sidebar_html)
        self.assertIn("Follow me:", self.sidebar_html)

        unexpected_tokens = [
            'class="sidebar-contact-row"',
            'class="sidebar-contact-label"',
            'class="sidebar-contact-links"',
        ]
        for token in unexpected_tokens:
            with self.subTest(token=token):
                self.assertNotIn(token, self.sidebar_html)

    def test_contact_links_use_font_awesome_icons_only(self):
        expected_tokens = [
            'fa fa-envelope',
            'fa fa-graduation-cap',
            'fa fa-linkedin',
            'fa fa-github',
            'href="https://www.linkedin.com/in/you-won-jang-greeksharifa"',
        ]
        for token in expected_tokens:
            with self.subTest(token=token):
                self.assertIn(token, self.sidebar_html)

        unexpected_tokens = [
            'Google_Scholar_logo.svg.png',
            'LinkedIn_icon.svg.png',
            'sidebar-contact-link-image',
        ]
        for token in unexpected_tokens:
            with self.subTest(token=token):
                self.assertNotIn(token, self.sidebar_html)

    def test_contact_links_use_inline_spacing_like_reference_sidebar(self):
        expected_tokens = [
            ".sidebar-contact-inline",
            "font-size: 1em;",
            "line-height: 1;",
            "margin-left: .35rem;",
        ]
        for token in expected_tokens:
            with self.subTest(token=token):
                self.assertIn(token, self.sidebar_scss)

        unexpected_tokens = [
            ".sidebar-contact-row",
            ".sidebar-contact-label",
            ".sidebar-contact-links",
            "width: 1.25rem;",
            "height: 1.25rem;",
        ]
        for token in unexpected_tokens:
            with self.subTest(token=token):
                self.assertNotIn(token, self.sidebar_scss)


if __name__ == "__main__":
    unittest.main()
