import unittest
from pathlib import Path


class SidebarContactLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sidebar_html = Path("_includes/sidebar.html").read_text(encoding="utf-8")
        cls.sidebar_scss = Path("_scss/component/_sidebar.scss").read_text(encoding="utf-8")

    def test_follow_me_uses_inline_row_markup(self):
        self.assertIn('class="sidebar-contact-row"', self.sidebar_html)
        self.assertIn('class="sidebar-contact-label">Follow me:</p>', self.sidebar_html)

    def test_contact_links_use_shared_icon_box_size(self):
        expected_tokens = [
            ".sidebar-contact-row",
            ".sidebar-contact-label",
            "width: 1.25rem;",
            "height: 1.25rem;",
            "font-size: 1.25rem;",
            "object-fit: contain;",
        ]
        for token in expected_tokens:
            with self.subTest(token=token):
                self.assertIn(token, self.sidebar_scss)


if __name__ == "__main__":
    unittest.main()
