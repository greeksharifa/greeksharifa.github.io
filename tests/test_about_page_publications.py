import unittest
from pathlib import Path


class AboutPagePublicationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = Path("about.md").read_text(encoding="utf-8")

    def test_about_page_lists_new_papers_and_patent(self):
        expected_tokens = [
            "The Impact of Likert Scale Design on Judgment Reliability in Korean and English LLM-as-a-Judge",
            "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12606880",
            "Hybrid State Representation for Video Procedure Planning",
            "https://openaccess.thecvf.com/content/WACV2026/html/Choi_Hybrid_State_Representation_for_Video_Procedure_Planning_WACV_2026_paper.html",
            "Toddler-Inspired Bayesian Learning Method and Computing Apparatus for Performing the Same",
            "US Patent 12,536,454 | US Patent App. 17/467,971",
            "https://patents.justia.com/patent/12536454",
            "Apparatus and Method for Visual Question Answering Based on Self-Questioning Framework",
            "KR Patent App. 10-2025-0108505 | https://doi.org/10.8080/1020250108505",
            "https://doi.org/10.8080/1020250108505",
        ]
        for token in expected_tokens:
            with self.subTest(token=token):
                self.assertIn(token, self.text)

        self.assertEqual(
            1,
            self.text.count(
                "Apparatus and Method for Visual Question Answering Based on Self-Questioning Framework"
            ),
        )

        patent_titles_in_order = [
            "Toddler-Inspired Bayesian Learning Method and Computing Apparatus for Performing the Same",
            "Apparatus and Method for Visual Question Answering Based on Self-Questioning Framework",
            "Method for understanding video story with multi-level character attention, and apparatus for performing the same",
            "Question answering apparatus and method",
        ]
        positions = [self.text.index(title) for title in patent_titles_in_order]
        self.assertEqual(sorted(positions), positions)
        self.assertNotIn("Jan. 27, 2026", self.text)


if __name__ == "__main__":
    unittest.main()
