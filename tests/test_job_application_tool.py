import os
import unittest
from unittest.mock import patch

from tools.job_application_tool import (
    _auto_submit_allowed,
    _is_linkedin_url,
    _domain_matches,
    _flatten_profile,
    _score_text,
)


class JobApplicationToolHelpersTests(unittest.TestCase):
    def test_flatten_profile_preserves_nested_fields(self):
        profile = _flatten_profile({
            "name": {"first": "Ada", "last": "Lovelace"},
            "skills": ["Python", "SQL"],
        })
        self.assertEqual(profile["name.first"], "Ada")
        self.assertEqual(profile["name.last"], "Lovelace")
        self.assertEqual(profile["skills"], "Python, SQL")

    def test_score_increases_with_relevant_terms(self):
        profile = {"skills": "Python SQL Power BI data analysis"}
        relevant = _score_text(profile, "Junior data analyst Python SQL Power BI")
        unrelated = _score_text(profile, "Warehouse forklift night shift")
        self.assertGreater(relevant, unrelated)

    def test_domain_match_does_not_match_lookalike_host(self):
        self.assertTrue(_domain_matches("jobs.example.com", "example.com"))
        self.assertFalse(_domain_matches("example.com.attacker.test", "example.com"))

    def test_submit_allowed_in_test_mode(self):
        allowed, reason = _auto_submit_allowed("https://careers.example.com/jobs/123")
        self.assertTrue(allowed)
        self.assertIn("onaylandı", reason.lower())

    def test_submit_requires_valid_url(self):
        allowed, reason = _auto_submit_allowed("")
        self.assertFalse(allowed)

    def test_linkedin_urls_are_recognized(self):
        self.assertTrue(_is_linkedin_url("https://www.linkedin.com/jobs/view/123"))
        self.assertTrue(_is_linkedin_url("https://careers.linkedin.com/jobs/123"))
        self.assertFalse(_is_linkedin_url("https://linkedin.com.example.test/jobs/123"))


if __name__ == "__main__":
    unittest.main()
