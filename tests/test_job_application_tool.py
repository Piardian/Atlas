import os
import unittest
from unittest.mock import patch

from tools.job_application_tool import (
    _auto_submit_allowed,
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

    def test_submit_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            allowed, reason = _auto_submit_allowed("https://careers.example.com/jobs/123")
        self.assertFalse(allowed)
        self.assertIn("kapalı", reason.lower())

    def test_submit_requires_explicit_domain_allowlist(self):
        env = {
            "ATLAS_JOB_AUTO_SUBMIT": "true",
            "ATLAS_JOB_ALLOWED_DOMAINS": "example.com",
        }
        with patch.dict(os.environ, env, clear=True):
            allowed, _ = _auto_submit_allowed("https://careers.example.com/jobs/123")
            denied, _ = _auto_submit_allowed("https://jobs.other.test/123")
        self.assertTrue(allowed)
        self.assertFalse(denied)

    def test_linkedin_submit_is_blocked_even_if_allowlisted(self):
        env = {
            "ATLAS_JOB_AUTO_SUBMIT": "true",
            "ATLAS_JOB_ALLOWED_DOMAINS": "linkedin.com",
        }
        with patch.dict(os.environ, env, clear=True):
            allowed, reason = _auto_submit_allowed("https://www.linkedin.com/jobs/view/123")
        self.assertFalse(allowed)
        self.assertIn("LinkedIn", reason)


if __name__ == "__main__":
    unittest.main()
