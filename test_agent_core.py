"""Tests for the pure helpers in agent_core.

Run with:  python -m unittest -v

Deliberately no network and no API key: importing agent_core used to require
one, which is why none of this could be tested before.
"""

import os
import unittest
from unittest import mock

import agent_core
from agent_core import MissingAPIKeyError, extract_text, _sum_usage


class FakeMessage:
    """Stands in for a LangChain message carrying usage metadata."""

    def __init__(self, usage_metadata=None):
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata


class ExtractTextTests(unittest.TestCase):
    def test_plain_string_is_trimmed(self):
        self.assertEqual(extract_text("  hello  "), "hello")

    def test_gemini_block_list_is_flattened(self):
        # Gemini returns content as blocks rather than a string, which is the
        # whole reason this helper exists.
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        self.assertEqual(extract_text(content), "first\nsecond")

    def test_empty_blocks_are_dropped(self):
        content = [{"text": "kept"}, {"text": ""}, {"text": "also kept"}]
        self.assertEqual(extract_text(content), "kept\nalso kept")

    def test_block_without_a_text_key_does_not_raise(self):
        self.assertEqual(extract_text([{"type": "image"}, {"text": "x"}]), "x")

    def test_non_dict_blocks_are_stringified(self):
        self.assertEqual(extract_text(["a", 1]), "a\n1")

    def test_empty_list_yields_empty_string(self):
        self.assertEqual(extract_text([]), "")

    def test_unexpected_type_is_stringified(self):
        self.assertEqual(extract_text(42), "42")
        self.assertEqual(extract_text(None), "None")


class SumUsageTests(unittest.TestCase):
    def test_usage_is_summed_across_every_model_call(self):
        # An agent turn makes one model call per tool round-trip; counting only
        # the last would under-report the cost of the turn.
        messages = [
            FakeMessage({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
            FakeMessage({"input_tokens": 20, "output_tokens": 7, "total_tokens": 27}),
        ]
        self.assertEqual(
            _sum_usage(messages),
            {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
        )

    def test_messages_without_usage_are_skipped(self):
        messages = [
            FakeMessage(),  # e.g. a tool result, which carries no usage
            FakeMessage({"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}),
        ]
        self.assertEqual(_sum_usage(messages)["total_tokens"], 7)

    def test_partial_metadata_defaults_to_zero(self):
        messages = [FakeMessage({"input_tokens": 5})]
        self.assertEqual(
            _sum_usage(messages),
            {"input_tokens": 5, "output_tokens": 0, "total_tokens": 0},
        )

    def test_no_messages_yields_zeros(self):
        self.assertEqual(
            _sum_usage([]),
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )


class ApiKeyTests(unittest.TestCase):
    def setUp(self):
        # get_agent caches; clear it so each test builds fresh.
        agent_core._agent = None

    def tearDown(self):
        agent_core._agent = None

    def test_missing_key_raises_a_message_worth_showing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingAPIKeyError) as ctx:
                agent_core._require_api_key()
        message = str(ctx.exception)
        self.assertIn("GOOGLE_API_KEY", message)
        self.assertIn(".env", message)

    def test_blank_key_counts_as_missing(self):
        # An empty or whitespace value in .env is a typo, not a credential.
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "   "}, clear=True):
            with self.assertRaises(MissingAPIKeyError):
                agent_core._require_api_key()

    def test_gemini_api_key_is_accepted_as_an_alternative(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "abc123"}, clear=True):
            self.assertEqual(agent_core._require_api_key(), "abc123")

    def test_key_is_trimmed(self):
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": " abc123 "}, clear=True):
            self.assertEqual(agent_core._require_api_key(), "abc123")


class WebSearchTests(unittest.TestCase):
    """The tool wrapper, with the network stubbed out."""

    def test_results_are_formatted_with_their_sources(self):
        fake = [
            {"title": "T1", "body": "B1", "href": "https://example.com/1"},
            {"title": "T2", "body": "B2", "href": "https://example.com/2"},
        ]
        with mock.patch("ddgs.DDGS") as ddgs:
            ddgs.return_value.text.return_value = fake
            out = agent_core.web_search.invoke({"query": "anything"})
        self.assertIn("T1", out)
        self.assertIn("https://example.com/2", out)
        # Sources matter: the system prompt asks the model to cite them.
        self.assertEqual(out.count("Source:"), 2)

    def test_missing_fields_do_not_raise(self):
        with mock.patch("ddgs.DDGS") as ddgs:
            ddgs.return_value.text.return_value = [{}]
            out = agent_core.web_search.invoke({"query": "anything"})
        self.assertIn("Untitled", out)

    def test_no_results_reports_that_clearly(self):
        with mock.patch("ddgs.DDGS") as ddgs:
            ddgs.return_value.text.return_value = []
            out = agent_core.web_search.invoke({"query": "zzz"})
        self.assertIn("No results", out)

    def test_a_search_failure_is_returned_as_text_not_raised(self):
        # The agent should get the chance to answer anyway or retry; one flaky
        # search must not abort the whole run.
        with mock.patch("ddgs.DDGS") as ddgs:
            ddgs.return_value.text.side_effect = RuntimeError("rate limited")
            out = agent_core.web_search.invoke({"query": "anything"})
        self.assertIn("Search failed", out)
        self.assertIn("rate limited", out)


if __name__ == "__main__":
    unittest.main()
