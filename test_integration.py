"""Integration tests: the agent graph end to end, and the GUI's async flow.

`test_agent_core.py` covers the pure helpers in isolation. These drive the real
LangGraph agent and the real Tk widgets, replacing only the network call to
Gemini — so they catch the wiring mistakes a unit test cannot: a tool that is
never reached, usage counted from one model call instead of all of them, or a
reply that never makes it back from the worker thread to the UI.

Run with:  python -m unittest -v

No API key and no Gemini traffic. The GUI cases skip automatically where no
display is available (headless CI).
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-used-for-network-calls")

import agent_core  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402


class ToolCallingFake(GenericFakeChatModel):
    """A fake chat model a tool-calling agent will accept.

    GenericFakeChatModel raises NotImplementedError from bind_tools, so the
    agent cannot be built with it as-is. Returning self is enough: the graph,
    the tool and the message flow stay real, and only the HTTP call is gone.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def _install_fake_agent(*messages):
    """Point agent_core at an agent driven by the given scripted replies."""
    fake = ToolCallingFake(messages=iter(messages))
    agent_core._agent = agent_core.create_agent(
        fake, [agent_core.web_search], system_prompt=agent_core.SYSTEM_PROMPT
    )


def _has_display():
    """Whether a Tk display can be opened here."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.destroy()
        return True
    except Exception:  # noqa: BLE001 - any failure means no usable display
        return False


HAS_DISPLAY = _has_display()


class AgentRoundTripTests(unittest.TestCase):
    """The full loop: model asks for a search, gets one, then answers."""

    def tearDown(self):
        agent_core._agent = None

    def test_tool_call_then_answer_with_usage_from_every_call(self):
        wants_search = AIMessage(
            content=[],
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "capital of France"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
        )
        answers = AIMessage(
            # Gemini's block-list shape, which extract_text flattens.
            content=[{"type": "text", "text": "Paris is the capital of France."}],
            usage_metadata={
                "input_tokens": 400,
                "output_tokens": 30,
                "total_tokens": 430,
            },
        )
        _install_fake_agent(wants_search, answers)

        with mock.patch("ddgs.DDGS") as ddgs:
            ddgs.return_value.text.return_value = [
                {"title": "Paris", "body": "Capital of France.",
                 "href": "https://example.com/paris"}
            ]
            text, usage = agent_core.ask(
                [{"role": "user", "content": "What is the capital of France?"}]
            )
            ddgs.return_value.text.assert_called_once()

        self.assertEqual(text, "Paris is the capital of France.")
        # The whole point: a turn spans several model calls, and the cost is
        # all of them. Reading only the last message would report 430.
        self.assertEqual(usage["total_tokens"], 550)
        self.assertEqual(usage["input_tokens"], 500)
        self.assertEqual(usage["output_tokens"], 50)

    def test_a_turn_with_no_tool_call_still_works(self):
        _install_fake_agent(
            AIMessage(
                content="Two plus two is four.",
                usage_metadata={
                    "input_tokens": 8,
                    "output_tokens": 6,
                    "total_tokens": 14,
                },
            )
        )
        text, usage = agent_core.ask([{"role": "user", "content": "2+2?"}])
        self.assertEqual(text, "Two plus two is four.")
        self.assertEqual(usage["total_tokens"], 14)


@unittest.skipUnless(HAS_DISPLAY, "no display available for Tk")
class GuiTests(unittest.TestCase):
    """The GUI, driven through a real event loop."""

    def setUp(self):
        import app

        self.app_module = app
        # Keep the suite from reading or writing a developer's real chat log.
        self._history = app.HISTORY_FILE
        app.HISTORY_FILE = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_test_conversations.json"
        )

    def tearDown(self):
        if os.path.exists(self.app_module.HISTORY_FILE):
            os.remove(self.app_module.HISTORY_FILE)
        self.app_module.HISTORY_FILE = self._history
        agent_core._agent = None

    @staticmethod
    def _close(win):
        """Destroy a window, tolerating Tk's teardown complaints.

        CustomTkinter keeps its own `after` timers (a DPI-scaling check), so on
        destroy Tk may print `invalid command name ...` for a callback whose
        widget has gone. It is noise, not a failure — and cancelling those
        timers first turns it into a real error, because some ids have already
        fired by the time we look.
        """
        try:
            win.destroy()
        except Exception:  # noqa: BLE001 - teardown races are not test failures
            pass

    def _run_until_idle(self, win, timeout_ticks=200):
        """Pump the real mainloop until the agent finishes, or give up.

        A real mainloop, not repeated update() calls: `after` callbacks
        scheduled from the worker thread are not reliably dispatched by
        update(), which looks exactly like a hung reply.
        """
        state = {}

        def watch(ticks=0):
            if not win.busy:
                win.quit()
            elif ticks > timeout_ticks:
                state["timed_out"] = True
                win.quit()
            else:
                win.after(100, lambda: watch(ticks + 1))

        win.after(50, lambda: watch(0))
        win.mainloop()
        return state

    def test_window_builds_and_opens_a_conversation(self):
        win = self.app_module.ChatApp()
        try:
            win.update()
            self.assertEqual(win.title(), "Gemini Search Agent")
            self.assertIsNotNone(win.current_id)
            self.assertFalse(win.busy)
            self.assertEqual(win.send_btn.cget("state"), "normal")
        finally:
            self._close(win)

    def test_send_puts_the_reply_on_screen_and_clears_busy(self):
        _install_fake_agent(
            AIMessage(
                content=[{"type": "text", "text": "Hello from the agent."}],
                usage_metadata={
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                },
            )
        )
        win = self.app_module.ChatApp()
        try:
            win.update()
            win.entry.insert("1.0", "say hello")
            win._on_send()
            self.assertTrue(win.busy, "the send button should latch immediately")

            state = self._run_until_idle(win)
            self.assertFalse(state.get("timed_out"), "reply never reached the UI")

            messages = win._current()["messages"]
            self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
            self.assertEqual(messages[-1]["content"], "Hello from the agent.")
            self.assertEqual(messages[-1]["usage"]["total_tokens"], 18)
            self.assertFalse(win.busy)
        finally:
            self._close(win)

    def test_deleting_a_conversation_mid_reply_does_not_wedge_the_ui(self):
        # The reply comes back for a conversation that no longer exists. If
        # busy were left set, the send button would stay dead for good.
        _install_fake_agent(
            AIMessage(
                content="answer",
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            )
        )
        win = self.app_module.ChatApp()
        try:
            win.update()
            win.entry.insert("1.0", "question")
            win._on_send()
            win._delete_conversation(win.current_id)

            state = self._run_until_idle(win)
            self.assertFalse(state.get("timed_out"), "UI left permanently busy")
            self.assertFalse(win.busy)
            self.assertEqual(win.send_btn.cget("state"), "normal")
        finally:
            self._close(win)

    def test_missing_key_shows_a_dialog_and_exits(self):
        shown = {}
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "tkinter.messagebox.showerror",
            side_effect=lambda title, message: shown.update(message=message),
        ):
            agent_core._agent = None
            with self.assertRaises(SystemExit) as ctx:
                self.app_module.main()

        self.assertEqual(ctx.exception.code, 1)
        # A dialog the user can act on, not a pydantic traceback.
        self.assertIn("API key", shown.get("message", ""))
        self.assertIn(".env", shown.get("message", ""))


class PersistenceTests(unittest.TestCase):
    """Saved conversations must survive bad input rather than take the app down."""

    def setUp(self):
        import app

        self.app_module = app
        self._history = app.HISTORY_FILE
        app.HISTORY_FILE = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_test_conversations.json"
        )

    def tearDown(self):
        if os.path.exists(self.app_module.HISTORY_FILE):
            os.remove(self.app_module.HISTORY_FILE)
        self.app_module.HISTORY_FILE = self._history

    def _write(self, text):
        with open(self.app_module.HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write(text)

    def test_missing_file_is_an_empty_history(self):
        self.assertEqual(self.app_module.load_conversations(), [])

    def test_round_trip(self):
        convos = [
            {"id": "1", "title": "T", "created": 0,
             "messages": [{"role": "user", "content": "hi"}]}
        ]
        self.app_module.save_conversations(convos)
        self.assertEqual(self.app_module.load_conversations(), convos)

    def test_corrupt_json_does_not_raise(self):
        # A half-written file after a crash must not stop the app starting.
        self._write("{not valid json")
        self.assertEqual(self.app_module.load_conversations(), [])

    def test_wrong_shape_is_rejected(self):
        self._write('{"not": "a list"}')
        self.assertEqual(self.app_module.load_conversations(), [])

    def test_unicode_survives_a_round_trip(self):
        convos = [
            {"id": "2", "title": "日本語 café 🎉", "created": 0,
             "messages": [{"role": "user", "content": "emoji 🚀 and ünïcode"}]}
        ]
        self.app_module.save_conversations(convos)
        self.assertEqual(self.app_module.load_conversations(), convos)


if __name__ == "__main__":
    unittest.main()
