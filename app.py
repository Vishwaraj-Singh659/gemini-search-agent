"""Desktop GUI for the Gemini search agent.

A modern chat interface (CustomTkinter) with:
  * a sidebar listing past conversations,
  * message bubbles for the active chat,
  * background threading so the window never freezes while the agent thinks,
  * conversation history persisted to conversations.json.

Run with:  python app.py
"""

import json
import os
import threading
import time
import uuid

import customtkinter as ctk

from agent_core import ask

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations.json")


def load_conversations():
    """Return the list of saved conversations (oldest first)."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_conversations(conversations):
    """Persist all conversations to disk."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Could not save history: {e}")


def make_title(text):
    """Build a short sidebar title from the first user message."""
    text = " ".join(text.split())
    return (text[:32] + "…") if len(text) > 32 else (text or "New Chat")


# ---------------------------------------------------------------------------
# Colors / theme
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

USER_BUBBLE = "#2563eb"
BOT_BUBBLE = "#343541"
SIDEBAR_BG = "#1a1b26"
ACTIVE_ITEM = "#2a2d3e"


class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gemini Search Agent")
        self.geometry("1040x700")
        self.minsize(820, 560)

        # Each conversation: {"id", "title", "created", "messages": [{role, content}]}
        self.conversations = load_conversations()
        self.current_id = None
        self.busy = False

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_chat_area()

        self._refresh_sidebar()
        if self.conversations:
            self._open_conversation(self.conversations[-1]["id"])
        else:
            self._new_conversation()

    # ----- layout -------------------------------------------------------
    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=270, corner_radius=0, fg_color=SIDEBAR_BG)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(2, weight=1)
        sidebar.grid_propagate(False)

        title = ctk.CTkLabel(
            sidebar, text="  💬  Conversations",
            font=ctk.CTkFont(size=16, weight="bold"), anchor="w",
        )
        title.grid(row=0, column=0, sticky="ew", padx=16, pady=(18, 8))

        new_btn = ctk.CTkButton(
            sidebar, text="＋  New Chat", height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._new_conversation,
        )
        new_btn.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        self.sidebar_list = ctk.CTkScrollableFrame(sidebar, fg_color="transparent")
        self.sidebar_list.grid(row=2, column=0, sticky="nsew", padx=6)
        self.sidebar_list.grid_columnconfigure(0, weight=1)

        footer = ctk.CTkLabel(
            sidebar, text="Powered by Gemini 2.5 Flash",
            font=ctk.CTkFont(size=11), text_color="#6b7280",
        )
        footer.grid(row=3, column=0, pady=12)

    def _build_chat_area(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="#202123")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.chat_frame = ctk.CTkScrollableFrame(main, fg_color="#202123")
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        input_bar = ctk.CTkFrame(main, fg_color="#202123")
        input_bar.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))
        input_bar.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkTextbox(input_bar, height=52, wrap="word",
                                    font=ctk.CTkFont(size=14))
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Shift-Return>", lambda e: None)

        self.send_btn = ctk.CTkButton(input_bar, text="Send", width=90, height=52,
                                      font=ctk.CTkFont(size=14, weight="bold"),
                                      command=self._on_send)
        self.send_btn.grid(row=0, column=1)

        self.status = ctk.CTkLabel(main, text="", font=ctk.CTkFont(size=12),
                                   text_color="#9ca3af")
        self.status.grid(row=2, column=0, sticky="w", padx=18, pady=(0, 6))

    # ----- conversations -------------------------------------------------
    def _current(self):
        for c in self.conversations:
            if c["id"] == self.current_id:
                return c
        return None

    def _new_conversation(self):
        convo = {
            "id": uuid.uuid4().hex,
            "title": "New Chat",
            "created": time.time(),
            "messages": [],
        }
        self.conversations.append(convo)
        self.current_id = convo["id"]
        self._refresh_sidebar()
        self._render_messages()
        self.entry.focus_set()

    def _open_conversation(self, convo_id):
        self.current_id = convo_id
        self._refresh_sidebar()
        self._render_messages()

    def _delete_conversation(self, convo_id):
        self.conversations = [c for c in self.conversations if c["id"] != convo_id]
        save_conversations(self.conversations)
        if convo_id == self.current_id:
            if self.conversations:
                self.current_id = self.conversations[-1]["id"]
            else:
                self._new_conversation()
                return
        self._refresh_sidebar()
        self._render_messages()

    def _refresh_sidebar(self):
        for w in self.sidebar_list.winfo_children():
            w.destroy()
        for i, convo in enumerate(reversed(self.conversations)):
            active = convo["id"] == self.current_id
            row = ctk.CTkFrame(self.sidebar_list,
                               fg_color=ACTIVE_ITEM if active else "transparent",
                               corner_radius=8)
            row.grid(row=i, column=0, sticky="ew", pady=3, padx=2)
            row.grid_columnconfigure(0, weight=1)

            btn = ctk.CTkButton(
                row, text=convo["title"], anchor="w", height=34,
                fg_color="transparent", hover_color="#34374a",
                font=ctk.CTkFont(size=13),
                command=lambda cid=convo["id"]: self._open_conversation(cid),
            )
            btn.grid(row=0, column=0, sticky="ew", padx=(4, 0))

            dele = ctk.CTkButton(
                row, text="🗑", width=30, height=34, fg_color="transparent",
                hover_color="#7f1d1d", text_color="#9ca3af",
                command=lambda cid=convo["id"]: self._delete_conversation(cid),
            )
            dele.grid(row=0, column=1, padx=(0, 2))

    # ----- chat rendering ------------------------------------------------
    def _render_messages(self):
        for w in self.chat_frame.winfo_children():
            w.destroy()
        convo = self._current()
        if not convo or not convo["messages"]:
            hint = ctk.CTkLabel(
                self.chat_frame,
                text="Ask me anything — I can search the web for real-time info.",
                font=ctk.CTkFont(size=15), text_color="#6b7280",
            )
            hint.grid(row=0, column=0, pady=60)
            self.status.configure(text="")
            return
        for msg in convo["messages"]:
            self._add_bubble(msg["role"], msg["content"], msg.get("usage"))
        self._update_total_usage(convo)

    def _add_bubble(self, role, text, usage=None):
        is_user = role == "user"
        container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        container.grid(sticky="ew", pady=6, padx=4)
        container.grid_columnconfigure(0, weight=1)

        bubble = ctk.CTkFrame(
            container, corner_radius=14,
            fg_color=USER_BUBBLE if is_user else BOT_BUBBLE,
        )
        bubble.grid(row=0, column=0, sticky="e" if is_user else "w", padx=6)

        label = ctk.CTkLabel(
            bubble, text=text, justify="left", wraplength=620,
            font=ctk.CTkFont(size=14), text_color="#ffffff",
        )
        label.grid(row=0, column=0, padx=14, pady=10)

        # Footer row: sender name + a Copy button (labels aren't selectable,
        # so the button is how you get the text onto the clipboard).
        footer = ctk.CTkFrame(container, fg_color="transparent")
        footer.grid(row=1, column=0, sticky="e" if is_user else "w", padx=12)

        who = ctk.CTkLabel(
            footer, text="You" if is_user else "Agent",
            font=ctk.CTkFont(size=10), text_color="#6b7280",
        )
        who.grid(row=0, column=0, padx=(0, 6))

        copy_btn = ctk.CTkButton(
            footer, text="📋 Copy", width=64, height=22,
            font=ctk.CTkFont(size=10), fg_color="transparent",
            hover_color="#34374a", text_color="#9ca3af",
        )
        copy_btn.configure(command=lambda b=copy_btn: self._copy_text(text, b))
        copy_btn.grid(row=0, column=1)

        if usage:
            tok = ctk.CTkLabel(
                footer,
                text=(f"· {usage.get('total_tokens', 0)} tok "
                      f"(in {usage.get('input_tokens', 0)} / "
                      f"out {usage.get('output_tokens', 0)})"),
                font=ctk.CTkFont(size=10), text_color="#6b7280",
            )
            tok.grid(row=0, column=2, padx=(6, 0))

        self.after(30, self._scroll_to_bottom)
        return label

    def _copy_text(self, text, button):
        """Copy a message to the system clipboard and flash feedback."""
        self.clipboard_clear()
        self.clipboard_append(text)
        button.configure(text="✓ Copied")
        self.after(1200, lambda: button.configure(text="📋 Copy"))

    def _scroll_to_bottom(self):
        try:
            self.chat_frame._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    # ----- sending -------------------------------------------------------
    def _on_return(self, event):
        # Enter sends; Shift+Enter inserts a newline (handled by default binding).
        if event.state & 0x0001:  # Shift held
            return
        self._on_send()
        return "break"

    def _on_send(self):
        if self.busy:
            return
        text = self.entry.get("1.0", "end").strip()
        if not text:
            return
        self.entry.delete("1.0", "end")

        convo = self._current()
        convo["messages"].append({"role": "user", "content": text}) # type: ignore
        if convo["title"] == "New Chat": # type: ignore
            convo["title"] = make_title(text) # type: ignore
        save_conversations(self.conversations)
        self._refresh_sidebar()
        self._add_bubble("user", text)

        self._set_busy(True, "Agent is thinking…")
        # Send only role/content to the agent (strip UI-only keys like usage).
        history = [{"role": m["role"], "content": m["content"]}
                   for m in convo["messages"]] # type: ignore
        threading.Thread(target=self._run_agent, args=(convo["id"], history), # type: ignore
                         daemon=True).start()

    def _run_agent(self, convo_id, history):
        usage = None
        try:
            answer, usage = ask(history)
        except Exception as e:
            answer = f"⚠️ An error occurred: {e}"
        # Hand the result back to the UI thread.
        self.after(0, lambda: self._on_answer(convo_id, answer, usage))

    def _on_answer(self, convo_id, answer, usage):
        convo = next((c for c in self.conversations if c["id"] == convo_id), None)
        if convo is not None:
            msg = {"role": "assistant", "content": answer}
            if usage:
                msg["usage"] = usage
            convo["messages"].append(msg)
            save_conversations(self.conversations)
            if convo_id == self.current_id:
                self._add_bubble("assistant", answer, usage)
            self._set_busy(False, "")
            self._update_total_usage(convo)
            return
        self._set_busy(False, "")

    def _update_total_usage(self, convo):
        """Show cumulative token usage for the conversation in the status bar."""
        total = sum(m.get("usage", {}).get("total_tokens", 0)
                    for m in convo["messages"])
        self.status.configure(
            text=f"Conversation tokens: {total:,}" if total else ""
        )

    def _set_busy(self, busy, status_text):
        self.busy = busy
        self.status.configure(text=status_text)
        self.send_btn.configure(state="disabled" if busy else "normal",
                                text="…" if busy else "Send")


if __name__ == "__main__":
    ChatApp().mainloop()
