<div align="center">

# 🔎 Gemini Search Agent

**A desktop AI assistant that searches the web in real time to answer your questions.**

Powered by Google **Gemini 2.5 Flash** + DuckDuckGo, orchestrated with LangChain/LangGraph,
wrapped in a modern CustomTkinter chat GUI.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-8E75B2?logo=googlegemini&logoColor=white)
![LangChain](https://img.shields.io/badge/Built%20with-LangChain-1C3C3C?logo=langchain&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-0078D6)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📑 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#️-usage)
- [Project Structure](#️-project-structure)
- [Configuration](#️-configuration)
- [Built With](#-built-with)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- 🌐 **Real-time web search** — the agent decides when to query DuckDuckGo and folds the
  results into its answer (no search API key required).
- 💬 **Modern desktop GUI** — chat bubbles, dark theme, and a conversation sidebar.
- 🧠 **Conversation memory** — each follow-up sees the full thread, so "tell me more about
  that" just works.
- 💾 **Persistent history** — chats are saved to `conversations.json` and restored on the
  next launch. Browse, reopen, or delete past conversations from the sidebar.
- 📋 **Copy button** on every message for quick clipboard access.
- 🔢 **Token usage** shown per message and as a running conversation total.
- 🖥️ **CLI mode** for a quick terminal chat without the GUI.

---

## 📸 Demo

> _Add a screenshot or GIF of the app here._
>
> ```
> docs/screenshot.png
> ```

<!-- ![Gemini Search Agent screenshot](docs/screenshot.png) -->

---

## 📦 Requirements

- **Python 3.10+** (with Tcl/Tk — bundled with the standard [python.org](https://www.python.org/downloads/) installer)
- A free **Google AI Studio API key** → https://aistudio.google.com/apikey

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/gemini-search-agent.git
cd gemini-search-agent

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirement.txt

# 4. Add your API key — copy the template and edit it:
cp .env.example .env        # Windows: copy .env.example .env
# then set GOOGLE_API_KEY=... inside .env
```

Without a key the app stops with a short message telling you exactly what to
create — the GUI shows it in a dialog, the CLI prints it and exits 1.

> ⚠️ **Never commit your `.env`.** It contains your secret API key. This repo's
> `.gitignore` already excludes it.

---

## ▶️ Usage

### Desktop app (recommended)

```bash
python app.py
```

| Action            | How                              |
| ----------------- | -------------------------------- |
| Send a message    | **Enter**                        |
| New line          | **Shift + Enter**                |
| New conversation  | **＋ New Chat** button           |
| Copy a message    | **📋 Copy** under any bubble     |
| Delete a chat     | **🗑** next to a sidebar item    |

Token counts appear under each reply and as a running total in the status bar.

### Command-line chat

```bash
python main.py
```

Type your questions at the prompt; enter `exit` or `quit` to stop.

---

## 🗂️ Project Structure

| File                 | Purpose                                                           |
| -------------------- | ---------------------------------------------------------------- |
| `app.py`             | CustomTkinter desktop GUI (chat, sidebar, history, token usage). |
| `agent_core.py`      | Shared agent: Gemini LLM + DuckDuckGo tool, `ask()` entry point. |
| `main.py`            | Command-line chat loop using the same core.                      |
| `conversations.json` | Auto-generated saved chat history (git-ignored).                 |
| `requirement.txt`    | Python dependencies.                                             |
| `.env`               | Your `GOOGLE_API_KEY` (you create this — git-ignored).           |

---

## ⚙️ Configuration

The model and behavior are set in `agent_core.py`:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # swap for another Gemini model if you like
    temperature=0,              # raise for more creative answers
)
```

You can also edit the `system_prompt` passed to `create_agent` to change the
assistant's persona or instructions.

---

## 🧰 Built With

- [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) — agent orchestration
- [langchain-google-genai](https://pypi.org/project/langchain-google-genai/) — Gemini integration
- [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) — free web search tool
- [CustomTkinter](https://customtkinter.tomschimansky.com/) — modern desktop UI

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo and create your branch: `git checkout -b feature/my-feature`
2. Commit your changes: `git commit -m "Add my feature"`
3. Push to the branch: `git push origin feature/my-feature`
4. Open a Pull Request.

Please open an issue first to discuss any major changes.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 🔒 Security Notes

- If you ever expose your API key, rotate it at https://aistudio.google.com/apikey.
- DuckDuckGo search is rate-limited; heavy use may occasionally return throttling errors.

<div align="center">

⭐ If you find this project useful, consider giving it a star!

</div>

---

## 🧪 Tests

```bash
python -m unittest -v
```

30 tests, no API key and no network calls.

- `test_agent_core.py` (19) — the pure helpers: Gemini's block-list content
  shape, token totals, API-key validation, and the search tool with `ddgs`
  stubbed.
- `test_integration.py` (11) — the real agent graph and the real Tk widgets,
  with only the call to Gemini replaced: a full tool-call round trip (model
  asks to search → tool runs → model answers, with usage summed across *both*
  model calls), the GUI's threaded send, deleting a conversation while its
  reply is in flight, the missing-key dialog, and conversation persistence
  against corrupt and non-ASCII files.

The GUI cases skip automatically where no display is available, so the suite
runs on headless CI.
