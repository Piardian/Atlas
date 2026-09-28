# 🏛️ Atlas: Autonomous Multi-Agent Orchestrator CLI

**Atlas** is a modular multi-agent orchestration framework powered by **Google Gemini (AI Studio)** and **CrewAI**, equipped with pluggable execution capabilities:
- **Browser Capability (`tools/browser_tool.py`)**: Real browser automation powered by **Browser Use** & **Playwright Chromium** for autonomous web research, DOM navigation, and structured extraction.
- **Code Execution Capability (`tools/aider_tool.py`)**: Local repository code generation and refactoring powered by **Aider CLI**.
- **Cascading Model & Key Pool Shield (`tools/key_manager.py`)**: Automatic 4-key rotation and multi-tier model fallback on `429 RESOURCE_EXHAUSTED` errors.

---

## 🏗️ Architecture (`feature/browser-use`)

```text
                           ATLAS ORCHESTRATOR
                                   │
                        User / Mission Prompt
                                   │
                                   ▼
                     Dynamic Agent Factory (Lead)
                    ┌──────────────┼──────────────┐
                    │              │              │
                 Research       Browser        Reviewer
                  Agent          Agent          Agent
                    │              │              │
                    └──────┬───────┘              │
                           ▼                      │
                 BrowserAutomationTool            │
              (Browser Use / Playwright)          │
                           │                      │
                     Real Chromium                │
                 ┌─────────┴─────────┐            │
                 │                   │            │
            Web Search        DOM Extraction      │
                 │                   │            │
                 └─────────┬─────────┘            │
                           ▼                      │
                   Structured Result ─────────────┘
                                                  │
                                                  ▼
                                            Final Report
```

### Separation of Concerns
- **Atlas** = Multi-Agent Orchestrator (Mission planning, dynamic workforce synthesis, capability routing, verification).
- **Browser Use (`BrowserAutomationTool`)** = Pluggable Browser Capability (`browser_tool.run(task="...")`).
- **Aider (`AiderExecutionTool`)** = Pluggable Code Execution Capability (`aider_tool.run(instruction="...")`).

---

## 📂 Project Structure

```text
Atlas
├── agents.py
├── crew.py                      # Capability-aware closed-loop orchestrator
├── dynamic_factory.py           # Runtime agent & tool synthesis (browser / aider)
├── spec_expander.py             # Task mode detector (web_research / software_engineering / hybrid)
├── critic_verifier.py           # AST, entrypoint & report verification
├── main.py                      # Unified CLI entrypoint
├── tools/
│   ├── key_manager.py           # 4-key pool & cascading model router
│   ├── aider_tool.py            # Code execution capability
│   └── browser_tool.py          # Abstracted Browser Use / Playwright capability
└── examples/
    └── web_research_demo.py     # Internship PoC: AI Web Research Agent
```

---

## 🚀 Quickstart

### 1. Prerequisites & Installation
- Python 3.11+
- Git & Chromium (via Playwright)

```bash
git clone -b feature/browser-use https://github.com/Piardian/Atlas.git
cd Atlas
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
# Add your GEMINI_USER_*_KEY values into .env
```

### 2. Run Internship PoC: AI Web Research Agent
Researches the top 5 AI companies in Turkey via real browser navigation, extracts their products and official contact details, verifies the findings through the Reviewer Agent, and outputs a structured Markdown report:

```bash
# Full Multi-Agent Orchestration (Atlas -> Research Agent -> Browser Agent -> Reviewer Agent)
python examples/web_research_demo.py

# Watch Chromium live in a visible browser window
python examples/web_research_demo.py --show-browser

# Or via main CLI
python main.py --browser-demo

# Direct Capability Test (calls browser_tool.run(...) directly)
python examples/web_research_demo.py --direct-tool
```

### 3. Run Custom Web Research or Software Engineering Missions
```bash
# Custom Web Research Mission
python main.py --prompt "Türkiye'deki yapay zeka şirketlerini araştır. İlk 5 şirketi bul, web sitelerinden ürünlerini ve iletişim bilgilerini çıkar, tablo halinde raporla." --workspace ./workspace_research

# Custom Software Engineering Mission (Aider Capability)
python main.py --prompt "FastAPI ile JWT tabanlı görev yönetim mikroservisi geliştir" --workspace ./workspace_api
```

---

## 📄 License
MIT License
