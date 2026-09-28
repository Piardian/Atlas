# 🏛️ Atlas — Multi-Agent Web Research & Browser Automation System (`CrewAI + Browser Use + Aider + Gemini`)

**Atlas** is an autonomous multi-agent orchestration system built on **CrewAI**, **Browser Use**, **Aider**, and **Google Gemini (AI Studio)**. It dynamically synthesizes specialized agent teams at runtime and equips them with real browser automation and local code execution capabilities:

- **Primary Browser Automation (`tools/browser_tool.py`)**: Powered by **`BrowserUseBackend` (`browser-use` + Chromium)** as the default engine, allowing autonomous agents to navigate live websites, inspect DOM elements, and extract structured corporate/product intelligence, with `PlaywrightSmartBackend` as a resilient fallback.
- **Code Execution Capability (`tools/aider_tool.py`)**: Local repository code generation, refactoring, and automated verification powered by **Aider CLI**.
- **Cascading Model & 4-Key Pool Shield (`tools/key_manager.py`)**: Step-level automatic rotation across a 4-key Gemini API pool (`AtlasRotatingChatGoogle` & `SmartFallbackRouter`) with instant cooldown handling on `429 RESOURCE_EXHAUSTED` and `503 UNAVAILABLE` spikes.

---

## 🏗️ End-to-End Architecture

```text
Atlas Orchestrator
  │
  ▼
CrewAI (Dynamic Agent Factory & Spec Expander)
  │
  ├──► Browser Research & Web Extraction Agent
  │      │
  │      ▼
  │    BrowserAutomationTool (Default: BrowserUseBackend)
  │      │
  │      ▼
  │    Browser Use Agent (AtlasRotatingChatGoogle + 4-Key Pool)
  │      │
  │      ▼
  │    Real Chromium Browser (Live Web Navigation & DOM Extraction)
  │      │
  │      ▼
  │    Structured Findings + Telemetry Proof (browser_telemetry.jsonl)
  │
  └──► Lead Research Reviewer & Structured Report Synthesizer
         │
         ▼
       Verified Final Markdown Report (TR_AI_COMPANIES_REPORT.md / WEB_RESEARCH_REPORT.md)
```

### Separation of Concerns
- **Atlas (`crew.py`, `dynamic_factory.py`)**: Mission analysis (`ARCH_SPEC`), dynamic agent/task synthesis, capability routing, and Stage 4 verification.
- **Browser Capability (`BrowserAutomationTool`)**: Pluggable tool interface (`browser_tool.run(task="...")`) that executes `BrowserUseBackend` (`browser-use`) first and logs verifiable run telemetry (`backend_used`, `visited_urls`, `elapsed_seconds`, `fallback_triggered`) to `browser_telemetry.jsonl`.
- **Code Capability (`AiderExecutionTool`)**: Pluggable tool interface (`aider_tool.run(instruction="...")`) for autonomous software engineering missions.

---

## 📂 Project Structure

```text
Atlas
├── agents.py                    # Static baseline agent definitions
├── crew.py                      # 4-stage closed-loop orchestrator (Spec -> Blueprint -> Execution -> Verifier)
├── dynamic_factory.py           # Runtime agent & capability synthesis (Browser Use / Aider)
├── spec_expander.py             # Task mode detector (web_research / software_engineering / hybrid)
├── critic_verifier.py           # AST syntax, entrypoint & report verification
├── main.py                      # Unified CLI entrypoint
├── tools/
│   ├── key_manager.py           # 4-key Gemini pool & cascading model router
│   ├── aider_tool.py            # Aider CLI code execution capability
│   └── browser_tool.py          # BrowserUseBackend (primary) & PlaywrightSmartBackend (fallback)
├── examples/
│   └── web_research_demo.py     # End-to-end Web Research & Browser Automation PoC
└── workspace_research/
    ├── TR_AI_COMPANIES_REPORT.md
    ├── WEB_RESEARCH_REPORT.md
    └── browser_telemetry.jsonl  # Verifiable execution telemetry log
```

---

## 🚀 Quickstart

### 1. Prerequisites & Installation
- **Python 3.11+** (tested with Python 3.12 and `browser-use 0.11.13`)
- **Git & Chromium**

```bash
git clone -b feature/browser-use https://github.com/Piardian/Atlas.git
cd Atlas
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
# Add your GEMINI_USER_1_KEY ... GEMINI_USER_4_KEY values into .env
```

### 2. Run the End-to-End Web Research & Browser Automation Demo
Researches the top 5 AI companies in Turkey (`vispera.co`, `cbot.ai`, `tazi.ai`, `intenseye.com`, `sestek.com`) using real Chromium navigation via `BrowserUseBackend`, extracts their core AI products and official contact details, verifies the output through the Reviewer Agent, and writes the final Markdown report and telemetry log:

```bash
# Full Multi-Agent Pipeline (Atlas -> CrewAI -> Browser Research Agent -> Browser Use -> Chromium -> Reviewer -> Final Report)
python -X utf8 examples/web_research_demo.py --show-browser

# Headless mode
python -X utf8 examples/web_research_demo.py

# Or via main CLI
python -X utf8 main.py --browser-demo --show-browser

# Direct Capability Test (calls BrowserAutomationTool -> BrowserUseBackend directly)
python -X utf8 examples/web_research_demo.py --direct-tool --show-browser
```

### 3. Run Custom Web Research or Software Engineering Missions
```bash
# Custom Web Research Mission (Browser Use Capability)
python -X utf8 main.py --show-browser --prompt "Türkiye'deki yapay zeka şirketlerini araştır. İlk 5 şirketi bul, web sitelerinden ürünlerini ve iletişim bilgilerini çıkar, tablo halinde raporla." --workspace ./workspace_research

# Custom Software Engineering Mission (Aider Capability)
python -X utf8 main.py --prompt "FastAPI ile JWT tabanlı görev yönetim mikroservisi geliştir" --workspace ./workspace_api
```

---

## 📊 Verifiable Execution Telemetry (`browser_telemetry.jsonl`)

Every browser mission appends a structured JSONL record to `workspace_research/browser_telemetry.jsonl` proving which backend executed the task, whether any fallback was triggered, how long it took, and every URL visited:

```json
{
  "timestamp": "2026-09-28T18:58:00Z",
  "agent_role": "Browser Research & Web Extraction Agent",
  "backend_used": "browser-use (gemini-3.5-flash-lite)",
  "success": true,
  "visited_urls": [
    "https://vispera.co/",
    "https://www.cbot.ai/",
    "https://tazi.ai/",
    "https://www.intenseye.com/",
    "https://www.sestek.com/"
  ],
  "elapsed_seconds": 64.2,
  "fallback_triggered": false
}
```

---

## 📄 License
MIT License
