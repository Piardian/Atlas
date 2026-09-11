# 🏛️ Atlas: Autonomous Multi-Agent Orchestrator CLI (v3.0)

**Atlas** is an advanced AI engineering and task orchestration framework powered by Google Gemini (AI Studio), CrewAI, and Aider. It features runtime dynamic agent synthesis and high-resilience API rate-limit protection.

---

## 🎯 Key Capabilities

### 1. 🧠 Dynamic Agent Factory (Dinamik Ajan Fabrikası)
- **Zero Static Limits**: Rather than relying on hardcoded roles, the **Lead Architect** dynamically generates specialized agents tailored on-the-fly to the specific task domain (Finance/SMC, Web Full-Stack, Quantitative Engineering, Cybersecurity, Data Pipelines, etc.).
- **Role Isolation**: Each generated agent receives custom goals, domain backstories, and precise file/tool boundaries.

### 2. 🛡️ Cascading Model & Key Pool Fallback (Kademeli Model & Anahtar Zırhı)
Never stall on `429 RESOURCE_EXHAUSTED` or rate limit errors:

| Tier | Primary | Secondary (Fallback) | Tertiary (Fallback) |
|---|---|---|---|
| **Lead Architect & Verifier** | `gemini-3.7-flash` (Keys 1→4) | `gemini-3.6-flash` (Keys 1→4) | `gemini-3.5-flash-lite` (Keys 1→4) |
| **Parallel Specialized Agents** | `gemini-3.5-flash-lite` (Keys 1→4) | `gemini-3.1-flash-lite` (Keys 1→4) | `gemma-4-31b-it` (Keys 1→4) |

- Automatic rotation through an array of API keys.
- Automatic downgrade to secondary model tiers if an entire key pool exhausts its quota, with cooldown management.

### 3. 🛠️ Controlled Code Editing via Aider Integration
- Dispatches granular file modifications and refactorings through Aider subprocess tooling for deterministic and auditable repository modifications.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- Git

### Installation
```bash
git clone https://github.com/Piardian/Atlas.git
cd Atlas
pip install -r requirements.txt
cp .env.example .env
# Fill in your GEMINI_USER_*_KEY values in .env
```

### Usage
```bash
# 1. Interactive Mode
python main.py

# 2. Direct Task Execution with Custom Workspace
python main.py --prompt "Build a real-time SMC Liquidity & Fair Value Gap tracker using Binance WebSocket" --workspace ./crypto_smc_bot

# 3. Built-in Verification Demo
python main.py --demo
```

---

## 📄 License
MIT License
