import argparse
import sys
from pathlib import Path

# Windows konsol UTF-8 uyumlulugu
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ust dizini sys.path'e ekle
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

from crewai_aider_orchestrator.crew import AtlasOrchestrator
from crewai_aider_orchestrator.tools.browser_tool import BrowserAutomationTool

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    console = Console(force_terminal=True, legacy_windows=False)
except Exception:
    console = None


DEFAULT_INTERNSHIP_MISSION = (
    "Türkiye'deki yapay zeka şirketlerini araştır. "
    "İlk 5 şirketi bul (örneğin Vispera, CBOT, Tazi AI, Intenseye, Cortea / Sestek gibi resmi web siteleri üzerinden), "
    "gerçek web sitelerini ziyaret ederek ana yapay zeka ürünlerini ve resmi iletişim bilgilerini çıkar, "
    "karşılaştırmalı tablo halinde Türkçe olarak raporla."
)


def run_demo():
    parser = argparse.ArgumentParser(description="Atlas + Browser Use — AI Web Research Agent Staj PoC Demosu")
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=DEFAULT_INTERNSHIP_MISSION,
        help="Araştırma görevi açıklaması",
    )
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=str(ROOT_DIR / "workspace_research"),
        help="Araştırma raporunun ve telemetri kayıtlarının yazılacağı dizin",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Chromium tarayıcısını görünür pencerede (headless=False) çalıştırır",
    )
    parser.add_argument(
        "--direct-tool",
        action="store_true",
        help="Sadece soyutlanmış browser_tool.run(...) katmanını doğrudan test eder",
    )
    args = parser.parse_args()

    workspace_path = Path(args.workspace).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    header = (
        "🏛️ ATLAS — AI Web Research Agent (Staj PoC Demosu)\n"
        "• Orkestratör: Atlas (Dynamic Agent Factory + CrewAI + Reviewer)\n"
        "• Capability: BrowserAutomationTool (Browser Use + Playwright Chromium)\n"
        f"• Çıktı Dizini: {workspace_path}"
    )
    if console:
        console.print(Panel(header, title="[bold cyan]Atlas Browser Capability PoC[/bold cyan]", border_style="cyan"))
    else:
        print(header)

    if args.direct_tool:
        print("\n🔧 [Direct Capability Mode] browser_tool.run(task=...) doğrudan çağrılıyor...\n")
        browser_tool = BrowserAutomationTool(
            default_working_dir=str(workspace_path),
            default_headless=not args.show_browser,
        )
        output = browser_tool.run(task=args.prompt)
        report_path = workspace_path / "TR_AI_COMPANIES_REPORT.md"
        report_path.write_text(output, encoding="utf-8")
        if console:
            console.print(Panel(Markdown(output), title="[green]BrowserAutomationTool Çıktısı[/green]", border_style="green"))
        else:
            print(output)
        print(f"\n✅ Rapor kaydedildi: {report_path}")
        return

    orchestrator = AtlasOrchestrator(
        workspace_dir=str(workspace_path),
        headless=not args.show_browser,
    )
    result = orchestrator.kickoff(user_prompt=args.prompt)

    final_report_path = workspace_path / "TR_AI_COMPANIES_REPORT.md"
    final_report_path.write_text(str(result), encoding="utf-8")

    if console:
        console.print("\n")
        console.print(
            Panel(
                Markdown(str(result)),
                title="[bold green]Nihai Doğrulanmış Web Araştırma Raporu[/bold green]",
                border_style="green",
            )
        )
    else:
        print("\n=== NİHAİ RAPOR ===\n")
        print(result)

    print(f"\n📄 Staj Sunum Raporu Kaydedildi: {final_report_path}")
    print(f"📊 Tarayıcı Telemetri Kaydı: {workspace_path / 'browser_telemetry.jsonl'}")


if __name__ == "__main__":
    run_demo()
