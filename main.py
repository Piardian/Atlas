import os
import sys
import argparse
from pathlib import Path

# Windows konsol Unicode uyumlulugu
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv

# Modul importu icin sys.path ayari
sys.path.insert(0, str(Path(__file__).parent.parent))

from crewai_aider_orchestrator.crew import AiderCrewOrchestrator
from crewai_aider_orchestrator.tools.key_manager import SmartFallbackRouter, ARCHITECT_CASCADE, WORKER_CASCADE

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    console = Console(force_terminal=True, legacy_windows=False)
except Exception:
    console = None


def print_banner():
    banner_text = """
===============================================================
   🏛️ ATLAS: Autonomous Multi-Agent Orchestrator CLI
   Dynamic Agent Factory | Browser Use | Aider | Gemini Shield
===============================================================
    """
    if console:
        try:
            console.print(f"[bold cyan]{banner_text}[/bold cyan]")
        except Exception:
            print(banner_text)
    else:
        print(banner_text)


def check_prerequisites():
    """Gerekli Google AI Studio Gemini anahtarlarini dogrular."""
    router = SmartFallbackRouter()
    keys = router.get_all_keys()

    if not keys:
        msg = (
            "[!] UYARI: Hicbir Gemini API anahtari bulunamadi!\n\n"
            "Lutfen .env dosyaniza GEMINI_USER_A_KEY veya GEMINI_API_KEY ekleyin.\n"
        )
        if console:
            try:
                console.print(Panel(msg, title="[red]Eksik Konfigurasyon[/red]", border_style="red"))
            except Exception:
                print(msg)
        else:
            print(msg)
        return False
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Atlas: Google AI Studio (Gemini) + CrewAI + Browser Use + Aider Dinamik Ajan CLI"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Yurutulmesini istediginiz arastirma veya yazilim gelistirme gorevi.",
    )
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=os.getenv("TARGET_WORKSPACE", "./workspace_project"),
        help="Ciktilarin ve raporlarin yazilacagi calisma dizini (varsayilan: ./workspace_project).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ornek bir Finans / SMC Matematiksel Denetim gorevi calistirir.",
    )
    parser.add_argument(
        "--browser-demo",
        action="store_true",
        help="Staj PoC: Turkiye'deki Yapay Zeka Sirketlerini Browser Use ile arastirip tablo halinde raporlar.",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Tarayiciyi gorunur pencerede (headless=False) acar.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    print_banner()

    if not check_prerequisites():
        sys.exit(1)

    router = SmartFallbackRouter()
    keys = router.get_all_keys()

    if args.browser_demo:
        prompt = (
            "Türkiye'deki yapay zeka şirketlerini araştır. "
            "İlk 5 şirketi bul (örneğin Vispera, CBOT, Tazi AI, Mindsite, Intenseye vb. resmi siteleri üzerinden), "
            "web sitelerinden ana ürünlerini ve iletişim bilgilerini çıkar, tablo halinde raporla."
        )
    elif args.demo:
        prompt = (
            "SMC (Smart Money Concepts) ve Piyasa Yapisi (Market Structure) algoritmalarinin matematiksel denetimini yap.\n"
            "1. Swing High/Low (Fraktal) tepe-dip algoritmalarinda indis kaymasi ve wick vs body kurallarini incele.\n"
            "2. FVG (Fair Value Gap) 3 mumluk dengesizlik ve kapanma (mitigation) formullerini dogrula.\n"
            "3. OTE (0.618 / 0.705 / 0.786) hesaplamalarinda float hassasiyetini Decimal ile optimize et.\n"
            "4. Lookahead bias riskine karsi tum gostergeleri i-1 ve i-2 indislerine sabitleyen koruyucu birim testleri yaz.\n"
            "5. Bulgulari, matematiksel duzeltmeleri ve mimari onerileri SMC_MATHEMATICAL_AUDIT_REPORT.md dosyasina kaydet."
        )
    elif args.prompt:
        prompt = args.prompt
    else:
        if console:
            try:
                console.print("[bold yellow]Lutfen arastirma veya gelistirme gorevini girin:[/bold yellow]")
                prompt = console.input("[bold green]> [/bold green]").strip()
            except Exception:
                prompt = input("Gorev: ").strip()
        else:
            prompt = input("Gorev: ").strip()

    if not prompt:
        print("Hata: Gorev bos birakilamaz.")
        sys.exit(1)

    workspace_path = Path(args.workspace).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    arch_cascade_str = " -> ".join([m.replace("gemini/", "") for m in ARCHITECT_CASCADE])
    worker_cascade_str = " -> ".join([m.replace("gemini/", "") for m in WORKER_CASCADE])

    config_info = (
        f"Calisma Alani: {workspace_path}\n"
        f"Aktif Gemini API Anahtari: {len(keys)} Adet (Key Pool Aktif)\n"
        f"Yetenekler (Capabilities): Browser Use (Chromium) + Aider CLI\n"
        f"Bas Mimar Kademesi: {arch_cascade_str}\n"
        f"Calisanlar Kademesi: {worker_cascade_str}"
    )

    if console:
        try:
            console.print(Panel(config_info, title="[cyan]Atlas Calisma Ortami[/cyan]", border_style="cyan"))
        except Exception:
            print("\n" + config_info + "\n")
    else:
        print("\n" + config_info + "\n")

    orchestrator = AiderCrewOrchestrator(
        workspace_dir=str(workspace_path),
        headless=not args.show_browser,
    )

    try:
        result = orchestrator.kickoff(user_prompt=prompt)

        if console:
            try:
                console.print("\n")
                console.print(Panel(
                    Markdown(str(result)),
                    title="[green]Atlas Otonom Gorev Raporu[/green]",
                    border_style="green",
                ))
            except Exception:
                print("\nATLAS GOREV RAPORU:\n" + str(result))
        else:
            print("\n" + "=" * 60)
            print("ATLAS GOREV RAPORU:")
            print("=" * 60)
            print(result)

    except KeyboardInterrupt:
        print("\n[!] Islem kullanici tarafindan durduruldu.")
    except Exception as e:
        if console:
            try:
                console.print(Panel(f"[red]{str(e)}[/red]", title="Calisma Hatasi", border_style="red"))
            except Exception:
                print(f"\nHATA OLUSTU: {e}")
        else:
            print(f"\nHATA OLUSTU: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
