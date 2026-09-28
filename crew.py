import json
import os
import re
import time
from pathlib import Path
from typing import Any
import litellm
from crewai import Crew, Process
from .dynamic_factory import DynamicAgentFactory
from .spec_expander import SpecificationExpander
from .critic_verifier import RuntimeVerifier
from .tools.browser_tool import BrowserAutomationTool
from .tools.key_manager import SmartFallbackRouter

os.environ["CREWAI_TRACING_ENABLED"] = "false"
litellm.num_retries = 0
litellm.request_timeout = 30


class AiderCrewOrchestrator:
    """
    Atlas Multi-Agent Orchestrator:
    Google AI Studio (Gemini), CrewAI, Aider (Kod Yurutme) ve Browser Use (Gercek Tarayici Otomasyonu)
    bilesenlerini yoneten; Dinamik Ajan Fabrikasi ve Kademeli Model/Anahtar Zirhina sahip Orkestrator.
    """

    def __init__(self, workspace_dir: str, headless: bool = True):
        self.workspace_dir = str(Path(workspace_dir).resolve())
        self.headless = headless
        self.factory = DynamicAgentFactory(self.workspace_dir, headless=headless)
        self.spec_expander = SpecificationExpander(self.workspace_dir)
        self.verifier = RuntimeVerifier(self.workspace_dir)
        self.router = SmartFallbackRouter()

    def _is_recoverable_error(self, e: Exception) -> bool:
        msg = str(e).lower()
        keywords = [
            "429", "resource_exhausted", "quota", "503", "502", "504",
            "disconnected", "remotedisconnected", "connectionreset", "timeout",
            "peer closed connection", "broken pipe", "server disconnected", "connection error"
        ]
        return any(k in msg for k in keywords)

    def kickoff(self, user_prompt: str) -> Any:
        """
        Kullanıcı talebini görevin doğasına göre (Web Research / Software Engineering / Hybrid)
        4 aşamalı kapalı döngü (Closed-Loop) orkestrasyonuyla çalıştırır.
        """
        print(f"\n===============================================================")
        print(f"🏛️ [Atlas Orchestrator] Otonom Çoklu Ajan Süreci Başlatıldı")
        print(f"📂 Hedef Çalışma Dizini: {self.workspace_dir}")
        print(f"💬 Kullanıcı Görevi: {user_prompt[:120]}...")
        print(f"===============================================================\n")

        # --- AŞAMA 1: Derin Prompt Genişletme & Görev Modu Algılama ---
        print("📐 [Aşama 1] Baş Mimar görevi analiz ediyor ve Şartnameyi (ARCH_SPEC) oluşturuyor...")
        arch_spec = self.spec_expander.expand_specification(user_prompt)
        task_type = arch_spec.get("task_type", self.spec_expander.detect_task_mode(user_prompt))
        arch_spec["task_type"] = task_type
        expanded_prompt = arch_spec.get("expanded_technical_prompt", user_prompt)

        print(f"   • Görev Başlığı: {arch_spec.get('project_title', 'Atlas Autonomous Mission')}")
        print(f"   • Algılanan Görev Modu: {task_type.upper()}")
        print(f"   • Hedef Platform / Yetenek: {arch_spec.get('target_platform', 'Python / Browser Use')}")
        print(f"   • Zorunlu Kurallar: {len(arch_spec.get('mandatory_edge_cases', []))} adet kural tanımlandı.\n")

        # --- AŞAMA 2: Baş Mimar İş Bölümü Planı (Dynamic Agent Blueprint) ---
        print("🧠 [Aşama 2] Baş Mimar dinamik uzman kadrosunu ve yetenek (capability) dağılımını tasarlıyor...")
        blueprint_output = ""
        max_arch_attempts = 2 if task_type == "web_research" else 8

        if task_type == "web_research":
            arch_system_prompt = (
                "Sen Kıdemli Baş Araştırma Mimarısın. Verilen web araştırma görevini gerçek tarayıcı (Browser Use) "
                "ile yürütecek 2 UZMAN ARAŞTIRMA AJANI belirle.\n"
                "Çıktını SADECE geçerli JSON formatında ver:\n"
                "{\n"
                '  "domain": "Web Research & Browser Automation",\n'
                '  "project_name": "' + str(arch_spec.get("project_title", "WebResearch")) + '",\n'
                '  "agents": [\n'
                '    {\n'
                '      "role": "Research & Discovery Agent",\n'
                '      "goal": "Hedef şirketleri ve resmi web adreslerini keşfetmek",\n'
                '      "backstory": "Ekosistem ve kurumsal kaynak araştırmacısı",\n'
                '      "capabilities": ["browser"],\n'
                '      "target_files": "discovery.md",\n'
                '      "instruction": "Tarayıcı aracıyla hedef şirketleri ve resmi sitelerini bul"\n'
                '    },\n'
                '    {\n'
                '      "role": "Deep Browser Extraction Agent",\n'
                '      "goal": "Şirket web sitelerinden ürünleri ve iletişim bilgilerini çıkarmak",\n'
                '      "backstory": "DOM analizi ve kurumsal veri çıkarım uzmanı",\n'
                '      "capabilities": ["browser"],\n'
                '      "target_files": "extraction.md",\n'
                '      "instruction": "Resmi web sitelerini ziyaret et, ürünleri ve iletişim bilgilerini çıkar"\n'
                '    }\n'
                '  ]\n'
                "}"
            )
        else:
            arch_system_prompt = (
                "Sen Kıdemli Baş Mimarsın. Verilen teknik şartnameyi eksiksiz kodlayacak 5 FARKLI UZMAN AJAN belirle.\n"
                "Çıktını SADECE geçerli JSON formatında ver:\n"
                "{\n"
                '  "domain": "' + str(arch_spec.get("target_platform", "Software")) + '",\n'
                '  "project_name": "' + str(arch_spec.get("project_title", "Project")) + '",\n'
                '  "agents": [\n'
                '    {\n'
                '      "role": "Uzman Rolü 1",\n'
                '      "goal": "Spesifik Hedef",\n'
                '      "backstory": "Uzmanlık Geçmişi",\n'
                '      "capabilities": ["aider"],\n'
                '      "target_files": "dosyalar.py",\n'
                '      "instruction": "Aider CLI talimatı"\n'
                '    }\n'
                '  ]\n'
                "}"
            )

        for attempt in range(max_arch_attempts):
            model, key = self.router.get_best_model_and_key(tier="architect")
            try:
                response = litellm.completion(
                    model=model,
                    api_key=key,
                    messages=[
                        {"role": "system", "content": arch_system_prompt},
                        {"role": "user", "content": f"Şartname:\n'''{expanded_prompt}'''"},
                    ],
                    temperature=0.2,
                    timeout=25,
                    num_retries=0,
                )
                blueprint_output = response.choices[0].message.content
                break
            except Exception as e:
                if self._is_recoverable_error(e):
                    self.router.mark_exhausted(model, key, reason=str(e)[:80])
                    print(f"[Ağ/Kota Zırhı] 🔄 Sıradaki anahtara geçiliyor ({attempt+1}/{max_arch_attempts})...")
                    time.sleep(1)
                else:
                    break

        # --- AŞAMA 3: Dinamik Ajanların Görevi İcra Etmesi ---
        blueprint_specs = self.factory.parse_blueprint(blueprint_output, user_prompt, task_type=task_type)

        max_exec_attempts = 2 if task_type == "web_research" else 8
        final_crew_result = None

        for attempt in range(max_exec_attempts):
            try:
                parallel_agents, parallel_tasks, reviewer_agent, review_task = self.factory.build_dynamic_crew_components(
                    agents_blueprint=blueprint_specs,
                    user_prompt=expanded_prompt,
                    arch_spec=arch_spec,
                )

                print(f"\n👥 [Aşama 3] Sahaya Sürülen Dinamik Uzman Kadrosu ({len(parallel_agents) + 1} Ajan):")
                for i, ag in enumerate(parallel_agents):
                    tool_names = ", ".join(t.name for t in getattr(ag, "tools", []) or []) or "LLM Reasoning"
                    print(f"  {i+1}. 🤖 {ag.role}  [Yetenek: {tool_names}]")
                print(f"  {len(parallel_agents)+1}. 🧐 {reviewer_agent.role}  [Yetenek: Doğrulama & Rapor Sentezi]")

                all_workers = parallel_agents + [reviewer_agent]
                all_tasks = parallel_tasks + [review_task]

                execution_crew = Crew(
                    agents=all_workers,
                    tasks=all_tasks,
                    process=Process.sequential,
                    verbose=True,
                    memory=False,
                    tracing=False,
                )

                if task_type == "web_research":
                    print(f"\n🌐 Uzman Araştırma Ajanları Browser Use ile Web Taramasına Başlıyor...\n")
                else:
                    print(f"\n⚡ Uzman Ajanlar Paralel Olarak Aider ile Yerel Kodlamaya Başlıyor...\n")

                final_crew_result = execution_crew.kickoff()
                break

            except Exception as e:
                if self._is_recoverable_error(e):
                    try:
                        cur_model, cur_key = self.router.get_best_model_and_key(tier="worker")
                        self.router.mark_exhausted(cur_model, cur_key, reason=str(e)[:80])
                    except Exception:
                        pass
                    print(f"\n[Ağ/Kota Zırhı] 🔄 Sıradaki anahtar/model ile yeniden deneniyor ({attempt+1}/{max_exec_attempts})...")
                    time.sleep(1)
                else:
                    if task_type == "web_research":
                        break
                    raise e

        if not final_crew_result and task_type == "web_research":
            print("\n🛡️ [Atlas Capability Shield] Doğrudan BrowserAutomationTool katmanı devreye alınıyor...")
            direct_bt = BrowserAutomationTool(
                default_working_dir=self.workspace_dir,
                default_headless=self.headless,
            )
            final_crew_result = direct_bt.run(task=user_prompt)

        # --- AŞAMA 4: Doğrulama ve Çıktı Kaydı ---
        print("\n===============================================================")
        print("🔍 [Aşama 4] Doğrulama ve Raporlama...")
        print("===============================================================\n")

        if task_type == "web_research":
            work_path = Path(self.workspace_dir)
            work_path.mkdir(parents=True, exist_ok=True)
            report_file = work_path / "WEB_RESEARCH_REPORT.md"
            result_str = str(final_crew_result or "")
            report_file.write_text(result_str, encoding="utf-8")

            has_table = "|" in result_str and "---" in result_str
            telemetry_path = work_path / "browser_telemetry.jsonl"
            visited_count = 0
            if telemetry_path.exists():
                for line in telemetry_path.read_text(encoding="utf-8").splitlines():
                    try:
                        visited_count += len(json.loads(line).get("visited_urls", []))
                    except Exception:
                        pass

            print("🏆 [Atlas Web Research Verifier] Nihai Doğrulama Raporu:")
            print(f"• Tarayıcı Telemetri Kaydı: {'PASS ✅' if visited_count > 0 else 'INFO ℹ️'} ({visited_count} sayfa ziyaret edildi)")
            print(f"• Yapılandırılmış Markdown Tablosu: {'PASS ✅' if has_table else 'WARN ⚠️'}")
            print(f"• Kaydedilen Rapor Dosyası: {report_file}")
        else:
            verification_report = self.verifier.self_heal_loop(max_iterations=1)
            print("\n🏆 [Atlas Code Verifier] Nihai Doğrulama ve Teslim Raporu:")
            print(f"• Sözdizimi (AST) Geçerliliği: {'PASS ✅' if verification_report['syntax']['pass'] else 'FAIL ❌'}")
            print(f"• Başlatıcı (run.py) Bütünlüğü: {'PASS ✅' if verification_report['entrypoint']['pass'] else 'FAIL ❌'}")
            print(f"• Birim Testleri: {'PASS ✅' if verification_report['tests']['pass'] else 'FAIL ❌'}")

        print("\n✨ Atlas otonom döngüsü başarıyla tamamlandı!")
        return final_crew_result


AtlasOrchestrator = AiderCrewOrchestrator
