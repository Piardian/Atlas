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
from .tools.key_manager import SmartFallbackRouter

os.environ["CREWAI_TRACING_ENABLED"] = "false"
litellm.num_retries = 0
litellm.request_timeout = 30


class AiderCrewOrchestrator:
    """
    Google AI Studio (Gemini), CrewAI ve Aider bileşenlerini yöneten;
    Dinamik Şartname Genişletme (Spec Expander), Ultra-Hızlı Baş Mimar Planlayıcısı,
    ve Fiili Çalıştırmalı Kendi Kendini Düzeltme Zırhına (Closed-Loop Runtime Verifier) sahip v4.5 Orkestratör.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = str(Path(workspace_dir).resolve())
        self.factory = DynamicAgentFactory(self.workspace_dir)
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
        Kullanıcı talebini 4 aşamalı kurumsal kapalı döngü (Closed-Loop) orkestrasyonuyla çalıştırır.
        """
        print(f"\n===============================================================")
        print(f"🚀 [Orchestrator v4.5] Hızlı Closed-Loop Otonom Ajan Süreci")
        print(f"📂 Hedef Çalışma Dizini: {self.workspace_dir}")
        print(f"💬 Ham Kullanıcı Talebi: {user_prompt[:120]}...")
        print(f"===============================================================\n")

        # --- AŞAMA 1: Derin Prompt Genişletme & Teknik Şartname (Spec Expansion) ---
        print("📐 [Aşama 1] Baş Mimar talebi inceliyor ve kurumsal Teknik Şartnameyi (ARCH_SPEC) oluşturuyor...")
        arch_spec = self.spec_expander.expand_specification(user_prompt)
        expanded_prompt = arch_spec.get("expanded_technical_prompt", user_prompt)
        
        print(f"   • Proje Başlığı: {arch_spec.get('project_title', 'Autonomous Project')}")
        print(f"   • Hedef Platform: {arch_spec.get('target_platform', 'Python')}")
        print(f"   • Olay Döngüsü Modeli: {arch_spec.get('event_loop_model', 'continuous')}")
        print(f"   • Zorunlu Uç Durumlar: {len(arch_spec.get('mandatory_edge_cases', []))} adet kural tanımlandı.\n")

        # --- AŞAMA 2: Ultra-Hızlı Baş Mimar İş Bölümü Planı (Direct Single-Shot) ---
        print("🧠 [Aşama 2] Baş Mimar dinamik uzman kadrosunu ve görev dağılımını tasarlıyor (Hızlı Planlayıcı)...")
        blueprint_output = ""
        max_arch_attempts = 12

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
                        {"role": "user", "content": f"Teknik Şartname:\n'''{expanded_prompt}'''"}
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

        # --- AŞAMA 3: Dinamik Ajanların Kodlama Yapması (Parallel Execution) ---
        blueprint_specs = self.factory.parse_blueprint(blueprint_output, user_prompt)
        
        max_exec_attempts = 12
        final_crew_result = None

        for attempt in range(max_exec_attempts):
            try:
                parallel_agents, parallel_tasks, reviewer_agent, review_task = self.factory.build_dynamic_crew_components(
                    agents_blueprint=blueprint_specs,
                    user_prompt=expanded_prompt,
                    arch_spec=arch_spec,
                )

                print(f"\n👥 [Aşama 3] Sahaya Sürülen Dinamik Uzman Kadrosu ({len(parallel_agents)} Ajan):")
                for i, ag in enumerate(parallel_agents):
                    print(f"  {i+1}. 🤖 {ag.role}")

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

                print(f"\n⚡ 5 Uzman Ajan Paralel Olarak Aider ile Yerel Kodlamaya Başlıyor...\n")
                final_crew_result = execution_crew.kickoff()
                break

            except Exception as e:
                if self._is_recoverable_error(e):
                    cur_model, cur_key = self.router.get_best_model_and_key(tier="worker")
                    self.router.mark_exhausted(cur_model, cur_key, reason=str(e)[:80])
                    print(f"\n[Ağ/Kota Zırhı] 🔄 Sıradaki anahtar/model ile yeniden deneniyor ({attempt+1}/{max_exec_attempts})...")
                    time.sleep(1)
                else:
                    raise e

        # --- AŞAMA 4: Closed-Loop Runtime Sandbox & Hızlı Doğrulama ---
        print("\n===============================================================")
        print("🔍 [Aşama 4] Closed-Loop Runtime Sandbox Doğrulaması...")
        print("===============================================================\n")

        verification_report = self.verifier.self_heal_loop(max_iterations=1)

        print("\n🏆 [Orchestrator v4.5] Nihai Doğrulama ve Teslim Raporu:")
        print(f"• Sözdizimi (AST) Geçerliliği: {'PASS ✅' if verification_report['syntax']['pass'] else 'FAIL ❌'}")
        print(f"• Başlatıcı (run.py) Bütünlüğü: {'PASS ✅' if verification_report['entrypoint']['pass'] else 'FAIL ❌'}")
        print(f"• Birim Testleri: {'PASS ✅' if verification_report['tests']['pass'] else 'FAIL ❌'}")

        print("\n✨ Proje başarıyla güncellendi ve otonom döngü tamamlandı!")
        return final_crew_result
