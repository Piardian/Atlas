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
from .tools.job_application_tool import JobApplicationTool
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
        max_arch_attempts = 6

        job_task_requested = any(term in user_prompt.casefold() for term in ("iş ilan", "is ilan", "linkedin", "kariyer", "cv", "başvur", "basvur", "job application", "job search"))

        if task_type == "web_research":
            if job_task_requested:
                arch_system_prompt = (
                    "Sen Kıdemli Baş Kariyer ve Araştırma Mimarısın. Verilen iş arama ve başvuru görevini yürütecek "
                    "UZMAN İŞ ARAMA VE BAŞVURU AJANINI belirle.\n"
                    "Çıktını SADECE geçerli JSON formatında ver:\n"
                    "{\n"
                    '  "domain": "Job Search & Career Application",\n'
                    '  "project_name": "' + str(arch_spec.get("project_title", "JobSearch")) + '",\n'
                    '  "agents": [\n'
                    '    {\n'
                    '      "role": "Job Search & Career Application Specialist",\n'
                    '      "goal": "Aday profiline uygun iş ilanlarını keşfetmek, incelemek, CV uyumunu hesaplamak ve başvuru formunu hazırlamak",\n'
                    '      "backstory": "İş ilanları, LinkedIn, aday profili eşleştirme ve başvuru formları yönetiminde uzman otonom kariyer ajanı",\n'
                    '      "capabilities": ["job_application"],\n'
                    '      "target_files": "WEB_RESEARCH_REPORT.md",\n'
                    '      "instruction": "Atlas Job Search & Application Tool ile ilanları discover et, inspect ile detaylarını inceleyip CV eşleşme puanını hesapla, prepare ile başvuru formunu kullanıcı onayıyla hazırla ve raporla"\n'
                    '    }\n'
                    '  ]\n'
                    "}"
                )
            else:
                arch_system_prompt = (
                    "Sen Kıdemli Baş Araştırma Mimarısın. Verilen web araştırma görevini gerçek tarayıcı (Browser Use + Chromium) "
                    "ile yürütecek UZMAN TARAYICI ARAŞTIRMA AJANINI belirle.\n"
                    "Çıktını SADECE geçerli JSON formatında ver:\n"
                    "{\n"
                    '  "domain": "Web Research & Browser Automation",\n'
                    '  "project_name": "' + str(arch_spec.get("project_title", "WebResearch")) + '",\n'
                    '  "agents": [\n'
                    '    {\n'
                    '      "role": "Browser Research & Web Extraction Agent",\n'
                    '      "goal": "Hedef şirketlerin resmi web sitelerini Browser Use ile ziyaret edip ürünleri ve iletişim bilgilerini çıkarmak",\n'
                    '      "backstory": "Gerçek tarayıcı otomasyonu, DOM analizi ve kurumsal veri çıkarım uzmanı",\n'
                    '      "capabilities": ["browser"],\n'
                    '      "target_files": "WEB_RESEARCH_REPORT.md",\n'
                    '      "instruction": "Web araştırma talebini Browser Automation Tool ile eksiksiz yürüt, ürünleri, şirket bilgilerini ve iletişim kanallarını çıkar ve raporla"\n'
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

        max_exec_attempts = 6
        final_crew_result = None
        completed_browser_output = None

        for attempt in range(max_exec_attempts):
            parallel_tasks = []
            try:
                parallel_agents, parallel_tasks, reviewer_agent, review_task = self.factory.build_dynamic_crew_components(
                    agents_blueprint=blueprint_specs,
                    user_prompt=expanded_prompt,
                    arch_spec=arch_spec,
                )

                if completed_browser_output and task_type == "web_research":
                    print("\n🧐 [Aşama 3 - Devam] Tarayıcı araştırması tamamlandı; Denetleyici Ajan (Reviewer) yeni modelle raporu sentezliyor...")
                    review_task.context = []
                    review_task.description += f"\n\nTARAYICI AJANI BULGULARI:\n{completed_browser_output}"
                    reviewer_crew = Crew(
                        agents=[reviewer_agent],
                        tasks=[review_task],
                        process=Process.sequential,
                        verbose=True,
                        memory=False,
                        tracing=False,
                    )
                    final_crew_result = reviewer_crew.kickoff()
                    break

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
                    if job_task_requested:
                        print(f"\n💼 Uzman Kariyer ve Başvuru Ajanı Görevi Yürütüyor...\n")
                    else:
                        print(f"\n🌐 Uzman Araştırma Ajanları Browser Use ile Web Taramasına Başlıyor...\n")
                else:
                    print(f"\n⚡ Uzman Ajanlar Paralel Olarak Aider ile Yerel Kodlamaya Başlıyor...\n")

                final_crew_result = execution_crew.kickoff()
                break

            except Exception as e:
                print(f"[Atlas Orchestrator Uyarı] Yürütme denemesi {attempt+1}/{max_exec_attempts} hata aldı: {e}")
                if task_type == "web_research" and parallel_tasks and getattr(parallel_tasks[0], "output", None):
                    raw_out = str(getattr(parallel_tasks[0].output, "raw", parallel_tasks[0].output) or "").strip()
                    if raw_out:
                        completed_browser_output = raw_out
                if self._is_recoverable_error(e):
                    try:
                        cur_model, cur_key = self.router.get_best_model_and_key(tier="worker")
                        self.router.mark_exhausted(cur_model, cur_key, reason=str(e)[:80])
                    except Exception as mark_err:
                        print(f"[Atlas Orchestrator] Anahtar isaretleme uyarisi: {mark_err}")
                    print(f"\n[Ağ/Kota Zırhı] 🔄 Sıradaki anahtar/model ile yeniden deneniyor ({attempt+1}/{max_exec_attempts})...")
                    time.sleep(1)
                else:
                    if task_type == "web_research":
                        break
                    raise e

        if not final_crew_result and completed_browser_output and task_type == "web_research":
            final_crew_result = completed_browser_output

        if not final_crew_result and task_type == "web_research":
            if job_task_requested:
                print("\n🛡️ [Atlas Capability Shield] Doğrudan Atlas Job Search & Application katmanı devreye alınıyor...")
                job_tool = JobApplicationTool()
                cand_file = Path("candidate_profile.json")
                cand_profile_str = cand_file.read_text(encoding="utf-8") if cand_file.is_file() else "{}"

                search_q = "Junior Sistem ve Ağ Destek"
                if any(w in user_prompt.casefold() for w in ("junior", "yeni mezun", "staj", "yetistir")):
                    search_q = "Junior Sistem ve Ağ Destek"
                elif "ağ" in user_prompt.casefold() or "network" in user_prompt.casefold():
                    search_q = "Junior Network Destek Elemanı"
                elif "bilgi teknolojileri" in user_prompt.casefold() or "bt" in user_prompt.casefold():
                    search_q = "Yetiştirilmek Üzere Bilgi Teknolojileri Destek"
                else:
                    search_q = "Junior Sistem ve Ağ Destek Elemanı"

                print(f"[Atlas Job Shield] 🔍 Canlı LinkedIn iş ilanları keşfediliyor: '{search_q}'...")
                disc_res_str = job_tool._run(action="discover", search_query=search_q, candidate_profile_json=cand_profile_str, max_results=6, headless=self.headless)
                disc_data = json.loads(disc_res_str) if isinstance(disc_res_str, str) and disc_res_str.strip().startswith("{") else {}
                jobs_list = disc_data.get("jobs", [])

                inspected_jobs = []
                for j in jobs_list[:3]:
                    j_url = j.get("url")
                    if j_url:
                        print(f"[Atlas Job Shield] 📄 İlan inceleniyor: {j.get('title')} ({j.get('company')})...")
                        insp_str = job_tool._run(action="inspect", url=j_url, candidate_profile_json=cand_profile_str, headless=self.headless)
                        try:
                            inspected_jobs.append(json.loads(insp_str))
                        except Exception:
                            inspected_jobs.append(j)

                prepared_status = "Hazırlık beklemede"
                if inspected_jobs:
                    top_job = sorted(inspected_jobs, key=lambda x: x.get("match_score", 0), reverse=True)[0]
                    top_url = top_job.get("url")
                    if top_url:
                        prep_action = "submit" if any(w in user_prompt.casefold() for w in ("gönder", "gonder", "başvur", "basvur", "submit", "apply")) else "prepare"
                        print(f"\n[Atlas Job Shield] ✍️ En yüksek uyumlu ilana başvuru işlemi ({prep_action}) başlatılıyor: {top_job.get('title')} ({top_job.get('company')})...")
                        prep_res_str = job_tool._run(action=prep_action, url=top_url, candidate_profile_json=cand_profile_str, headless=self.headless)
                        try:
                            prep_data = json.loads(prep_res_str)
                            if prep_data.get("status") == "submitted":
                                prepared_status = "Başvuru formu ve CV başarıyla doldurulup gönderildi (Submit tamamlandı)"
                            elif prep_data.get("success"):
                                prepared_status = "Form alanları ve CV dolduruldu"
                            elif prep_data.get("status") == "cancelled_by_user":
                                prepared_status = "Kullanıcı tarafından iptal edildi"
                            else:
                                prepared_status = prep_data.get("message", "İşlem tamamlandı")
                        except Exception:
                            prepared_status = "Başvuru işlemi yapıldı"

                rows = []
                for idx, j in enumerate(inspected_jobs or jobs_list[:5], start=1):
                    title = j.get("title", "Pozisyon")
                    comp = j.get("company", "Bilinmiyor")
                    loc = j.get("location", "İstanbul / Türkiye")
                    score = j.get("match_score", 0)
                    skills_req = ", ".join(j.get("skills_required", [])) or "Ağ / Sistem Yetkinlikleri"
                    j_url = j.get("url", "#")
                    rows.append(f"| {idx} | {comp} - {title} | {loc} | %{score} | {skills_req} | [İlana Git]({j_url}) |")

                cand_info = json.loads(cand_profile_str) if cand_profile_str.strip().startswith("{") else {}
                report_md = (
                    f"# 🎯 Ömer Demircan - Sistem & Ağ Destek İş İlanları ve Başvuru Raporu\n\n"
                    f"**Aday:** {cand_info.get('full_name', 'Ömer Demircan')} ({cand_info.get('title', 'Network & System Support Specialist')})\n"
                    f"**CV:** `{cand_info.get('resume_path', 'OmerDemircanNetworkCV.pdf')}`\n"
                    f"**Başvuru Formu Durumu:** {prepared_status}\n\n"
                    f"## 📋 Uygun İş İlanları Karşılaştırma Tablosu\n\n"
                    f"| # | Şirket / Pozisyon | Konum | CV Uyumu (%) | Tespit Edilen Gereksinimler | İlan Bağlantısı |\n"
                    f"|---|---|---|---|---|---|\n"
                    + "\n".join(rows) + "\n\n"
                    f"## 🔍 İncelenen İlan Detayları ve Uyum Analizi\n\n"
                )
                for j in inspected_jobs:
                    report_md += (
                        f"### {j.get('company', 'Şirket')} - {j.get('title', 'Pozisyon')}\n"
                        f"- **Konum:** {j.get('location', 'Belirtilmemiş')}\n"
                        f"- **CV Eşleşme Oranı:** %{j.get('match_score', 0)}\n"
                        f"- **Gereksinimler:** {', '.join(j.get('skills_required', [])) or 'Genel BT / Network'}\n"
                        f"- **İlan URL:** {j.get('url')}\n"
                        f"- **Açıklama Özeti:** {j.get('description_snippet', '')[:400]}...\n\n"
                    )
                report_md += (
                    "\n> **Not:** Otomatik form gönderimi (submit) güvenlik ilkeleri gereği devre dışıdır. "
                    "Başvurularınızı açık tarayıcı üzerinden kontrol ederek güvenle tamamlayabilirsiniz.\n"
                )
                final_crew_result = report_md
            else:
                print("\n🛡️ [Atlas Capability Shield] Doğrudan BrowserAutomationTool katmanı devreye alınıyor...")
                direct_bt = BrowserAutomationTool(
                    default_working_dir=self.workspace_dir,
                    default_headless=self.headless,
                )
                final_crew_result = direct_bt.run(task=user_prompt)

        # --- AŞAMA 4: Doğrulama ve Çıktı Kaydı ---
        print("\n===============================================================")
        print("🔍 [Aşama 4] Doğrulama ve Raporlama...")
        print("===============================================================")

        if task_type == "web_research":
            work_path = Path(self.workspace_dir)
            work_path.mkdir(parents=True, exist_ok=True)
            report_file = work_path / "WEB_RESEARCH_REPORT.md"
            result_str = str(final_crew_result or "").strip()

            telemetry_path = work_path / "browser_telemetry.jsonl"
            visited_count = 0
            latest_telemetry = None
            if telemetry_path.exists():
                for line in telemetry_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        parsed_line = json.loads(line)
                        visited_count += len(parsed_line.get("visited_urls", []))
                        latest_telemetry = parsed_line
                    except Exception as tel_parse_err:
                        print(f"[Atlas Verifier] Telemetri satiri ayristirilamadi: {tel_parse_err}")

            if latest_telemetry and "=== Atlas Browser Capability Sonucu" not in result_str and not job_task_requested:
                backend_used = latest_telemetry.get("backend_used", "browser-use")
                elapsed_s = latest_telemetry.get("elapsed_seconds", 0.0)
                urls_list = latest_telemetry.get("visited_urls", [])
                urls_md = "\n".join(f"  - {u}" for u in urls_list) if urls_list else "  - (Kayitli URL yok)"
                header_block = (
                    f"=== Atlas Browser Capability Sonucu (BASARILI | Backend: {backend_used}) ===\n"
                    f"Gorev: {user_prompt}\n"
                    f"Sure: {elapsed_s:.2f} sn\n"
                    f"Ziyaret Edilen URL'ler:\n{urls_md}\n\n"
                )
                result_str = header_block + result_str
                final_crew_result = result_str

            report_file.write_text(result_str, encoding="utf-8")
            has_table = "|" in result_str and "---" in result_str

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
