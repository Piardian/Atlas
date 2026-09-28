import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from crewai import Agent, Task, LLM
import litellm
from .tools.aider_tool import AiderExecutionTool
from .tools.browser_tool import BrowserAutomationTool
from .tools.key_manager import SmartFallbackRouter, ARCHITECT_CASCADE, WORKER_CASCADE

# LiteLLM'in sessizce 20-30 dakika bekleyen sonsuz retry dongusunu KESINLIKLE DEVRE DISI BIRAK
litellm.num_retries = 0
litellm.request_timeout = 30
os.environ["LITELLM_NUM_RETRIES"] = "0"
os.environ["LITELLM_TIMEOUT"] = "30"


def get_resilient_llm(tier: str = "worker", agent_idx: int = 0) -> LLM:
    """
    1. Kural: Bas Mimar sadece 3.8 -> 3.7 -> 3.6 -> 3.5-flash-lite kullanir (Asla 3.5 altina dusmez).
    2. Kural: Calisanlar 3.8-flash-lite -> 3.5-flash-lite -> 3.1-flash-lite -> Gemma 4 kullanir.
    4 API anahtarini sirayla rotasyon yapar ve rate-limit korumasi saglar.
    """
    router = SmartFallbackRouter()
    model, key = router.get_best_model_and_key(tier=tier, preferred_agent_idx=agent_idx)
    cascade = ARCHITECT_CASCADE if tier.lower() in ["architect", "reviewer", "lead"] else WORKER_CASCADE
    fallbacks = [m for m in cascade if m != model]

    return LLM(
        model=model,
        api_key=key,
        temperature=0.2,
        fallbacks=fallbacks,
        timeout=30,
    )


class DynamicAgentFactory:
    """
    Gelen gorevin alanina (Web Research / Browser Automation, Finans, Web Full-Stack, Veri Bilimi vb.) gore
    en uygun uzman ajanlari ve yeteneklerini (BrowserAutomationTool / AiderExecutionTool)
    calisma aninda (runtime) dinamik ureten fabrika.
    """

    def __init__(self, workspace_dir: str, headless: bool = True):
        self.workspace_dir = workspace_dir
        self.headless = headless

    def create_architect(self) -> Agent:
        """En ust kademedeki Bas Mimar ajanini olusturur (3.8 -> 3.7 -> 3.6 -> 3.5)."""
        architect_llm = get_resilient_llm(tier="architect", agent_idx=0)
        return Agent(
            role="Lead System Architect & Dynamic Workforce Orchestrator",
            goal=(
                "Kullanicinin talebini alanina (Web Research, Finans/Trading, Web, CLI, Veri Analizi vb.) gore analiz etmek, "
                "en uygun uzman rolleri, yetenekleri (browser / aider) ve gorevlerini JSON formatinda dinamik tasarlamak."
            ),
            backstory=(
                "Sen her turlu yazilim ve otonom arastirma disiplininde 20 yillik tecrubeye sahip bir Bas Mimarsin. "
                "Gelen gorevin tabiatina gore en dogru uzmanliklari ve araclari (Browser Use veya Aider) belirler, "
                "isi kusursuz alt modullere bolersin."
            ),
            llm=architect_llm,
            verbose=True,
            allow_delegation=False,
        )

    def parse_blueprint(
        self,
        blueprint_text: str,
        user_prompt: str,
        task_type: str = "software_engineering",
    ) -> List[Dict[str, Any]]:
        """Bas Mimar'in urettigi JSON planini ayiklar; format hatasi olursa gorev tipine uygun akilli sablon uygular."""
        try:
            json_match = re.search(r"\{[\s\S]*\}", blueprint_text)
            if json_match:
                data = json.loads(json_match.group(0))
                min_agents = 1 if task_type == "web_research" else 2
                if "agents" in data and len(data["agents"]) >= min_agents:
                    return data["agents"][:1] if task_type == "web_research" else data["agents"]
        except Exception as e:
            print(f"[DynamicFactory] JSON ayiklama uyarisi: {e}. Akilli '{task_type}' sablonu kullaniliyor.")

        if task_type == "web_research":
            return [
                {
                    "role": "Browser Research & Web Extraction Agent",
                    "goal": f"Gercek tarayici (Browser Use + Chromium) ile hedef web sitelerini ziyaret edip urunleri ve resmi iletisim bilgilerini cikarmak: {user_prompt}",
                    "backstory": "Teknoloji ekosistemleri, sirket kesfi, DOM analizi ve kurumsal iletisim verisi cikariminda uzman tarayici arastirma ajani.",
                    "capabilities": ["browser"],
                    "target_files": "WEB_RESEARCH_REPORT.md",
                    "instruction": (
                        f"Gorev kapsamindaki hedef sirketlerin resmi web sitelerini 'Browser Automation & Web Research Tool' araciyla ziyaret et; "
                        f"ana yapay zeka urunlerini/cozumlerini ve resmi iletisim bilgilerini (e-posta, iletisim sayfasi URL'si, adres/konum) detayli cikar: {user_prompt}"
                    ),
                }
            ]

        return [
            {
                "role": "Core Architecture & Data Specialist",
                "goal": f"{user_prompt} icin gerekli veri yapilarini, modelleri ve semalari kodlamak.",
                "backstory": "Veri yapilari ve temel modeller uzmani kidemli muhendis.",
                "capabilities": ["aider"],
                "target_files": "models.py database.py",
                "instruction": f"Projenin temel veri modellerini ve semalarini olustur: {user_prompt}",
            },
            {
                "role": "Core Logic & Algorithms Specialist",
                "goal": f"{user_prompt} icin ana is mantigini, algoritmik hesaplamalari ve servisleri kodlamak.",
                "backstory": "Yüksek performansli is mantigi, servis katmani ve algoritma gelistiricisi.",
                "capabilities": ["aider"],
                "target_files": "services.py logic.py",
                "instruction": f"Projenin ana is mantigini ve servis fonksiyonlarini olustur: {user_prompt}",
            },
            {
                "role": "Interface, CLI & Endpoints Specialist",
                "goal": f"{user_prompt} icin API rotalarini, CLI veya arayuz kodlarini olusturmak.",
                "backstory": "REST API, CLI ve modern arayuz mimarisi uzmani.",
                "capabilities": ["aider"],
                "target_files": "routers.py main.py",
                "instruction": f"Projenin arayuz ve API katmanini olustur: {user_prompt}",
            },
            {
                "role": "Quality Assurance & Unit Test Specialist",
                "goal": f"{user_prompt} icin birim testleri ve matematiksel dogrulama testlerini yazmak.",
                "backstory": "Kapsamli test mimarisi ve kenar durum guvenligi uzmani.",
                "capabilities": ["aider"],
                "target_files": "tests/test_core.py",
                "instruction": f"Projenin tum modulleri icin pytest birim testlerini olustur: {user_prompt}",
            },
            {
                "role": "DevOps, Automation & Documentation Specialist",
                "goal": f"{user_prompt} icin requirements.txt, baslatici scriptler ve README.md hazirlamak.",
                "backstory": "CI/CD, ortam yonetimi ve teknik dokumantasyon uzmani.",
                "capabilities": ["aider"],
                "target_files": "run.py requirements.txt README.md",
                "instruction": f"Projenin tek komutla calismasini saglayan run.py ve dokumantasyonunu hazirla: {user_prompt}",
            },
        ]

    def _resolve_agent_tools(
        self,
        spec: Dict[str, Any],
        task_type: str,
        agent_idx: int,
        role_name: str,
    ) -> List[Any]:
        """Ajanin blueprint'teki yeteneklerine (capabilities) ve gorev tipine gore dogru araclari baglar."""
        raw_caps = spec.get("capabilities", [])
        if isinstance(raw_caps, str):
            raw_caps = [raw_caps]
        caps = [str(c).lower().strip() for c in raw_caps]

        if not caps:
            if task_type == "web_research":
                caps = ["browser"]
            elif task_type == "hybrid":
                caps = ["browser", "aider"]
            else:
                caps = ["aider"]

        tools_list: List[Any] = []
        if "browser" in caps or "web" in caps or task_type == "web_research":
            tools_list.append(
                BrowserAutomationTool(
                    default_working_dir=self.workspace_dir,
                    tier="worker",
                    agent_idx=agent_idx,
                    agent_role=role_name,
                    default_headless=self.headless,
                )
            )
        if "aider" in caps or "code" in caps or (task_type == "software_engineering" and not tools_list):
            tools_list.append(
                AiderExecutionTool(
                    default_working_dir=self.workspace_dir,
                    tier="worker",
                    agent_idx=agent_idx,
                    agent_role=role_name,
                )
            )
        return tools_list

    def build_dynamic_crew_components(
        self,
        agents_blueprint: List[Dict[str, Any]],
        user_prompt: str,
        arch_spec: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Agent], List[Task], Agent, Task]:
        """
        Dinamik blueprint'ten CrewAI Ajanlarini ve Task'larini olusturur.
        Gorev tipi 'web_research' ise BrowserAutomationTool odaklı araştırma ve tablo raporlama akışı kurar;
        'software_engineering' ise AiderExecutionTool odaklı paralel kodlama akışı kurar.
        """
        task_type = (arch_spec or {}).get("task_type", "software_engineering")
        parallel_agents: List[Agent] = []
        parallel_tasks: List[Task] = []

        spec_context_str = ""
        if arch_spec:
            spec_context_str = (
                f"\n--- GÖREV ŞARTNAMESİ VE KISITLAR ---\n"
                f"• Görev Tipi: {task_type}\n"
                f"• Hedef Platform: {arch_spec.get('target_platform', 'Python')}\n"
                f"• Zorunlu Kurallar: {', '.join(arch_spec.get('mandatory_edge_cases', []))}\n"
                f"------------------------------------\n"
            )

        max_workers = 1 if task_type == "web_research" else 5
        for idx, spec in enumerate(agents_blueprint[:max_workers]):
            role_name = spec.get("role", f"Specialist Agent #{idx+1}")
            goal = spec.get("goal", f"Gorevi basariyla tamamlamak: {user_prompt}")
            backstory = spec.get("backstory", f"{role_name} alaninda uzman kidemli ajan.")
            target_files = spec.get("target_files", "")
            instruction = spec.get("instruction", f"{role_name} kapsamindaki gorevleri tamamla.")

            agent_llm = get_resilient_llm(tier="worker", agent_idx=idx)
            agent_tools = self._resolve_agent_tools(spec, task_type, idx, role_name)

            agent = Agent(
                role=role_name,
                goal=goal,
                backstory=backstory,
                llm=agent_llm,
                tools=agent_tools,
                verbose=True,
                allow_delegation=False,
            )
            parallel_agents.append(agent)

            if task_type == "web_research":
                task_desc = (
                    f"GÖREVİN: {role_name}\n"
                    f"{spec_context_str}\n"
                    f"Kullanıcı Araştırma Talebi: {user_prompt}\n"
                    f"Özel Talimat: {instruction}\n\n"
                    f"KATI ARAŞTIRMA KURALLARI:\n"
                    f"- 'Browser Automation & Web Research Tool' aracını kullanarak gerçek web sitelerini ziyaret et.\n"
                    f"- Ziyaret edilen resmi web sitesi adreslerini (URL), şirket ürünlerini/hizmetlerini ve iletişim bilgilerini eksiksiz çıkar.\n"
                    f"- Asla hayali şirket veya uydurma iletişim bilgisi üretme; doğrudan tarayıcı çıktısına dayan."
                )
                expected_out = f"{role_name} tarafından tarayıcı üzerinden toplanan doğrulanmış şirket, ürün ve iletişim bulguları."
                is_async = False
            else:
                task_desc = (
                    f"GÖREVİN: {role_name}\n"
                    f"{spec_context_str}\n"
                    f"Kullanıcı Talebi: {user_prompt}\n"
                    f"Hedef Dosyalar: {target_files}\n"
                    f"Talimat: {instruction}\n\n"
                    f"KATI KURALLAR:\n"
                    f"- Tanımladığın tüm işlem fonksiyonlarını ana tetikleyiciye (OnTick/main/run) bağla.\n"
                    f"- Aider Code Execution Tool aracını kullanarak kodları doğrudan yerel dosyalara işle.\n"
                    f"Hedef Çalışma Dizini: {self.workspace_dir}"
                )
                expected_out = f"{role_name} tarafından yerel dosyalara yazılan kodlar ve Aider işlem özeti."
                is_async = True

            task = Task(
                description=task_desc,
                expected_output=expected_out,
                agent=agent,
                async_execution=is_async,
            )
            parallel_tasks.append(task)

        # Lead Reviewer / Synthesizer Ajanı
        reviewer_llm = get_resilient_llm(tier="reviewer", agent_idx=1)

        if task_type == "web_research":
            reviewer_agent = Agent(
                role="Lead Research Reviewer & Structured Report Synthesizer",
                goal=(
                    "Araştırma ve Tarayıcı ajanlarından gelen tüm bulguları denetlemek, eksik veya tutarsız bilgileri "
                    "ayıklamak ve kullanıcının istediği nihai yapılandırılmış Markdown tablo raporunu oluşturmak."
                ),
                backstory=(
                    "Sen kurumsal araştırma kalitesi ve veri doğrulamasından sorumlu Baş Denetleyicisin. "
                    "Tarayıcı ajanlarının topladığı ham web verilerini eksiksiz, kanıtlı ve net bir tabloya dönüştürürsün."
                ),
                llm=reviewer_llm,
                tools=[],
                verbose=True,
                allow_delegation=False,
            )

            review_task = Task(
                description=(
                    f"Araştırma ajanlarının topladığı tüm web verilerini incele ve doğrula:\n"
                    f"Kullanıcı Talebi: {user_prompt}\n\n"
                    f"1. Tüm hedef şirketlerin/kaynakların isimlerini, resmi web sitelerini, ana yapay zeka ürünlerini ve iletişim bilgilerini kontrol et.\n"
                    f"2. Sonuçları hem özet Markdown tablosu (| # | Şirket Adı | Web Sitesi | Ürünler & Çözümler | İletişim Bilgileri |) "
                    f"hem de şirket bazlı detaylı alt başlıklar halinde Türkçe olarak raporla.\n"
                    f"3. Raporun sonunda ziyaret edilen kaynak URL'leri kanıt olarak listele."
                ),
                expected_output="Markdown tablosu ve detaylı şirket profillerini içeren doğrulanmış nihai araştırma raporu.",
                agent=reviewer_agent,
                context=parallel_tasks,
            )
        else:
            reviewer_aider_tool = AiderExecutionTool(
                default_working_dir=self.workspace_dir,
                tier="reviewer",
                agent_idx=1,
                agent_role="Lead Reviewer",
            )
            reviewer_agent = Agent(
                role="Lead Code Reviewer & Systems Integrator",
                goal=(
                    "Paralel tamamlanan tum modulleri incelemek, import cakismalarini gidermek, "
                    "fonksiyonlarin ana donguye bagli oldugunu dogrulamak ve nihai entegrasyon raporunu sunmak."
                ),
                backstory=(
                    "Sen kod kalitesi, tutarlilik ve sistem entegrasyonundan sorumlu kidemli bas denetleyicisin. "
                    "Farkli uzmanlarin yazdigi kodlarin puruzsuz sekilde calismasini saglarsin."
                ),
                llm=reviewer_llm,
                tools=[reviewer_aider_tool],
                verbose=True,
                allow_delegation=False,
            )
            review_task = Task(
                description=(
                    f"Tüm paralel uzmanların yazdığı kodları incele:\n"
                    f"{spec_context_str}\n"
                    f"1. Modüller arasındaki import ve veri uyumunu denetle.\n"
                    f"2. Tanımlanan fonksiyonların ana döngüye çağrıldığından emin ol.\n"
                    f"3. Eksik veya çelişkili bir parça varsa Aider ile düzelt.\n"
                    f"4. Nihai sistem entegrasyon özetini raporla.\n"
                    f"Hedef Çalışma Dizini: {self.workspace_dir}"
                ),
                expected_output="Entegrasyon denetimi raporu ve düzeltme özeti.",
                agent=reviewer_agent,
            )

        return parallel_agents, parallel_tasks, reviewer_agent, review_task
