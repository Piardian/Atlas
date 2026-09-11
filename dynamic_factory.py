import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from crewai import Agent, Task, LLM
import litellm
from .tools.aider_tool import AiderExecutionTool
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
    Gelen gorevin alanina (Finans, Web, Veri Bilimi, Guvenlik vb.) gore
    en uygun 5 uzman ajani ve gorevleri calisma aninda (runtime) dinamik ureten fabrika.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    def create_architect(self) -> Agent:
        """En ust kademedeki Bas Mimar ajanini olusturur (3.7 -> 3.6 -> 3.5)."""
        architect_llm = get_resilient_llm(tier="architect", agent_idx=0)
        return Agent(
            role="Lead System Architect & Dynamic Workforce Orchestrator",
            goal=(
                "Kullanicinin talebini alanina (Finans/Trading, Web, CLI, Veri Analizi, Guvenlik vb.) gore analiz etmek, "
                "en uygun 5 uzman rolu ve gorevlerini JSON formatinda dinamik olarak tasarlamak."
            ),
            backstory=(
                "Sen her turlu yazilim disiplininde (Trading/SMC, Web Full-Stack, Veri Muhendisligi, "
                "Mikroservisler) 20 yillik tecrubeye sahip bir Bas Mimarsin. Gelen gorevin tabiatina "
                "gore en dogru uzmanliklari belirler ve isi kusursuz 5 paralel alt modüle bolersin."
            ),
            llm=architect_llm,
            verbose=True,
            allow_delegation=False,
        )

    def parse_blueprint(self, blueprint_text: str, user_prompt: str) -> List[Dict[str, Any]]:
        """Bas Mimar'in urettigi JSON planini ayiklar; format hatasi olursa akilli fallback uygular."""
        try:
            json_match = re.search(r"\{[\s\S]*\}", blueprint_text)
            if json_match:
                data = json.loads(json_match.group(0))
                if "agents" in data and len(data["agents"]) >= 3:
                    return data["agents"]
        except Exception as e:
            print(f"[DynamicFactory] JSON ayiklama uyarisi: {e}. Akilli genel sablon kullaniliyor.")

        return [
            {
                "role": "Core Architecture & Data Specialist",
                "goal": f"{user_prompt} icin gerekli veri yapilarini, modelleri ve semalari kodlamak.",
                "backstory": "Veri yapilari ve temel modeller uzmani kidemli muhendis.",
                "target_files": "models.py database.py",
                "instruction": f"Projenin temel veri modellerini ve semalarini olustur: {user_prompt}",
            },
            {
                "role": "Core Logic & Algorithms Specialist",
                "goal": f"{user_prompt} icin ana is mantigini, algoritmik hesaplamalari ve servisleri kodlamak.",
                "backstory": "Yüksek performansli is mantigi, servis katmani ve algoritma gelistiricisi.",
                "target_files": "services.py logic.py",
                "instruction": f"Projenin ana is mantigini ve servis fonksiyonlarini olustur: {user_prompt}",
            },
            {
                "role": "Interface, CLI & Endpoints Specialist",
                "goal": f"{user_prompt} icin API rotalarini, CLI veya arayuz kodlarini olusturmak.",
                "backstory": "REST API, CLI ve modern arayuz mimarisi uzmani.",
                "target_files": "routers.py main.py",
                "instruction": f"Projenin arayuz ve API katmanini olustur: {user_prompt}",
            },
            {
                "role": "Quality Assurance & Unit Test Specialist",
                "goal": f"{user_prompt} icin birim testleri ve matematiksel dogrulama testlerini yazmak.",
                "backstory": "Kapsamli test mimarisi ve kenar durum guvenligi uzmani.",
                "target_files": "tests/test_core.py",
                "instruction": f"Projenin tum modulleri icin pytest birim testlerini olustur: {user_prompt}",
            },
            {
                "role": "DevOps, Automation & Documentation Specialist",
                "goal": f"{user_prompt} icin requirements.txt, baslatici scriptler ve README.md hazirlamak.",
                "backstory": "CI/CD, ortam yonetimi ve teknik dokumantasyon uzmani.",
                "target_files": "run.py requirements.txt README.md",
                "instruction": f"Projenin tek komutla calismasini saglayan run.py ve dokumantasyonunu hazirla: {user_prompt}",
            },
        ]

    def build_dynamic_crew_components(
        self,
        agents_blueprint: List[Dict[str, Any]],
        user_prompt: str,
        arch_spec: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Agent], List[Task], Agent, Task]:
        """
        Dinamik blueprint'ten CrewAI Ajanlarini ve async_execution=True Task'larini olusturur.
        arch_spec parametresi ile her ajana tam mimari baglam (context) enjekte eder.
        """
        parallel_agents: List[Agent] = []
        parallel_tasks: List[Task] = []

        spec_context_str = ""
        if arch_spec:
            spec_context_str = (
                f"\n--- TEKNİK ŞARTNAME VE MİMARİ KISITLAR ---\n"
                f"• Hedef Platform: {arch_spec.get('target_platform', 'Python')}\n"
                f"• Olay Döngüsü Modeli: {arch_spec.get('event_loop_model', 'continuous')}\n"
                f"• Zorunlu Uç Durumlar: {', '.join(arch_spec.get('mandatory_edge_cases', []))}\n"
                f"-----------------------------------------\n"
            )

        for idx, spec in enumerate(agents_blueprint[:5]):
            role_name = spec.get("role", f"Specialist Agent #{idx+1}")
            goal = spec.get("goal", f"Gorevi basariyla tamamlamak: {user_prompt}")
            backstory = spec.get("backstory", f"{role_name} alaninda uzman kidemli muhendis.")
            target_files = spec.get("target_files", "")
            instruction = spec.get("instruction", f"{role_name} kapsamindaki gorevleri tamamla.")

            agent_llm = get_resilient_llm(tier="worker", agent_idx=idx)
            aider_tool = AiderExecutionTool(
                default_working_dir=self.workspace_dir,
                tier="worker",
                agent_idx=idx,
                agent_role=role_name,
            )

            agent = Agent(
                role=role_name,
                goal=goal,
                backstory=backstory,
                llm=agent_llm,
                tools=[aider_tool],
                verbose=True,
                allow_delegation=False,
            )
            parallel_agents.append(agent)

            task = Task(
                description=(
                    f"GÖREVİN: {role_name}\n"
                    f"{spec_context_str}\n"
                    f"Kullanıcı Talebi: {user_prompt}\n"
                    f"Hedef Dosyalar: {target_files}\n"
                    f"Talimat: {instruction}\n\n"
                    f"KATI KURALLAR:\n"
                    f"- Tanımladığın tüm işlem fonksiyonlarını ana tetikleyiciye (OnTick/main/run) bağla.\n"
                    f"- Aider Code Execution Tool aracını kullanarak kodları doğrudan yerel dosyalara işle.\n"
                    f"Hedef Çalışma Dizini: {self.workspace_dir}"
                ),
                expected_output=f"{role_name} tarafından yerel dosyalara yazılan kodlar ve Aider işlem özeti.",
                agent=agent,
                async_execution=True,
            )
            parallel_tasks.append(task)

        # Lead Reviewer / Integrator Ajanı (3.7 -> 3.6 -> 3.5)
        reviewer_llm = get_resilient_llm(tier="reviewer", agent_idx=1)
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
                f"2. Tanımlanan fonksiyonların (örn. ExecuteNewsOrder, AsymmetricEngine) ana döngüye çağrıldığından emin ol.\n"
                f"3. Eksik veya çelişkili bir parça varsa Aider ile düzelt.\n"
                f"4. Nihai sistem entegrasyon özetini raporla.\n"
                f"Hedef Çalışma Dizini: {self.workspace_dir}"
            ),
            expected_output="Entegrasyon denetimi raporu ve düzeltme özeti.",
            agent=reviewer_agent,
        )

        return parallel_agents, parallel_tasks, reviewer_agent, review_task
