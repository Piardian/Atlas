import os
from dotenv import load_dotenv
from crewai import Agent, LLM
from .tools.aider_tool import AiderExecutionTool
from .tools.key_manager import GeminiKeyPool

load_dotenv()

def get_gemini_llm(model_name: str, api_key: str) -> LLM:
    """
    Google AI Studio (Gemini / Gemma) yapilandirmali CrewAI LLM nesnesi olusturur.
    """
    if not api_key:
        raise ValueError("Gecerli bir Gemini API anahtari saglanmadi.")
    
    formatted_model = model_name if model_name.startswith("gemini/") else f"gemini/{model_name}"
    
    return LLM(
        model=formatted_model,
        api_key=api_key,
        temperature=0.2,
    )


def create_agents(workspace_dir: str):
    """
    4 Gemini API anahtarini 5 paralel gelistirici + mimar + denetleyici ajanlara
    rate limit asimini onleyecek sekilde paylastirarak CrewAI ajanlarini olusturur.
    """
    key_pool = GeminiKeyPool()
    all_keys = key_pool.get_all_keys()
    
    if not all_keys:
        raise ValueError("GeminiKeyPool icerisinde hicbir Gemini API anahtari bulunamadi!")

    # Model tanimlari (gemini-3.8-flash-lite ve gemini-3.8-flash)
    planner_model_name = os.getenv("PLANNER_MODEL", "gemini/gemini-3.8-flash")
    coder_model_name = os.getenv("CODER_MODEL", "gemini/gemini-3.8-flash-lite")
    reviewer_model_name = os.getenv("REVIEWER_MODEL", "gemini/gemini-3.8-flash")

    # Ajanlara ozel API anahtarlarini tahsis et
    arch_key = key_pool.get_key_for_agent("architect")
    core_key = key_pool.get_key_for_agent("core_dev")
    service_key = key_pool.get_key_for_agent("service_dev")
    interface_key = key_pool.get_key_for_agent("interface_dev")
    qa_key = key_pool.get_key_for_agent("qa_dev")
    devops_key = key_pool.get_key_for_agent("devops_doc")
    reviewer_key = key_pool.get_key_for_agent("reviewer")

    # Ajan LLM nesneleri
    planner_llm = get_gemini_llm(planner_model_name, arch_key)
    core_llm = get_gemini_llm(coder_model_name, core_key)
    service_llm = get_gemini_llm(coder_model_name, service_key)
    interface_llm = get_gemini_llm(coder_model_name, interface_key)
    qa_llm = get_gemini_llm(coder_model_name, qa_key)
    devops_llm = get_gemini_llm(coder_model_name, devops_key)
    reviewer_llm = get_gemini_llm(reviewer_model_name, reviewer_key)

    # Ajanlara ozel Aider Araclari (her biri kendi API anahtariyla calisir)
    core_aider_tool = AiderExecutionTool(
        default_working_dir=workspace_dir,
        default_model=coder_model_name,
        assigned_api_key=core_key,
        agent_role="core_dev",
    )

    service_aider_tool = AiderExecutionTool(
        default_working_dir=workspace_dir,
        default_model=coder_model_name,
        assigned_api_key=service_key,
        agent_role="service_dev",
    )

    interface_aider_tool = AiderExecutionTool(
        default_working_dir=workspace_dir,
        default_model=coder_model_name,
        assigned_api_key=interface_key,
        agent_role="interface_dev",
    )

    qa_aider_tool = AiderExecutionTool(
        default_working_dir=workspace_dir,
        default_model=coder_model_name,
        assigned_api_key=qa_key,
        agent_role="qa_dev",
    )

    devops_aider_tool = AiderExecutionTool(
        default_working_dir=workspace_dir,
        default_model=coder_model_name,
        assigned_api_key=devops_key,
        agent_role="devops_doc",
    )

    reviewer_aider_tool = AiderExecutionTool(
        default_working_dir=workspace_dir,
        default_model=reviewer_model_name,
        assigned_api_key=reviewer_key,
        agent_role="reviewer",
    )

    # 1. Bas Mimar / Planlayici Ajan
    architect_agent = Agent(
        role="Lead Software Architect & Task Decomposer",
        goal=(
            "Kullanicidan gelen karmasik yazilim gelistirme talebini analiz etmek, moduler mimariyi tasarlamak ve "
            "isi 5 bagimsiz, paralel gelistirilebilecek alt goreve ayirmak."
        ),
        backstory=(
            "Sen 15+ yil deneyimli bir Bas Mimarsin. Google Gemini'nin gucuyle buyuk yazilim projelerini "
            "cakisma riski olmadan paralel yurutulebilecek net, sinirlari cizilmis modullere "
            "(Veri Modelleri, Is Mantigi/Servisler, API/Arayuz, Testler, Dokumantasyon/Yapilandirma) bolersin."
        ),
        llm=planner_llm,
        verbose=True,
        allow_delegation=False,
    )

    # 2. Paralel Gelistirici 1: Veri Modelleri & Semalar
    core_dev_agent = Agent(
        role="Core Data Models & Schemas Specialist",
        goal=(
            "Projenin temel veri modellerini, veritabani semalarini, DTO / Pydantic modellerini "
            "Aider aracini kullanarak yerel dosyalara eksiksiz ve hatasiz kodlamak."
        ),
        backstory=(
            "Sen veritabani semalari ve veri modelleme konusunda uzmanlasmis kidemli bir muhendissin. "
            "Gorevleri dogrudan yerel dosyada hayata gecirmek icin Aider aracini yetkinlikle kullanirsin."
        ),
        llm=core_llm,
        tools=[core_aider_tool],
        verbose=True,
        allow_delegation=False,
    )

    # 3. Paralel Gelistirici 2: Is Mantigi & Servis Katmani
    service_dev_agent = Agent(
        role="Business Logic & Core Services Specialist",
        goal=(
            "Uygulamanin ana is mantigini (business logic), algoritma ve servis siniflarini "
            "Aider aracini kullanarak yerel dosyalarda eksiksiz insa etmek."
        ),
        backstory=(
            "Sen yuksek performansli servis katmanlari, is kurallari ve backend algoritmalari yazmada "
            "uzman bir yazilim muhendisisin. Aider araci ile dogrudan yerel kod tabanini insa edersin."
        ),
        llm=service_llm,
        tools=[service_aider_tool],
        verbose=True,
        allow_delegation=False,
    )

    # 4. Paralel Gelistirici 3: API & CLI Arayuz Katmani
    interface_dev_agent = Agent(
        role="API & User Interface / CLI Specialist",
        goal=(
            "Uygulamanin dis dunya ile iletisim kuracagi REST API endpoint'lerini, yonlendiricileri (routers) "
            "veya CLI komut satiri arayuzunu Aider aracini kullanarak olusturmak."
        ),
        backstory=(
            "Sen modern REST/GraphQL API'lar, FastAPI/Flask endpoint'leri ve CLI araclari gelistiren bir arayuz uzmanisin. "
            "Servislerle uyumlu calisan uc noktalari Aider vasitasiyla hizlica kodlarsin."
        ),
        llm=interface_llm,
        tools=[interface_aider_tool],
        verbose=True,
        allow_delegation=False,
    )

    # 5. Paralel Gelistirici 4: Test & Kalite Guvencesi (QA)
    qa_dev_agent = Agent(
        role="Quality Assurance & Unit Test Specialist",
        goal=(
            "Gelistirilen modeller, servisler ve API'lar icin kapsamli pytest/unittest birim ve entegrasyon testlerini "
            "Aider aracini kullanarak test dosyalarina yazmak."
        ),
        backstory=(
            "Sen test kapsamini hedefleyen, mock'lama ve assertion stratejilerinde usta bir QA muhendisisin. "
            "Test senaryolarini yerel test dosyalarina Aider ile uygularsin."
        ),
        llm=qa_llm,
        tools=[qa_aider_tool],
        verbose=True,
        allow_delegation=False,
    )

    # 6. Paralel Gelistirici 5: Yapilandirma & Dokumantasyon (DevOps)
    devops_doc_agent = Agent(
        role="DevOps, Configuration & Documentation Specialist",
        goal=(
            "Projenin calistirilabilir olmasi icin gereken requirements.txt/pyproject.toml, .env.example, "
            "Docker veya ayar dosyalarini ve detayli README.md dokumantasyonunu Aider araciyla olusturmak."
        ),
        backstory=(
            "Sen CI/CD, paketleme ve teknik dokumantasyon uzmanisin. Projenin tek komutla ayaga kalkabilmesi "
            "icin gerekli tum konfigurasyonu Aider ile yerel projeye eklersin."
        ),
        llm=devops_llm,
        tools=[devops_aider_tool],
        verbose=True,
        allow_delegation=False,
    )

    # 7. Denetleyici & Entegrasyon Ajani (Reviewer)
    reviewer_agent = Agent(
        role="Lead Code Reviewer & Integration Inspector",
        goal=(
            "Paralel olarak tamamlanan tum modulleri incelemek, import ve bagimlilik cakismalarini kontrol etmek, "
            "gerekiyorsa kucuk duzeltmeler icin Aider'i tetiklemek ve nihai dogrulama raporunu sunmak."
        ),
        backstory=(
            "Sen kod kalitesi, tutarlilik ve sistem entegrasyonundan sorumlu bas denetleyicisin. "
            "Google Gemini gucuyle farkli ajanlarin yazdigi kod parcalarinin birbiriyle puruzsuz entegre olmasini garanti edersin."
        ),
        llm=reviewer_llm,
        tools=[reviewer_aider_tool],
        verbose=True,
        allow_delegation=False,
    )

    return {
        "architect": architect_agent,
        "core_dev": core_dev_agent,
        "service_dev": service_dev_agent,
        "interface_dev": interface_dev_agent,
        "qa_dev": qa_dev_agent,
        "devops_doc": devops_doc_agent,
        "reviewer": reviewer_agent,
    }
