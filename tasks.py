from typing import Dict, List, Tuple
from crewai import Task, Agent


def create_tasks(
    agents: Dict[str, Agent],
    user_prompt: str,
    workspace_dir: str,
) -> Tuple[List[Task], Task, Task]:
    """
    CrewAI görevlerini oluşturur:
    1. Planlama Görevi (Sequential)
    2. 5 Paralel Kodlama Görevi (async_execution=True)
    3. Entegrasyon & Doğrulama Görevi (Sequential, context=parallel_tasks)
    """

    # 1. Planlama & Mimari Ayrıştırma Görevi
    planning_task = Task(
        description=(
            f"Kullanıcının şu geliştirme talebini derinlemesine analiz et:\n\n"
            f"'''{user_prompt}'''\n\n"
            f"Hedef Çalışma Dizini: {workspace_dir}\n\n"
            f"Yapılması Gerekenler:\n"
            f"1. Modüler bir dosya ve paket hiyerarşisi oluştur.\n"
            f"2. Çakışma olmayacak şekilde 5 alt göreve ayır:\n"
            f"   - Modül 1 (Core Models): Veri modelleri, veritabanı şemaları (hedef dosyaları net belirt).\n"
            f"   - Modül 2 (Services/Logic): İş kuralları, veri erişimi ve servis fonksiyonları.\n"
            f"   - Modül 3 (API/Interface): HTTP route'ları, CLI komutları veya uç noktalar.\n"
            f"   - Modül 4 (QA/Tests): Birim testleri ve senaryoları.\n"
            f"   - Modül 5 (DevOps/Doc): requirements.txt, konfigürasyon ve README.md.\n"
            f"3. Her modül için kesin dosya yollarını ve Aider'a verilecek talimat metnini net olarak listele."
        ),
        expected_output=(
            "5 modül için detaylı dosya yolları, fonksiyon/sınıf tanımları ve "
            "her ajanın Aider CLI aracılığıyla oluşturacağı spesifik geliştirme talimatlarını içeren mimari plan."
        ),
        agent=agents["architect"],
    )

    # 2. Paralel Görev 1: Veri Modelleri (async_execution=True)
    core_dev_task = Task(
        description=(
            f"Mimari plandaki 'Modül 1: Core Models & Schemas' kısmını hayata geçir.\n"
            f"Kullanıcı Talebi: {user_prompt}\n"
            f"Aider Code Execution Tool'u kullanarak ilgili modelleri ve şemaları yerel dosyalara kodla.\n"
            f"Hedef Çalışma Dizini: {workspace_dir}\n"
            f"İpucu: Aider aracına net hedef dosya yollarını ve detaylı kodlama talimatını ilet."
        ),
        expected_output=(
            "Aider aracı ile oluşturulmuş veri modelleri ve dosya oluşturma çıktısı özeti."
        ),
        agent=agents["core_dev"],
        async_execution=True,
    )

    # 3. Paralel Görev 2: İş Mantığı & Servisler (async_execution=True)
    service_dev_task = Task(
        description=(
            f"Mimari plandaki 'Modül 2: Business Logic & Services' kısmını hayata geçir.\n"
            f"Kullanıcı Talebi: {user_prompt}\n"
            f"Aider Code Execution Tool'u kullanarak servis sınıflarını ve ana fonksiyonları yerel dosyalara kodla.\n"
            f"Hedef Çalışma Dizini: {workspace_dir}"
        ),
        expected_output=(
            "Aider aracı ile oluşturulmuş servis katmanı dosyaları ve işlem özeti."
        ),
        agent=agents["service_dev"],
        async_execution=True,
    )

    # 4. Paralel Görev 3: API & CLI Uç Noktaları (async_execution=True)
    interface_dev_task = Task(
        description=(
            f"Mimari plandaki 'Modül 3: API & Interface' kısmını hayata geçir.\n"
            f"Kullanıcı Talebi: {user_prompt}\n"
            f"Aider Code Execution Tool'u kullanarak endpoint yönlendiricilerini (routers/controllers) "
            f"veya CLI arayüzünü yerel dosyalara kodla.\n"
            f"Hedef Çalışma Dizini: {workspace_dir}"
        ),
        expected_output=(
            "Aider aracı ile oluşturulmuş API/CLI arayüz dosyaları ve işlem özeti."
        ),
        agent=agents["interface_dev"],
        async_execution=True,
    )

    # 5. Paralel Görev 4: Testler (async_execution=True)
    qa_dev_task = Task(
        description=(
            f"Mimari plandaki 'Modül 4: QA & Tests' kısmını hayata geçir.\n"
            f"Kullanıcı Talebi: {user_prompt}\n"
            f"Aider Code Execution Tool'u kullanarak kapsamlı test dosyalarını kodla.\n"
            f"Hedef Çalışma Dizini: {workspace_dir}"
        ),
        expected_output=(
            "Aider aracı ile oluşturulmuş birim ve entegrasyon test dosyaları ve işlem özeti."
        ),
        agent=agents["qa_dev"],
        async_execution=True,
    )

    # 6. Paralel Görev 5: DevOps, Konfigürasyon ve Dokümantasyon (async_execution=True)
    devops_doc_task = Task(
        description=(
            f"Mimari plandaki 'Modül 5: DevOps & Documentation' kısmını hayata geçir.\n"
            f"Kullanıcı Talebi: {user_prompt}\n"
            f"Aider Code Execution Tool'u kullanarak requirements.txt, .env.example ve kapsamlı README.md oluştur.\n"
            f"Hedef Çalışma Dizini: {workspace_dir}"
        ),
        expected_output=(
            "Aider aracı ile oluşturulmuş konfigürasyon, paket ve dokümantasyon dosyaları özeti."
        ),
        agent=agents["devops_doc"],
        async_execution=True,
    )

    parallel_dev_tasks = [
        core_dev_task,
        service_dev_task,
        interface_dev_task,
        qa_dev_task,
        devops_doc_task,
    ]

    # 7. Denetim & Entegrasyon Görevi (Tüm paralel işlerin çıktısını toplar)
    review_task = Task(
        description=(
            f"Tüm paralel geliştiricilerin ({len(parallel_dev_tasks)} ajan) tamamladığı kodları incele.\n"
            f"Hedef Çalışma Dizini: {workspace_dir}\n"
            f"Yapılacaklar:\n"
            f"1. Paralel ajanların oluşturduğu dosya ve fonksiyonların birbiriyle uyumlu (import hatasız) olduğunu doğrula.\n"
            f"2. Eğer ufak bir uyumsuzluk veya eksik varsa, Aider Code Execution Tool ile hızlıca düzeltme yap.\n"
            f"3. Projenin genel durumunu, oluşturulan dosya listesini ve çalıştırma komutlarını içeren nihai bir entegrasyon raporu hazırla."
        ),
        expected_output=(
            "Oluşturulan tüm dosyaların listesi, entegrasyon durumu, doğrulama notları ve "
            "kullanıcının projeyi nasıl çalıştıracağını anlatan kapsamlı özet rapor."
        ),
        agent=agents["reviewer"],
        context=parallel_dev_tasks,  # 5 paralel görevin tamamlanmasını bekler ve çıktılarını alır
    )

    return planning_task, parallel_dev_tasks, review_task
