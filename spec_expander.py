import json
import re
from typing import Dict, Any
import litellm
from .tools.key_manager import SmartFallbackRouter


class SpecificationExpander:
    """
    Kullanıcının girdiği kısa, genel veya eksik promptları derinlemesine analiz eden,
    görevin doğasını (Web Araştırma / Yazılım Geliştirme / Hibrit) algılayan ve
    kurumsal bir Teknik Şartnameye (Architectural Specification) dönüştüren motor.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.router = SmartFallbackRouter()

    @staticmethod
    def detect_task_mode(user_prompt: str) -> str:
        """Prompt içeriğinden görevin 'web_research', 'software_engineering' veya 'hybrid' olduğunu algılar."""
        p_low = user_prompt.lower()
        research_keywords = [
            "araştır", "arastir", "web site", "şirketlerini bul", "sirketlerini bul",
            "iletişim bilgilerini çıkar", "iletisim bilgilerini cikar", "tablo halinde raporla",
            "browser", "tarayıcı", "tarayici", "web research", "scrape", "ürünlerini çıkar",
            "linkedin", "iş ilan", "is ilan", "staj ilan", "kariyer", "cv", "karşıma getir", "karsima getir",
        ]
        coding_keywords = [
            "fastapi", "endpoint", "veritabanı", "database", "mql5", "ea geliştir",
            "uygulama geliştir", "bot geliştir", "pytest", "unit test", "backtest",
        ]

        has_research = any(k in p_low for k in research_keywords)
        has_coding = any(k in p_low for k in coding_keywords)

        if has_research and has_coding:
            return "hybrid"
        if has_research:
            return "web_research"
        return "software_engineering"

    def expand_specification(self, user_prompt: str) -> Dict[str, Any]:
        """Kullanıcı promptunu kurumsal teknik veya araştırma şartnamesine genişletir."""
        detected_mode = self.detect_task_mode(user_prompt)
        model, api_key = self.router.get_best_model_and_key(tier="architect")

        if detected_mode == "web_research":
            system_instruction = (
                "Sen Kıdemli bir AI Web Araştırma ve Orkestrasyon Mimarısın.\n"
                "Kullanıcı sana gerçek web tarayıcısı (Browser Use / Chromium) kullanılarak yürütülecek bir araştırma görevi verdi.\n"
                "Görevi; hedef kaynakların keşfi, web sitelerinden gerçek ürün ve iletişim verilerinin çıkarılması, "
                "doğrulanması ve kurumsal Markdown/JSON tablo raporuna dönüştürülmesi şeklinde yapılandır.\n\n"
                "Çıktını SADECE aşağıdaki geçerli JSON formatında ver:\n"
                "{\n"
                '  "project_title": "AI Web Research Mission",\n'
                '  "task_type": "web_research",\n'
                '  "target_platform": "Atlas Browser Use + Structured Report",\n'
                '  "event_loop_model": "multi_agent_browser_pipeline",\n'
                '  "expanded_technical_prompt": "Detaylandırılmış web araştırma, DOM veri çıkarımı ve tablo raporlama talimatı",\n'
                '  "mandatory_edge_cases": ["Erişilemeyen URL durumunda alternatif kaynak kontrolü", "Eksik iletişim bilgisinde resmi iletişim sayfası ziyareti", "Halüsinasyon önleme ve gerçek URL kanıtı"],\n'
                '  "verification_commands": []\n'
                "}"
            )
        else:
            system_instruction = (
                "Sen 20 yıllık deneyime sahip Kıdemli Bir Sistem Mimarısın.\n"
                "Kullanıcı sana kısa veya genel bir yazılım talebi verdiğinde, bunu yüzeysel bir betik olarak değil, "
                "gerçek dünyada çalışan kurumsal bir sistem olarak tasarlamalısın.\n\n"
                "Özellikle şunlara dikkat et:\n"
                "1. PLATFORM AYRIMI: Eğer MetaTrader, Web, Blockchain, Donanım gibi özel platformlar varsa, "
                "sadece Python değil, platformun kendi yerel dillerini (.mq5, .sol, .rs, .html vb.) de plana dahil et.\n"
                "2. SÜREKLİ OLAY DÖNGÜSÜ (EVENT LOOP): Asla tek seferlik statik örnek çalıştırma ('sample_data'); "
                "sürekli çalışan bir olay döngüsü (while loop, WebSocket, Polling, OnTick) şart koş.\n"
                "3. UÇ DURUMLAR (EDGE CASES): Sıfıra bölme, ağ kopması, veri eksikliği, rate-limit ve hata kurtarma mekanizmalarını zorunlu kıl.\n"
                "4. DOĞRULANABİLİRLİK: Sistemin fiilen çalıştırılabileceği 'run.py' ve kapsamlı birim testlerini şart koş.\n\n"
                "Çıktını SADECE aşağıdaki geçerli JSON formatında ver:\n"
                "{\n"
                '  "project_title": "Proje Başlığı",\n'
                '  "task_type": "' + detected_mode + '",\n'
                '  "target_platform": "Python / MT5 MQL5 / Web Full-Stack / vb.",\n'
                '  "event_loop_model": "continuous_listener / polling_loop / event_driven / etc.",\n'
                '  "expanded_technical_prompt": "Detaylı, uç durumları kapsayan genişletilmiş mimari talimat",\n'
                '  "mandatory_edge_cases": ["Uç durum 1", "Uç durum 2", "Uç durum 3"],\n'
                '  "verification_commands": ["python -m unittest discover tests", "python run.py"]\n'
                "}"
            )

        try:
            response = litellm.completion(
                model=model,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Kullanıcı Talebi:\n'''{user_prompt}'''"},
                ],
                temperature=0.2,
                timeout=25,
                num_retries=0,
            )
            text_resp = response.choices[0].message.content

            json_match = re.search(r"\{[\s\S]*\}", text_resp)
            if json_match:
                spec = json.loads(json_match.group(0))
                if "expanded_technical_prompt" in spec:
                    spec.setdefault("task_type", detected_mode)
                    return spec
        except Exception as e:
            print(f"[SpecExpander] Uyarı: LLM şartname genişletme hatası ({e}). Akıllı genel şartnameye geçiliyor.")

        if detected_mode == "web_research":
            return {
                "project_title": "Atlas AI Web Research Mission",
                "task_type": "web_research",
                "target_platform": "Browser Use + Chromium Automation",
                "event_loop_model": "multi_agent_browser_pipeline",
                "expanded_technical_prompt": (
                    f"{user_prompt}\n\n"
                    "ARAŞTIRMA KURALLARI:\n"
                    "- Gerçek tarayıcı (Browser Automation Tool) kullanarak hedef şirketlerin resmi web sitelerini ziyaret et.\n"
                    "- Her şirketin ana ürünlerini/çözümlerini ve resmi iletişim bilgilerini (web sitesi, e-posta, adres/konum) çıkar.\n"
                    "- Sonuçları doğrulanmış Markdown tablosu olarak raporla."
                ),
                "mandatory_edge_cases": [
                    "Resmi web sitesi URL doğrulaması",
                    "İletişim sayfası (/contact veya /iletisim) alt link taraması",
                    "Eksik alanların şeffaf biçimde belirtilmesi (halüsinasyon yasağı)",
                ],
                "verification_commands": [],
            }

        return {
            "project_title": "Autonomous Enterprise Project",
            "task_type": detected_mode,
            "target_platform": "Python Multi-Module Architecture",
            "event_loop_model": "continuous_event_driven",
            "expanded_technical_prompt": (
                f"{user_prompt}\n\n"
                "MİMARİ ZORUNLULUKLAR:\n"
                "- Statik tek seferlik çalıştırma yerine canlı olay döngüsü (event loop) kur.\n"
                "- Tüm uç durumları (sıfıra bölme, ağ kopması, veri eksikliği) try-except bloklarıyla yakala.\n"
                "- Tek komutla çalışan 'run.py' ve bağımsız birim testleri (tests/) oluştur."
            ),
            "mandatory_edge_cases": [
                "Ağ ve sunucu kesintisi durumunda otomatik kurtarma",
                "Boş veya geçersiz girdi durumunda güvenli varsayılanlar",
                "İş parçacığı ve asenkron yarış durumlarının (race condition) önlenmesi",
            ],
            "verification_commands": ["python run.py"],
        }
