import json
import re
from typing import Dict, Any
import litellm
from .tools.key_manager import SmartFallbackRouter


class SpecificationExpander:
    """
    Kullanıcının girdiği kısa, genel veya eksik promptları derinlemesine analiz eden,
    platform gereksinimlerini, sürekli olay döngülerini (event loop) ve uç durumları (edge cases)
    kurumsal bir Teknik Şartnameye (Architectural Specification) dönüştüren motor.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.router = SmartFallbackRouter()

    def expand_specification(self, user_prompt: str) -> Dict[str, Any]:
        """Kullanıcı promptunu kurumsal teknik şartnameye genişletir."""
        model, api_key = self.router.get_best_model_and_key(tier="architect")

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
                    {"role": "user", "content": f"Kullanıcı Talebi:\n'''{user_prompt}'''"}
                ],
                temperature=0.2,
                timeout=25,
                num_retries=0,
            )
            text_resp = response.choices[0].message.content
            
            # JSON Ayıklama
            json_match = re.search(r"\{[\s\S]*\}", text_resp)
            if json_match:
                spec = json.loads(json_match.group(0))
                if "expanded_technical_prompt" in spec:
                    return spec
        except Exception as e:
            print(f"[SpecExpander] Uyarı: LLM şartname genişletme hatası ({e}). Akıllı genel şartnameye geçiliyor.")

        # Akıllı Genel Fallback Şartnamesi
        return {
            "project_title": "Autonomous Enterprise Project",
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
                "İş parçacığı ve asenkron yarış durumlarının (race condition) önlenmesi"
            ],
            "verification_commands": ["python run.py"]
        }
