import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

FALLBACK_ENV_PATH = Path(r"C:\Users\piard\Downloads\orkestrat-r-test9-fixed\agent-core\.env")

# 1. KURAL: Bas Mimar 3.8 -> 3.7 -> 3.6 -> 3.5
ARCHITECT_CASCADE = [
    "gemini/gemini-3.8-flash",
    "gemini/gemini-3.7-flash",
    "gemini/gemini-3.6-flash",
    "gemini/gemini-3.5-flash-lite",
]

# 2. KURAL: Calisanlar 3.8 Flash Lite -> 3.5 Flash Lite -> 3.1 Flash Lite -> Gemma 4
WORKER_CASCADE = [
    "gemini/gemini-3.8-flash-lite",
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-3.1-flash-lite",
    "gemini/gemma-4-31b-it",
]


class SmartFallbackRouter:
    """
    Google AI Studio (Gemini) API anahtarlari ve modelleri icin
    akilli kademeli dusus (Cascading Fallback) ve Rate Limit zirhi saglayan yonetici.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SmartFallbackRouter, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, env_path: Optional[str] = None):
        if self._initialized:
            return

        self.keys: List[str] = []
        self.exhausted_pairs: Dict[Tuple[str, str], float] = {}
        self.cooldown_seconds = 180  # 3 dakika boyunca gecici kota asimi olan (model, key) pasif
        self._load_keys(env_path)
        self._initialized = True

    def _load_keys(self, custom_env_path: Optional[str] = None) -> None:
        if custom_env_path and Path(custom_env_path).exists():
            load_dotenv(custom_env_path, override=True)
        else:
            load_dotenv(override=False)

        if not any(k.startswith("GEMINI_USER_") for k in os.environ) and FALLBACK_ENV_PATH.exists():
            load_dotenv(FALLBACK_ENV_PATH, override=True)

        loaded_keys = []
        for letter in ["A", "B", "C", "D", "E", "F"]:
            key = os.getenv(f"GEMINI_USER_{letter}_KEY")
            if key and key.strip() and key.strip() not in loaded_keys:
                loaded_keys.append(key.strip())

        comma_keys = os.getenv("GEMINI_API_KEYS")
        if comma_keys:
            for k in comma_keys.split(","):
                clean = k.strip()
                if clean and clean not in loaded_keys:
                    loaded_keys.append(clean)

        single_key = os.getenv("GEMINI_API_KEY")
        if single_key and single_key.strip() and single_key.strip() not in loaded_keys:
            loaded_keys.append(single_key.strip())

        self.keys = loaded_keys

    def mark_exhausted(self, model: str, key: str, reason: str = "") -> None:
        with self._lock:
            pair = (model, key)
            self.exhausted_pairs[pair] = time.time()
            masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
            print(f"\n[Rate-Limit Zırhı] ⚠️ {model} modeli için Key ({masked_key}) 429/Kota sınırına ulaştı!")

    def is_exhausted(self, model: str, key: str) -> bool:
        pair = (model, key)
        if pair not in self.exhausted_pairs:
            return False
        if time.time() - self.exhausted_pairs[pair] > self.cooldown_seconds:
            del self.exhausted_pairs[pair]
            return False
        return True

    def get_best_model_and_key(self, tier: str = "worker", preferred_agent_idx: int = 0) -> Tuple[str, str]:
        """
        Model kademesinde:
        Once Model 1 icin Key 1 -> Key 2 -> Key 3 -> Key 4 denenir.
        Model 1 tum anahtarlarda biterse Model 2'ye (Key 1 -> 2 -> 3 -> 4) gecilir.
        """
        if not self.keys:
            raise ValueError("Hicbir Google Gemini API anahtari tanimli degil!")

        cascade = ARCHITECT_CASCADE if tier.lower() in ["architect", "reviewer", "lead"] else WORKER_CASCADE

        with self._lock:
            for model in cascade:
                num_keys = len(self.keys)
                ordered_keys = [self.keys[(preferred_agent_idx + i) % num_keys] for i in range(num_keys)]
                
                for key in ordered_keys:
                    if not self.is_exhausted(model, key):
                        return model, key

        tier_name = "Baş Mimar (3.8 -> 3.7 -> 3.6 -> 3.5)" if tier.lower() in ["architect", "reviewer"] else "Çalışanlar (3.8 -> 3.5 -> 3.1 -> Gemma 4)"
        raise RuntimeError(
            f"🛑 DIKKAT: {tier_name} kademesindeki tum API anahtarlarinin kotalari tukendi!\n"
            f"Lutfen kotalarinizin sifirlanmasini bekleyin veya yeni bir Gemini API anahtari ekleyin."
        )

    def get_all_keys(self) -> List[str]:
        return list(self.keys)


class GeminiKeyPool:
    def __init__(self, env_path: Optional[str] = None):
        self.router = SmartFallbackRouter(env_path)

    def get_key_for_agent(self, agent_role: str) -> str:
        role_map = {"architect": 0, "core_dev": 0, "service_dev": 1, "interface_dev": 2, "qa_dev": 3, "devops_doc": 0, "reviewer": 1}
        idx = role_map.get(agent_role.lower(), 0)
        tier = "architect" if agent_role.lower() in ["architect", "reviewer"] else "worker"
        _, key = self.router.get_best_model_and_key(tier=tier, preferred_agent_idx=idx)
        return key

    def get_all_keys(self) -> List[str]:
        return self.router.get_all_keys()
