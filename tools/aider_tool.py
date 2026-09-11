import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from .key_manager import SmartFallbackRouter, ARCHITECT_CASCADE, WORKER_CASCADE


class AiderToolInput(BaseModel):
    instruction: str = Field(
        ...,
        description="Aider'in yerel kod tabaninda yapmasini istediginiz detayli gelistirme/duzenleme talimati.",
    )
    target_files: str = Field(
        default="",
        description="Aider'in olusturmasi veya duzenlemesi gereken hedef dosya yollari (bosluk veya virgulle ayrilmis, orn: 'models/user.py models/task.py').",
    )
    working_directory: Optional[str] = Field(
        default=None,
        description="Kodun yazilacagi hedef proje kok dizini (varsayilan: aktif hedef workspace).",
    )


class AiderExecutionTool(BaseTool):
    name: str = "Aider Code Execution Tool"
    description: str = (
        "Yerel dosya sisteminde dogrudan kod yazmak, dosyalari olusturmak veya duzenlemek icin Aider CLI'i calistirir. "
        "Akilli kademeli model ve 4'lu API anahtari zirhi ile calisir, rate limit durumunda otomatik yedek modele duser."
    )
    args_schema: Type[BaseModel] = AiderToolInput
    default_working_dir: str = "./workspace_project"
    tier: str = "worker"
    agent_idx: int = 0
    agent_role: str = "general"

    def __init__(
        self,
        default_working_dir: str = "./workspace_project",
        tier: str = "worker",
        agent_idx: int = 0,
        agent_role: str = "general",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.default_working_dir = default_working_dir
        self.tier = tier
        self.agent_idx = agent_idx
        self.agent_role = agent_role

    def _ensure_git_repo(self, work_dir: Path) -> None:
        """Aider'in sorunsuz calismasi icin calisma dizininin git reposu olmasini garanti eder."""
        work_dir.mkdir(parents=True, exist_ok=True)
        git_dir = work_dir / ".git"
        if not git_dir.exists():
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                subprocess.run(
                    ["git", "config", "user.name", "Gemini Aider Agent"],
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "config", "user.email", "gemini-aider@orchestrator.local"],
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "commit", "--allow-empty", "-m", "Initial commit by GeminiAiderTool"],
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                print(f"[AiderTool] Git init uyarisi: {e}")

    def _find_aider_cmd(self) -> list:
        """Aider calistirma komutunu bulur."""
        if sys.platform == "win32":
            scripts_aider = Path(sys.prefix) / "Scripts" / "aider.exe"
            if scripts_aider.exists():
                return [str(scripts_aider)]
        return ["aider"]

    def _is_rate_limit_error(self, output_text: str) -> bool:
        """Aider ciktisinda kota / rate limit / model hatasi olup olmadigini algilar."""
        indicators = [
            "429",
            "RESOURCE_EXHAUSTED",
            "Quota exceeded",
            "quota_exceeded",
            "RateLimitError",
            "exceeded your current quota",
            "503 Service Unavailable",
            "experiencing high demand",
            "disconnected",
            "server disconnected",
            "remotedisconnected",
            "connection error",
            "timeout",
            "internalservererror",
            "geminiexception",
            "vertex ai",
        ]
        return any(ind.lower() in output_text.lower() for ind in indicators)

    def _run(
        self,
        instruction: str,
        target_files: str = "",
        working_directory: Optional[str] = None,
        **kwargs,
    ) -> str:
        target_work_dir = Path(working_directory or self.default_working_dir).resolve()
        target_work_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_git_repo(target_work_dir)

        router = SmartFallbackRouter()

        # Hedef dosyalari ayristir
        file_list = []
        if target_files:
            raw_tokens = [f.strip() for f in target_files.replace(",", " ").split() if f.strip()]
            for token in raw_tokens:
                file_path = target_work_dir / token
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_list.append(str(file_path.relative_to(target_work_dir)))

        aider_base = self._find_aider_cmd()
        max_retries = 6  # 4 anahtar x 3 model arasinda akilli gecis
        last_error_summary = ""

        for attempt in range(max_retries):
            try:
                # Modeli ve anahtari DAIMA SmartFallbackRouter'dan al, ajanin gpt-4o gibi gecersiz modeller vermesine izin verme!
                selected_model, gemini_api_key = router.get_best_model_and_key(
                    tier=self.tier,
                    preferred_agent_idx=self.agent_idx
                )

            except RuntimeError as quota_err:
                return f"HATA: {str(quota_err)}"

            # Cevre degiskenlerini hazirla
            env = os.environ.copy()
            env["GEMINI_API_KEY"] = gemini_api_key
            env["AIDER_CHECK_UPDATE"] = "false"
            env["AIDER_SHOW_RELEASE_NOTES"] = "false"

            cmd = aider_base + [
                "--model",
                selected_model,
                "--yes-always",
                "--no-check-update",
                "--no-show-release-notes",
                "--no-browser",
                "--no-show-model-warnings",
                "--no-suggest-shell-commands",
                "--message",
                instruction,
            ]

            if file_list:
                cmd.extend(file_list)

            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(target_work_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,  # 60 saniye ile kesin hiz siniri
                    encoding="utf-8",
                    errors="replace",
                )

                combined_output = (result.stdout or "") + "\n" + (result.stderr or "")

                # Eger Rate Limit / Internal Server / Kota hatasi alindiysa -> Bu anahtari/modeli isaretle ve bir sonrakini dene!
                if result.returncode != 0 and self._is_rate_limit_error(combined_output):
                    router.mark_exhausted(selected_model, gemini_api_key, reason="Aider 429/InternalError Hatasi")
                    last_error_summary = combined_output[-300:]
                    continue  # Donguye devam et, siradaki kademeyi calistir

                # Git durumunu incele
                git_status = subprocess.run(
                    ["git", "status", "--short"],
                    cwd=str(target_work_dir),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                output_summary = []
                output_summary.append(f"=== Aider Islem Sonucu ({selected_model} | Ajan: {self.agent_role}) ===")
                output_summary.append(f"Cikis Kodu: {result.returncode}")
                
                if result.stdout:
                    stdout_lines = result.stdout.strip().splitlines()
                    output_summary.append("\n[Aider CLI Ciktisi (Ozet)]:")
                    output_summary.extend(stdout_lines[-25:])
                    
                if result.stderr and result.returncode != 0:
                    output_summary.append(f"\n[Aider Hata Ciktisi]:\n{result.stderr[-400:]}")

                if git_status.stdout:
                    output_summary.append(f"\n[Git Durumu / Degisen Dosyalar]:\n{git_status.stdout.strip()}")
                else:
                    output_summary.append("\n[Git Durumu]: Calisma agaci guncellendi.")

                return "\n".join(output_summary)

            except subprocess.TimeoutExpired:
                router.mark_exhausted(selected_model, gemini_api_key, reason="Zaman Asimi (120s)")
                continue
            except Exception as e:
                return f"HATA: Aider calistirilirken beklenmeyen hata: {str(e)}"

        return f"HATA: Tum model ve anahtar kademeleri denendi ancak tamamlanamadi.\nSon Hata: {last_error_summary}"
