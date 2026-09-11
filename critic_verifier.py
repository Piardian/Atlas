import os
import sys
import subprocess
import py_compile
from pathlib import Path
from typing import Dict, Any, List, Tuple
from .tools.aider_tool import AiderExecutionTool


class RuntimeVerifier:
    """
    Yazılan kodları sadece metin olarak bırakmayan,
    terminalde fiilen AST/Sözdizimi kontrolü yapan, testleri çalıştıran,
    ve hata çıktığında Aider ile hızlıca kendi kendini düzelten (Self-Healing Critic) motor.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir).resolve()

    def check_syntax(self) -> Tuple[bool, str]:
        """Tüm Python dosyalarının sözdizimi (Syntax/AST) geçerliliğini derleyerek kontrol eder."""
        py_files = list(self.workspace_dir.rglob("*.py"))
        errors = []

        for py_file in py_files:
            if ".git" in py_file.parts or ".aider" in py_file.parts:
                continue
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"Sözdizimi Hatası [{py_file.name}]: {str(e)}")

        if errors:
            return False, "\n".join(errors)
        return True, "Tüm Python dosyaları başarıyla derlendi (Sözdizimi Hatası Yok)."

    def verify_entrypoints(self) -> Tuple[bool, str]:
        """run.py veya ana başlatıcı dosyaların geçerli Python kodu olduğunu doğrular."""
        run_file = self.workspace_dir / "run.py"
        if not run_file.exists():
            return True, "run.py bulunamadı (Özel başlatıcı mevcut)."

        try:
            content = run_file.read_text(encoding="utf-8", errors="replace").strip()
            if content.startswith("python ") or not ("import" in content or "def " in content or "__main__" in content):
                return False, f"HATA: run.py geçerli bir Python kodu değil, terminal komut metni içeriyor: '{content[:80]}'"
        except Exception as e:
            return False, f"run.py okuma hatası: {e}"

        return True, "run.py geçerli bir Python başlatıcı dosyası."

    def run_tests(self) -> Tuple[bool, str]:
        """Projedeki birim testlerini terminalde fiilen çalıştırır."""
        tests_dir = self.workspace_dir / "tests"
        if not tests_dir.exists():
            return True, "tests/ dizini bulunamadı (Birim testi yok)."

        try:
            cmd = [sys.executable, "-m", "unittest", "discover", "tests"]
            result = subprocess.run(
                cmd,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=15,
                encoding="utf-8",
                errors="replace",
            )

            if result.returncode != 0:
                combined_err = (result.stderr or "") + "\n" + (result.stdout or "")
                return False, f"Birim Testleri Çıktısı:\n{combined_err[-400:]}"
            return True, f"Birim testleri başarıyla geçti."

        except subprocess.TimeoutExpired:
            return False, "Birim testleri 15 saniyelik zaman aşımına uğradı."
        except Exception as e:
            return False, f"Test çalıştırma hatası: {e}"

    def run_full_verification(self) -> Dict[str, Any]:
        """Tüm kontrolleri sırayla çalıştırır ve detaylı bir doğrulama raporu üretir."""
        syntax_pass, syntax_msg = self.check_syntax()
        entry_pass, entry_msg = self.verify_entrypoints()
        tests_pass, tests_msg = self.run_tests()

        all_passed = syntax_pass and entry_pass and tests_pass
        return {
            "all_passed": all_passed,
            "syntax": {"pass": syntax_pass, "message": syntax_msg},
            "entrypoint": {"pass": entry_pass, "message": entry_msg},
            "tests": {"pass": tests_pass, "message": tests_msg},
        }

    def self_heal_loop(self, max_iterations: int = 1) -> Dict[str, Any]:
        """
        Eğer kritik bir sözdizimi/başlatıcı hatası çıkarsa Aider ile hızlıca tek döngüde onarır.
        Sonsuz döngüye girmeden temiz ve hızlı teslimat sağlar.
        """
        report = self.run_full_verification()
        if report["all_passed"] or (report["syntax"]["pass"] and report["entrypoint"]["pass"]):
            print(f"✅ [Self-Healing Sandbox] Sözdizimi ve Başlatıcı Denetimi %100 BAŞARILI.")
            return report

        print(f"\n⚠️ [Self-Healing Sandbox] Kritik hata tespit edildi! Hızlı otomatik onarım uygulanıyor...")
        
        error_details = []
        if not report["syntax"]["pass"]:
            error_details.append(f"SÖZDİZİMİ HATASI:\n{report['syntax']['message']}")
        if not report["entrypoint"]["pass"]:
            error_details.append(f"BAŞLATICI HATASI:\n{report['entrypoint']['message']}")

        if error_details:
            aider_tool = AiderExecutionTool(
                default_working_dir=str(self.workspace_dir),
                tier="reviewer",
                agent_role="Self-Healing Debugger",
            )
            combined_error_prompt = (
                "Aşağıdaki kritik sözdizimi ve başlatıcı hatalarını düzelt:\n\n"
                + "\n---\n".join(error_details)
                + "\n\nKATI KURALLAR:\n"
                "- run.py geçerli ve import içeren bir Python çalıştırma kodu olmalıdır.\n"
                "- Tüm sözdizimi ve eksik import hatalarını gider."
            )
            aider_tool._run(
                instruction=combined_error_prompt,
                working_directory=str(self.workspace_dir),
            )

        return self.run_full_verification()
