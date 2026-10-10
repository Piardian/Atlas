"""Job-search and application capability for Atlas.

Uses a persistent Playwright browser profile. Automated submission is disabled by
default and requires an explicitly configured allowlist. LinkedIn automation is
intentionally blocked in this skill; use an authorized integration or manual flow.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import quote_plus, urljoin, urlparse

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


_BLOCKED_AUTO_SUBMIT_DOMAINS = {"linkedin.com", "www.linkedin.com"}
_JOB_URL_HINTS = (
    "job", "jobs", "career", "careers", "vacanc", "opening", "position",
    "apply", "application", "ilan", "kariyer", "pozisyon", "is-ilan",
)
_JOB_TEXT_HINTS = (
    "job", "jobs", "career", "careers", "vacancy", "vacancies", "apply",
    "application", "ilan", "kariyer", "pozisyon", "işe alım", "iş ilanı",
)
_CAPTCHA_HINTS = (
    "captcha", "verify you are human", "are you a robot", "security check",
    "unusual activity", "robot olmadığınızı", "insan olduğunuzu doğrulayın",
)


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9çğıöşü]+", " ", value.casefold()).strip()


def _flatten_profile(value: Any, prefix: str = "") -> Dict[str, str]:
    result: Dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_profile(item, path))
    elif isinstance(value, list):
        result[prefix] = ", ".join(str(item) for item in value if item is not None)
    elif value is not None and str(value).strip():
        result[prefix] = str(value).strip()
    return result


def _domain_matches(host: str, allowed_domain: str) -> bool:
    host = host.casefold().rstrip(".")
    allowed_domain = allowed_domain.casefold().strip().lstrip(".").rstrip(".")
    return bool(allowed_domain) and (host == allowed_domain or host.endswith("." + allowed_domain))


def _is_linkedin_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").casefold()
    return _domain_matches(host, "linkedin.com")


def _auto_submit_allowed(url: str) -> tuple[bool, str]:
    host = (urlparse(url).hostname or "").casefold()
    if not host:
        return False, "Geçerli bir URL/host bulunamadı."
    if any(_domain_matches(host, blocked) for blocked in _BLOCKED_AUTO_SUBMIT_DOMAINS):
        return False, "LinkedIn üzerinde otomatik gönderim bu araçta kapalı."
    if os.getenv("ATLAS_JOB_AUTO_SUBMIT", "").casefold() not in {"1", "true", "yes"}:
        return False, "Otomatik gönderim kapalı. ATLAS_JOB_AUTO_SUBMIT=true ayarlanmalı."
    allowed = [
        item.strip() for item in os.getenv("ATLAS_JOB_ALLOWED_DOMAINS", "").split(",")
        if item.strip()
    ]
    if not any(_domain_matches(host, domain) for domain in allowed):
        return False, f"{host} ATLAS_JOB_ALLOWED_DOMAINS izin listesinde değil."
    return True, "Gönderim için yerel izin kontrolleri geçti; site şartları ve insan doğrulaması ayrıca gözetilir."


def _score_text(profile: Dict[str, str], job_text: str) -> int:
    """Simple explainable lexical baseline; not a hiring prediction."""
    profile_text = " ".join(profile.values())
    tokens = set(re.findall(r"[a-z0-9çğıöşü+#.]{2,}", _normalise(profile_text)))
    job_tokens = set(re.findall(r"[a-z0-9çğıöşü+#.]{2,}", _normalise(job_text)))
    if not tokens or not job_tokens:
        return 0
    overlap = tokens & job_tokens
    # Normalize against candidate terms to avoid rewarding long generic job pages.
    return min(100, round(100 * len(overlap) / max(1, min(len(tokens), 18))))


class JobApplicationToolInput(BaseModel):
    action: Literal["discover", "inspect", "prepare", "submit"] = Field(
        description="discover=ilan ara; inspect=ilan detaylarını oku; prepare=formu doldur ama gönderme; submit=izin kontrolleri geçerse gönder."
    )
    url: Optional[str] = Field(default=None, description="İlan veya iş arama sayfası URL'si.")
    search_query: Optional[str] = Field(default=None, description="discover için iş arama sorgusu; URL verilmezse web araması açılır.")
    candidate_profile_json: str = Field(default="{}", description="Aday profilinin JSON metni. Gerçek ve doğru bilgiler kullan.")
    answers_json: str = Field(default="{}", description="Form sorularına önceden verilmiş cevaplar için JSON nesnesi.")
    max_results: int = Field(default=10, ge=1, le=25, description="discover için azami ilan sayısı.")
    headless: bool = Field(default=False, description="İlk giriş ve kullanıcı doğrulaması için görünür tarayıcı önerilir.")


class JobApplicationTool(BaseTool):
    """Playwright persistent-context job discovery and application preparation."""
    _profile_dir: Path = PrivateAttr()
    name: str = "Atlas Job Search & Application Tool"
    description: str = (
        "İş ilanlarını Playwright ile keşfeder, ilan detaylarını okur, aday profiline göre temel eşleşme puanı verir "
        "ve desteklenen formları hazırlar. Kalıcı tarayıcı profili kullanır. Gönderim varsayılan olarak kapalıdır; "
        "yalnızca izin listesine alınmış desteklenen platformlarda etkinleştirilebilir. CAPTCHA/MFA atlatmaz."
    )
    args_schema: type[BaseModel] = JobApplicationToolInput

    def __init__(self, profile_dir: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        default_dir = Path.home() / ".atlas" / "browser_profiles" / "jobs"
        self._profile_dir = Path(profile_dir or os.getenv("ATLAS_JOB_BROWSER_PROFILE") or str(default_dir)).expanduser()

    @staticmethod
    def _parse_json(raw: str, label: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} geçerli JSON değil: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} bir JSON nesnesi olmalı.")
        return parsed

    @staticmethod
    def _captcha_detected(page: Any, body_text: str) -> bool:
        text = body_text.casefold()
        if any(hint in text for hint in _CAPTCHA_HINTS):
            return True
        try:
            return page.locator(
                'iframe[src*="captcha"], iframe[title*="challenge" i], '
                '[class*="captcha" i], [id*="captcha" i]'
            ).count() > 0
        except Exception:
            return False

    @staticmethod
    def _job_links(page: Any, base_url: str, limit: int) -> List[Dict[str, str]]:
        candidates: List[Dict[str, str]] = []
        seen = set()
        for anchor in page.locator("a[href]").all():
            try:
                href = urljoin(base_url, anchor.get_attribute("href") or "")
                label = re.sub(r"\s+", " ", anchor.inner_text(timeout=700)).strip()
                parsed = urlparse(href)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    continue
                if href in seen or not label or len(label) < 3:
                    continue
                if _domain_matches((parsed.hostname or ""), "linkedin.com"):
                    continue
                searchable = (href + " " + label).casefold()
                if not any(hint in searchable for hint in _JOB_URL_HINTS + _JOB_TEXT_HINTS):
                    continue
                seen.add(href)
                candidates.append({"title": label[:220], "url": href})
                if len(candidates) >= limit:
                    break
            except Exception:
                continue
        return candidates

    @staticmethod
    def _fill_form(page: Any, profile: Dict[str, str], answers: Dict[str, Any]) -> Dict[str, Any]:
        values: Dict[str, str] = {}
        values.update(profile)
        values.update({str(k): str(v) for k, v in _flatten_profile(answers).items()})
        alias_map = {
            "email": ("email", "e mail", "e posta"),
            "phone": ("phone", "telephone", "mobile", "telefon", "cep"),
            "first_name": ("first name", "firstname", "given name", "ad"),
            "last_name": ("last name", "lastname", "surname", "soyad"),
            "full_name": ("full name", "name", "ad soyad", "isim"),
            "city": ("city", "location", "şehir", "sehir", "il"),
            "linkedin_url": ("linkedin",),
            "portfolio_url": ("portfolio", "website", "personal site", "portfolyo", "web sitesi"),
        }
        flattened = {_normalise(k.split(".")[-1]): v for k, v in values.items()}
        filled, skipped = [], []
        controls = page.locator("input:not([type=hidden]):not([type=submit]):not([type=button]), textarea")
        for index in range(min(controls.count(), 80)):
            control = controls.nth(index)
            try:
                if not control.is_visible() or not control.is_enabled():
                    continue
                kind = (control.get_attribute("type") or "text").casefold()
                if kind in {"password", "file", "checkbox", "radio", "submit", "button"}:
                    continue
                key_text = " ".join(filter(None, [
                    control.get_attribute("name"), control.get_attribute("id"),
                    control.get_attribute("placeholder"), control.get_attribute("aria-label"),
                ]))
                try:
                    label_text = control.locator("xpath=ancestor::label[1]").inner_text(timeout=300)
                    key_text += " " + label_text
                except Exception:
                    pass
                key = _normalise(key_text)
                match = None
                for profile_key, value in flattened.items():
                    if profile_key and key and (profile_key in key or key in profile_key):
                        match = value
                        break
                if match is None:
                    for canonical, aliases in alias_map.items():
                        if any(alias in key for alias in aliases):
                            match = flattened.get(canonical)
                            if match:
                                break
                if match:
                    control.fill(match)
                    filled.append({"field": key_text[:100], "source": "candidate_profile_or_answers"})
                else:
                    skipped.append(key_text[:100] or f"field_{index}")
            except Exception:
                continue
        return {"filled": filled, "unmatched_fields": skipped}

    def _run(
        self,
        action: str,
        url: Optional[str] = None,
        search_query: Optional[str] = None,
        candidate_profile_json: str = "{}",
        answers_json: str = "{}",
        max_results: int = 10,
        headless: bool = False,
        **kwargs: Any,
    ) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return "Playwright yüklü değil. Kurulum: pip install playwright && playwright install chromium"

        if url and _is_linkedin_url(url):
            return json.dumps({
                "success": False,
                "status": "platform_automation_disabled",
                "url": url,
                "message": "LinkedIn üzerinde Playwright ile otomatik gezinme/veri çıkarma/başvuru bu skill tarafından yapılmıyor. Yetkili bir entegrasyon kullanın veya LinkedIn işlemini manuel yürütün."
            }, ensure_ascii=False)

        try:
            profile_raw = self._parse_json(candidate_profile_json, "candidate_profile_json")
            answers = self._parse_json(answers_json, "answers_json")
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)

        profile = _flatten_profile(profile_raw)
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        result: Dict[str, Any] = {"action": action, "success": False}
        try:
            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self._profile_dir),
                    headless=bool(headless),
                    accept_downloads=False,
                    viewport={"width": 1440, "height": 1000},
                )
                try:
                    page = context.pages[0] if context.pages else context.new_page()
                    if action == "discover":
                        target = url
                        if not target:
                            if not search_query:
                                return json.dumps({"success": False, "error": "discover için url veya search_query gerekli."}, ensure_ascii=False)
                            target = "https://html.duckduckgo.com/html/?q=" + quote_plus(search_query + " jobs careers")
                        page.goto(target, wait_until="domcontentloaded", timeout=45000)
                        if _is_linkedin_url(page.url):
                            return json.dumps({"success": False, "status": "platform_automation_disabled", "url": page.url}, ensure_ascii=False)
                        page.wait_for_timeout(900)
                        body = page.locator("body").inner_text(timeout=5000)
                        if self._captcha_detected(page, body):
                            return json.dumps({"success": False, "status": "human_verification_required", "url": page.url, "message": "Doğrulama ekranı algılandı; otomasyon durduruldu."}, ensure_ascii=False)
                        links = self._job_links(page, page.url, max_results)
                        scored = []
                        for item in links:
                            score = _score_text(profile, item["title"])
                            scored.append({**item, "match_score": score})
                        scored.sort(key=lambda item: item["match_score"], reverse=True)
                        result.update({"success": True, "source_url": page.url, "jobs": scored, "count": len(scored), "note": "Bu ilk sürüm bağlantı ve başlık temelli eşleşme yapar; ilan açıklamasını inspect ile doğrulayın."})
                    elif action == "inspect":
                        if not url:
                            return json.dumps({"success": False, "error": "inspect için url gerekli."}, ensure_ascii=False)
                        page.goto(url, wait_until="domcontentloaded", timeout=45000)
                        if _is_linkedin_url(page.url):
                            return json.dumps({"success": False, "status": "platform_automation_disabled", "url": page.url}, ensure_ascii=False)
                        page.wait_for_timeout(500)
                        body = page.locator("body").inner_text(timeout=8000)
                        if self._captcha_detected(page, body):
                            return json.dumps({"success": False, "status": "human_verification_required", "url": page.url}, ensure_ascii=False)
                        result.update({"success": True, "url": page.url, "title": page.title(), "page_text": body[:14000], "match_score": _score_text(profile, body[:14000])})
                    elif action in {"prepare", "submit"}:
                        if not url:
                            return json.dumps({"success": False, "error": f"{action} için url gerekli."}, ensure_ascii=False)
                        allowed, reason = _auto_submit_allowed(url)
                        if action == "submit" and not allowed:
                            return json.dumps({"success": False, "status": "submission_blocked", "url": url, "reason": reason}, ensure_ascii=False)
                        page.goto(url, wait_until="domcontentloaded", timeout=45000)
                        if _is_linkedin_url(page.url):
                            return json.dumps({"success": False, "status": "platform_automation_disabled", "url": page.url}, ensure_ascii=False)
                        page.wait_for_timeout(500)
                        body = page.locator("body").inner_text(timeout=8000)
                        if self._captcha_detected(page, body):
                            return json.dumps({"success": False, "status": "human_verification_required", "url": page.url, "message": "CAPTCHA/insan doğrulaması algılandı; atlatılmadı."}, ensure_ascii=False)
                        form_context = (page.title() + " " + body[:7000]).casefold()
                        if not any(term in form_context for term in (
                            "apply", "application", "career", "careers", "job", "position",
                            "resume", "curriculum vitae", "iş başvuru", "iş ilan", "kariyer", "pozisyon", "cv",
                        )):
                            return json.dumps({
                                "success": False,
                                "status": "application_form_not_detected",
                                "url": page.url,
                                "message": "Sayfa iş başvurusu formu olarak doğrulanamadı; hiçbir alan doldurulmadı."
                            }, ensure_ascii=False)
                        fill_result = self._fill_form(page, profile, answers)
                        result.update({"success": True, "status": "prepared_only", "url": page.url, "title": page.title(), "form_fill": fill_result})
                        if action == "submit":
                            # Re-check before the irreversible action and avoid generic consent checkboxes.
                            allowed, reason = _auto_submit_allowed(page.url)
                            if not allowed:
                                result.update({"success": False, "status": "submission_blocked", "reason": reason})
                            else:
                                submit = page.locator('button[type="submit"], input[type="submit"]')
                                if submit.count() == 0:
                                    result.update({"success": False, "status": "submit_control_not_found", "message": "Gönderim düğmesi bulunamadı; form gönderilmedi."})
                                else:
                                    submit.first.click(timeout=5000)
                                    page.wait_for_timeout(900)
                                    confirmation = page.locator("body").inner_text(timeout=5000)
                                    result.update({"success": True, "status": "submission_attempted", "confirmation_excerpt": confirmation[:1800], "message": "Gönderim denendi; başarı ancak sayfadaki doğrulanabilir teyitle kesinleştirilmelidir."})
                    else:
                        result.update({"error": f"Bilinmeyen action: {action}"})
                    result["elapsed_seconds"] = round(time.time() - started, 2)
                    return json.dumps(result, ensure_ascii=False, indent=2)
                finally:
                    context.close()
        except Exception as exc:
            result.update({"error": f"{type(exc).__name__}: {str(exc)[:500]}", "elapsed_seconds": round(time.time() - started, 2)})
            return json.dumps(result, ensure_ascii=False, indent=2)
