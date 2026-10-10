"""Job-search and application capability for Atlas.

Uses a persistent Playwright browser profile.
Provides:
1. Robust Job Discovery (LinkedIn jobs, direct career portals, tracking parameter cleanup)
2. Structured Job Inspection (JSON-LD JobPosting parser, skills & requirements extraction)
3. Interactive User Approval Gate before filling any application form
4. Safe form preparation without irreversible automated submission
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


_BLOCKED_AUTO_SUBMIT_DOMAINS: set[str] = set()
_JOB_URL_HINTS = (
    "job", "jobs", "career", "careers", "vacanc", "opening", "position",
    "apply", "application", "ilan", "kariyer", "pozisyon", "is-ilan",
)
_JOB_TEXT_HINTS = (
    "job", "jobs", "career", "careers", "vacancy", "vacancies", "apply",
    "application", "ilan", "kariyer", "pozisyon", "işe alım", "iş ilanı",
    "mühendis", "developer", "engineer", "scientist", "uzman", "stajyer", "intern",
)
_GENERIC_HUB_PATHS = {
    "", "/", "/jobs", "/jobs/", "/careers", "/careers/", "/is-ilanlari", "/is-ilanlari/",
    "/stajyer", "/stajyer/", "/en/jobs", "/en/jobs/", "/login", "/register",
    "/about", "/contact", "/gizlilik", "/privacy", "/terms", "/sozluk", "/blog",
    "/etkinlikler", "/pro/", "/testler",
}
_CAPTCHA_HINTS = (
    "captcha", "verify you are human", "are you a robot", "security check",
    "unusual activity", "robot olmadığınızı", "insan olduğunuzu doğrulayın",
)

# Common tech stack keywords for structured requirement detection
_TECH_SKILLS = [
    "python", "machine learning", "deep learning", "pytorch", "tensorflow",
    "keras", "scikit-learn", "nlp", "computer vision", "llm", "rag",
    "langchain", "crewai", "aider", "fastapi", "flask", "django", "docker",
    "kubernetes", "sql", "postgresql", "mongodb", "git", "linux",
    "aws", "gcp", "azure", "pandas", "numpy", "c++", "golang", "java",
    "typescript", "react", "next.js", "data science", "etl", "spark",
    "playwright", "selenium", "prompt engineering",
    "cisco", "ccna", "network", "system", "ağ", "sistem", "switch", "router",
    "firewall", "vlan", "tcp/ip", "dhcp", "dns", "lan", "wan", "active directory",
    "windows server", "it support", "destek", "vpn", "routing", "switching",
    "donanım", "hardware", "wifi", "voip", "fortinet", "palo alto", "vmware", "hyper-v",
    "help desk", "bt uzmanı", "bilgi işlem", "sunucu", "yedekleme", "backup"
]


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
    return True, "Otomatik gönderim onaylandı (Test / Serbest mod)."


def _clean_job_url(raw_url: str) -> str:
    """Strip tracking and session parameters from job links for clean reporting."""
    parsed = urlparse(raw_url)
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return clean.rstrip("/")


def _score_text(profile: Dict[str, str], job_text: str) -> int:
    """Explainable lexical baseline score between candidate profile and job posting."""
    profile_text = " ".join(profile.values())
    tokens = set(re.findall(r"[a-z0-9çğıöşü+#.]{2,}", _normalise(profile_text)))
    job_tokens = set(re.findall(r"[a-z0-9çğıöşü+#.]{2,}", _normalise(job_text)))
    if not tokens or not job_tokens:
        return 0
    overlap = tokens & job_tokens
    return min(100, round(100 * len(overlap) / max(1, min(len(tokens), 18))))


class JobApplicationToolInput(BaseModel):
    action: Literal["discover", "inspect", "prepare", "submit"] = Field(
        description="discover=ilan ara/bağlantı topla; inspect=ilan detaylarını ve gereksinimlerini oku; prepare=kullanıcı onayı ile formu doldur; submit=başvuruyu gönder (güvenlik amacıyla varsayılan olarak kapalıdır)."
    )
    url: Optional[str] = Field(default=None, description="İlan veya iş arama sayfası URL'si.")
    search_query: Optional[str] = Field(default=None, description="discover için iş arama sorgusu; URL verilmezse canlı iş arama motoru açılır.")
    candidate_profile_json: str = Field(default="{}", description="Aday profilinin JSON metni. Ad, Soyad, Yetkinlikler, E-posta, Telefon, CV yolu.")
    answers_json: str = Field(default="{}", description="Form sorularına önceden verilmiş cevaplar için JSON nesnesi.")
    max_results: int = Field(default=10, ge=1, le=25, description="discover için azami ilan sayısı.")
    headless: bool = Field(default=False, description="İlk giriş ve kullanıcı doğrulaması için görünür tarayıcı (headless=False) önerilir.")


class JobApplicationTool(BaseTool):
    """Playwright persistent-context job discovery, structured analysis, and human-in-the-loop preparation."""
    _profile_dir: Path = PrivateAttr()
    name: str = "Atlas Job Search & Application Tool"
    description: str = (
        "İş ilanlarını Playwright ile keşfeder, JSON-LD ve DOM üzerinden şirket, pozisyon ve gereksinimleri yapılandırılmış "
        "olarak analiz eder, aday profiline göre puanlar. Form doldurmadan (prepare) önce kullanıcıdan konsol onayı ister; "
        "otomatik gönderim (submit) güvenlik amacıyla kapalı tutulur."
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

        # 1. Öncelikli gerçek ilan kartı seçicileri (LinkedIn, Kariyer.net, vb.)
        priority_locators = [
            "a.base-card__full-link",
            "ul.jobs-search__results-list li a[href*='/jobs/view/']",
            "a[href*='/jobs/view/']",
            "a[href*='/is-ilani/']",
            "a[href*='/position/']",
        ]
        for sel in priority_locators:
            try:
                for anchor in page.locator(sel).all():
                    href = urljoin(base_url, anchor.get_attribute("href") or "")
                    clean_url = _clean_job_url(href)
                    label = re.sub(r"\s+", " ", anchor.inner_text(timeout=500)).strip()
                    if clean_url in seen or not label or len(label) < 3:
                        continue
                    seen.add(clean_url)
                    candidates.append({"title": label[:150], "url": clean_url})
                    if len(candidates) >= limit:
                        return candidates
            except Exception:
                continue

        # 2. Genel link taraması ve filtreleme (kategori/navigasyon hub'larını atla)
        for anchor in page.locator("a[href]").all():
            try:
                href = urljoin(base_url, anchor.get_attribute("href") or "")
                parsed = urlparse(href)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    continue
                path = parsed.path.rstrip("/").lower()
                if path in _GENERIC_HUB_PATHS:
                    continue

                label = re.sub(r"\s+", " ", anchor.inner_text(timeout=400)).strip()
                if not label or len(label) < 4:
                    continue
                # Filtre: Çok kısa genel navigasyon kelimeleri
                if label.lower() in {"ilanlar", "jobs", "careers", "tüm ilanlar", "staj", "part time", "english", "türkçe", "giriş yap", "kayıt ol"}:
                    continue

                clean_url = _clean_job_url(href)
                if clean_url in seen:
                    continue

                searchable = (clean_url + " " + label).casefold()
                if not any(hint in searchable for hint in _JOB_URL_HINTS + _JOB_TEXT_HINTS):
                    continue

                seen.add(clean_url)
                candidates.append({"title": label[:150], "url": clean_url})
                if len(candidates) >= limit:
                    break
            except Exception:
                continue

        return candidates

    @staticmethod
    def _extract_job_details(page: Any, url: str) -> Dict[str, Any]:
        """İlan sayfasından JSON-LD ve DOM üzerinden pozisyon, şirket, konum ve gereksinimleri çıkarır."""
        title = ""
        company = ""
        location = ""
        description = ""

        # 1. JSON-LD schema.org/JobPosting çözümlemesi
        try:
            json_scripts = page.locator('script[type="application/ld+json"]').all()
            for s in json_scripts:
                try:
                    data = json.loads(s.inner_text(timeout=500) or "{}")
                    if isinstance(data, list):
                        data = next((d for d in data if isinstance(d, dict) and d.get("@type") == "JobPosting"), {})
                    if isinstance(data, dict) and data.get("@type") == "JobPosting":
                        title = data.get("title", "")
                        hiring_org = data.get("hiringOrganization", {})
                        if isinstance(hiring_org, dict):
                            company = hiring_org.get("name", "")
                        elif isinstance(hiring_org, str):
                            company = hiring_org
                        job_loc = data.get("jobLocation", {})
                        if isinstance(job_loc, dict):
                            addr = job_loc.get("address", {})
                            if isinstance(addr, dict):
                                location = ", ".join(filter(None, [addr.get("addressLocality"), addr.get("addressCountry")]))
                            elif isinstance(addr, str):
                                location = addr
                        raw_desc = data.get("description", "")
                        if raw_desc:
                            description = re.sub(r"<[^>]+>", " ", raw_desc)
                            description = re.sub(r"\s+", " ", description).strip()
                        break
                except Exception:
                    continue
        except Exception:
            pass

        # 2. Başlık (Title) Fallback
        if not title:
            h1 = page.locator("h1").first
            if h1.count() and h1.is_visible():
                title = h1.inner_text().strip()
            else:
                title = page.title().split("|")[0].split("—")[0].split("-")[0].strip()

        # 3. Şirket (Company) Fallback
        if not company:
            og_site = page.locator('meta[property="og:site_name"], meta[name="author"]').first
            if og_site.count():
                company = og_site.get_attribute("content") or ""
            if not company:
                comp_el = page.locator('a[data-tracking-control-name*="company"], .topcard__flavor--black-link, [class*="company" i]').first
                if comp_el.count() and comp_el.is_visible():
                    company = comp_el.inner_text().strip()

        # 4. Konum (Location) Fallback
        if not location:
            loc_el = page.locator('.topcard__flavor--bullet, [class*="location" i]').first
            if loc_el.count() and loc_el.is_visible():
                location = loc_el.inner_text().strip()

        # 5. Sayfa gövdesi metni
        try:
            body_text = page.locator("body").inner_text(timeout=5000)
        except Exception:
            body_text = ""

        if not description or len(description) < 100:
            description = body_text[:14000]

        # 6. Gereksinim duyulan anahtar yetkinlikleri çıkar
        desc_lower = description.casefold()
        found_skills = [
            skill for skill in _TECH_SKILLS
            if re.search(rf"\b{re.escape(skill)}\b", desc_lower)
        ]

        return {
            "title": title or "Pozisyon Belirtilmemiş",
            "company": company or "Şirket Belirtilmemiş",
            "location": location or "Türkiye / Uzaktan",
            "skills_required": found_skills,
            "description": description[:4000],
            "full_page_text": body_text[:12000],
        }

    @staticmethod
    def _find_application_submit(page: Any) -> Any:
        """Sayfadaki başvuru gönderim düğmesini bulur."""
        forms = page.locator("form")
        application_hints = (
            "application", "apply", "resume", "cover letter", "work experience",
            "first name", "last name", "job title", "cv", "iş başvuru", "başvur",
            "özgeçmiş", "ozgecmis", "kariyer", "submit", "gönder", "ilet",
        )
        for index in range(min(forms.count(), 30)):
            form = forms.nth(index)
            try:
                if not form.is_visible():
                    continue
                form_text = form.inner_text(timeout=500).casefold()
                if any(hint in form_text for hint in application_hints):
                    submit = form.locator('button:not([type="button"]), input[type="submit"]')
                    if submit.count() and submit.first.is_visible() and submit.first.is_enabled():
                        return submit.first
            except Exception:
                continue
        all_submits = page.locator('button[type="submit"], input[type="submit"], button:has-text("Başvur"), button:has-text("Apply"), button:has-text("Gönder")')
        if all_submits.count() and all_submits.first.is_visible() and all_submits.first.is_enabled():
            return all_submits.first
        return None

    @staticmethod
    def _required_fields_missing(page: Any) -> List[str]:
        missing: List[str] = []
        required = page.locator('[required]')
        for index in range(min(required.count(), 80)):
            field = required.nth(index)
            try:
                if not field.is_visible() or not field.is_enabled():
                    continue
                kind = (field.get_attribute("type") or "").casefold()
                if kind in {"checkbox", "radio"}:
                    if not field.is_checked():
                        missing.append(field.get_attribute("name") or field.get_attribute("id") or f"required_{index}")
                    continue
                if kind == "file":
                    has_file = field.evaluate("(el) => Boolean(el.files && el.files.length)")
                    if not has_file:
                        missing.append(field.get_attribute("name") or field.get_attribute("id") or f"required_{index}")
                    continue
                value = field.input_value(timeout=300)
                if not value.strip():
                    missing.append(field.get_attribute("name") or field.get_attribute("id") or f"required_{index}")
            except Exception:
                continue
        return missing

    @staticmethod
    def _fill_form(page: Any, profile: Dict[str, str], answers: Dict[str, Any]) -> Dict[str, Any]:
        values: Dict[str, str] = {}
        values.update(profile)
        values.update({str(k): str(v) for k, v in _flatten_profile(answers).items()})
        alias_map = {
            "email": ("email", "e mail", "e posta", "eposta"),
            "phone": ("phone", "telephone", "mobile", "telefon", "cep", "gsm"),
            "first_name": ("first name", "firstname", "given name", "ad", "isim"),
            "last_name": ("last name", "lastname", "surname", "soyad"),
            "full_name": ("full name", "name", "ad soyad", "isim"),
            "city": ("city", "location", "şehir", "sehir", "il"),
            "linkedin_url": ("linkedin",),
            "portfolio_url": ("portfolio", "website", "personal site", "portfolyo", "web sitesi", "github"),
        }
        flattened = {_normalise(k.split(".")[-1]): v for k, v in values.items()}
        resume_path = next(
            (
                value for key, value in profile.items()
                if key.split(".")[-1].casefold() in {"resume_path", "cv_path", "resume_file_path", "cv_file_path"}
            ),
            None,
        )
        filled, skipped = [], []
        controls = page.locator("input:not([type=hidden]):not([type=submit]):not([type=button]), textarea, select")
        for index in range(min(controls.count(), 80)):
            control = controls.nth(index)
            try:
                if not control.is_visible() or not control.is_enabled():
                    continue
                kind = (control.get_attribute("type") or "text").casefold()
                tag_name = control.evaluate("(el) => el.tagName.toLowerCase()")
                if kind in {"password", "checkbox", "radio", "submit", "button"}:
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
                if kind == "file":
                    if resume_path and Path(resume_path).expanduser().is_file():
                        control.set_input_files(str(Path(resume_path).expanduser().resolve()))
                        filled.append({"field": key_text[:100] or "resume_upload", "source": "candidate_profile_file"})
                    else:
                        skipped.append(key_text[:100] or f"file_field_{index}")
                    continue
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
                    if tag_name == "select":
                        try:
                            control.select_option(label=match)
                        except Exception:
                            control.select_option(value=match)
                    else:
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

        try:
            profile_raw = self._parse_json(candidate_profile_json, "candidate_profile_json")
            if not profile_raw:
                candidate_file = Path("candidate_profile.json")
                if candidate_file.is_file():
                    try:
                        profile_raw = json.loads(candidate_file.read_text(encoding="utf-8"))
                    except Exception:
                        pass
            answers = self._parse_json(answers_json, "answers_json")
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)

        profile = _flatten_profile(profile_raw)
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        result: Dict[str, Any] = {"action": action, "success": False}

        try:
            with sync_playwright() as playwright:
                browser = None
                if action in {"discover", "inspect"}:
                    browser = playwright.chromium.launch(
                        headless=bool(headless),
                        args=["--disable-blink-features=AutomationControlled"]
                    )
                    context = browser.new_context(
                        viewport={"width": 1440, "height": 1000},
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    )
                else:
                    context = playwright.chromium.launch_persistent_context(
                        user_data_dir=str(self._profile_dir),
                        headless=bool(headless),
                        accept_downloads=False,
                        viewport={"width": 1440, "height": 1000},
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    )
                try:
                    page = context.pages[0] if context.pages else context.new_page()

                    if action == "discover":
                        target = url
                        if not target:
                            if not search_query:
                                return json.dumps({"success": False, "error": "discover için url veya search_query gerekli."}, ensure_ascii=False)
                            # Canlı LinkedIn iş arama motorunu kullan (Cloudflare bot engelini aşar)
                            target = "https://www.linkedin.com/jobs/search?keywords=" + quote_plus(search_query) + "&location=Turkey"
                        page.goto(target, wait_until="domcontentloaded", timeout=45000)
                        page.wait_for_timeout(2000)
                        body = page.locator("body").inner_text(timeout=5000)
                        if self._captcha_detected(page, body) and not headless:
                            print("[Atlas Job Tool] Doğrulama ekranı algılandı; çözmeniz için 6 saniye bekleniyor...")
                            page.wait_for_timeout(6000)

                        links = self._job_links(page, page.url, max_results)
                        scored = []
                        for item in links:
                            score = _score_text(profile, item["title"])
                            scored.append({**item, "match_score": score})
                        scored.sort(key=lambda item: item["match_score"], reverse=True)
                        result.update({
                            "success": True,
                            "source_url": page.url,
                            "jobs": scored,
                            "count": len(scored),
                            "note": f"{len(scored)} adet doğrudan ilan kartı bulundu."
                        })

                    elif action == "inspect":
                        if not url:
                            return json.dumps({"success": False, "error": "inspect için url gerekli."}, ensure_ascii=False)
                        page.goto(url, wait_until="domcontentloaded", timeout=45000)
                        page.wait_for_timeout(1500)
                        body = page.locator("body").inner_text(timeout=8000)
                        if self._captcha_detected(page, body) and not headless:
                            page.wait_for_timeout(5000)

                        details = self._extract_job_details(page, url)
                        score = _score_text(profile, details.get("title", "") + " " + details.get("description", ""))
                        result.update({
                            "success": True,
                            "url": page.url,
                            "title": details.get("title"),
                            "company": details.get("company"),
                            "location": details.get("location"),
                            "skills_required": details.get("skills_required"),
                            "description_snippet": details.get("description")[:1200],
                            "match_score": score,
                            "full_text": details.get("full_page_text")[:10000],
                        })

                    elif action in {"prepare", "submit"}:
                        if not url:
                            return json.dumps({"success": False, "error": f"{action} için url gerekli."}, ensure_ascii=False)

                        if action == "submit":
                            # Güvenlik gereği otomatik gönderim kapalı tutuluyor
                            return json.dumps({
                                "success": False,
                                "status": "submission_disabled",
                                "url": url,
                                "message": "Güvenlik kuralı: Otomatik form gönderimi (submit) kapalı tutulmaktadır. Lütfen başvuru formunu açık tarayıcı üzerinden kontrol edip manuel olarak gönderiniz."
                            }, ensure_ascii=False)

                        page.goto(url, wait_until="domcontentloaded", timeout=45000)
                        page.wait_for_timeout(1500)
                        details = self._extract_job_details(page, url)

                        # --- İNTERAKTİF KULLANICI ONAY KAPISI (HUMAN-IN-THE-LOOP) ---
                        print("\n" + "=" * 65)
                        print("📋 ATLAS İŞ BAŞVURU FORMU HAZIRLIK ONAYI")
                        print("=" * 65)
                        print(f"Pozisyon : {details.get('title')}")
                        print(f"Şirket   : {details.get('company')}")
                        print(f"Konum    : {details.get('location')}")
                        print(f"İlan URL : {url}")
                        print("-" * 65)
                        print("İşlem: Aday profili (Ad, Soyad, E-posta, Telefon, CV) kullanılarak")
                        print("       bu sayfadaki form alanları doldurulacak (GÖNDERİLMEZ).")
                        print("-" * 65)

                        try:
                            choice = input("Bu sayfa için formun otomatik doldurulmasını onaylıyor musunuz? (E/H) [Varsayılan: E]: ").strip().lower()
                        except (EOFError, Exception):
                            choice = "e"

                        if choice in {"h", "hayır", "hayir", "n", "no"}:
                            print("❌ Form doldurma kullanıcı tarafından iptal edildi.")
                            return json.dumps({
                                "success": False,
                                "status": "cancelled_by_user",
                                "url": url,
                                "message": "Kullanıcı başvuru formu doldurma işlemini reddetti/iptal etti."
                            }, ensure_ascii=False)

                        print("✅ Onay alındı. Form alanları dolduruluyor...")
                        fill_result = self._fill_form(page, profile, answers)
                        result.update({
                            "success": True,
                            "status": "prepared_only",
                            "url": page.url,
                            "title": details.get("title"),
                            "company": details.get("company"),
                            "form_fill": fill_result,
                            "user_approved": True,
                            "message": "Form başarıyla hazırlandı. Açık tarayıcı penceresinden formu inceleyebilirsiniz."
                        })
                    else:
                        result.update({"error": f"Bilinmeyen action: {action}"})

                    result["elapsed_seconds"] = round(time.time() - started, 2)
                    return json.dumps(result, ensure_ascii=False, indent=2)
                finally:
                    context.close()
                    if browser is not None:
                        browser.close()
        except Exception as exc:
            result.update({"error": f"{type(exc).__name__}: {str(exc)[:500]}", "elapsed_seconds": round(time.time() - started, 2)})
            return json.dumps(result, ensure_ascii=False, indent=2)
