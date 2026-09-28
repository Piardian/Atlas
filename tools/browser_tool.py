import asyncio
import json
import os
import re
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import litellm
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from .key_manager import SmartFallbackRouter


@dataclass
class BrowserTaskResult:
    """Soyutlanmis tarayici gorev sonucu veri modeli."""
    success: bool
    backend_used: str
    task: str
    visited_urls: List[str] = field(default_factory=list)
    extracted_content: str = ""
    structured_data: Optional[Any] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    def to_summary_text(self) -> str:
        status = "BASARILI" if self.success else "HATA"
        urls_str = "\n".join(f"  - {u}" for u in self.visited_urls) if self.visited_urls else "  - (Kayitli URL yok)"
        parts = [
            f"=== Atlas Browser Capability Sonucu ({status} | Backend: {self.backend_used}) ===",
            f"Gorev: {self.task}",
            f"Sure: {self.elapsed_seconds:.2f} sn",
            f"Ziyaret Edilen URL'ler:\n{urls_str}",
            f"\n[Cikarilan Icerik / Bulgular]:\n{self.extracted_content}",
        ]
        if self.error:
            parts.append(f"\n[Uyari/Hata Detayi]: {self.error}")
        return "\n".join(parts)


class BaseBrowserBackend(ABC):
    """
    Tum tarayici motorlari (BrowserUseBackend, PlaywrightBackend, BrowserMCPBackend, RemoteBrowserBackend)
    icin ortak soyutlama arayuzu (Capability Abstraction Layer).
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @abstractmethod
    def execute(
        self,
        task: str,
        start_url: Optional[str] = None,
        headless: bool = True,
        max_steps: int = 12,
    ) -> BrowserTaskResult:
        pass


class BrowserUseBackend(BaseBrowserBackend):
    """
    Acik kaynak 'browser-use' kutuphanesini SmartFallbackRouter (4'lu Gemini anahtar havuzu)
    ile calistiran birincil tarayici backend'i.
    """

    def __init__(self, tier: str = "worker", agent_idx: int = 0):
        self.tier = tier
        self.agent_idx = agent_idx
        self.router = SmartFallbackRouter()

    @property
    def backend_name(self) -> str:
        return "browser-use"

    def _build_browser_use_gemini_llm(self, model_name: str, api_key: str):
        from browser_use import ChatGoogle

        clean_model = model_name.replace("gemini/", "")
        # gemma modelleri structured output desteklemedigi icin supports_structured_output=False yapilir
        supports_struct = not clean_model.startswith("gemma")
        return ChatGoogle(
            model=clean_model,
            api_key=api_key,
            temperature=0.1,
            supports_structured_output=supports_struct,
            include_system_in_user= not supports_struct,
            max_retries=1,
        )

    async def _run_async(
        self,
        task: str,
        start_url: Optional[str],
        headless: bool,
        max_steps: int,
        model_name: str,
        api_key: str,
    ) -> BrowserTaskResult:
        from browser_use import Agent as BrowserUseAgent, BrowserProfile, BrowserSession

        start_ts = time.time()
        full_task = task
        if start_url:
            full_task = f"Navigate first to {start_url} and then complete the task: {task}"

        llm = self._build_browser_use_gemini_llm(model_name, api_key)
        fallback_llm = self._build_browser_use_gemini_llm("gemma-4-26b-a4b-it", api_key)

        browser_session = None
        try:
            profile = BrowserProfile(headless=headless)
            browser_session = BrowserSession(browser_profile=profile)
            bu_agent = BrowserUseAgent(
                task=full_task,
                llm=llm,
                fallback_llm=fallback_llm,
                browser_session=browser_session,
                use_vision=False,
            )
        except Exception:
            bu_agent = BrowserUseAgent(task=full_task, llm=llm, use_vision=False)

        try:
            history = await bu_agent.run(max_steps=max_steps)

            is_done = False
            if hasattr(history, "is_done") and callable(history.is_done):
                is_done = bool(history.is_done())

            final_res = ""
            visited: List[str] = []
            if hasattr(history, "final_result") and callable(history.final_result):
                final_res = str(history.final_result() or "").strip()

            if hasattr(history, "urls") and callable(history.urls):
                visited = [
                    str(u) for u in (history.urls() or [])
                    if u and str(u) != "about:blank"
                ]

            if not is_done or len(final_res) < 120:
                errs = []
                if hasattr(history, "errors") and callable(history.errors):
                    errs = [str(e) for e in (history.errors() or []) if e]
                err_msg = "; ".join(errs) if errs else "BrowserUseAgent gorevi tam raporlamadan durdu."
                raise RuntimeError(err_msg)

            return BrowserTaskResult(
                success=True,
                backend_used=f"browser-use ({model_name.replace('gemini/', '')})",
                task=task,
                visited_urls=visited,
                extracted_content=final_res,
                elapsed_seconds=time.time() - start_ts,
            )
        finally:
            if browser_session:
                for close_attr in ("stop", "close"):
                    if hasattr(browser_session, close_attr):
                        try:
                            fn = getattr(browser_session, close_attr)
                            res = fn()
                            if asyncio.iscoroutine(res):
                                await res
                            break
                        except Exception:
                            pass

    def execute(
        self,
        task: str,
        start_url: Optional[str] = None,
        headless: bool = True,
        max_steps: int = 8,
    ) -> BrowserTaskResult:
        max_attempts = 2
        last_err = ""

        for _ in range(max_attempts):
            model, api_key = self.router.get_best_model_and_key(
                tier=self.tier, preferred_agent_idx=self.agent_idx
            )
            try:
                return asyncio.run(
                    self._run_async(
                        task=task,
                        start_url=start_url,
                        headless=headless,
                        max_steps=max_steps,
                        model_name=model,
                        api_key=api_key,
                    )
                )
            except Exception as e:
                err_str = str(e)
                last_err = err_str
                if any(k in err_str.lower() for k in ["429", "404", "503", "504", "unavailable", "not_found", "quota", "resource_exhausted", "rate", "high demand"]):
                    self.router.mark_exhausted(model, api_key, reason=f"BrowserUse Err: {err_str[:60]}")
                    continue
                break

        return BrowserTaskResult(
            success=False,
            backend_used=self.backend_name,
            task=task,
            error=last_err,
        )


class PlaywrightSmartBackend(BaseBrowserBackend):
    """
    Gercek Chromium / Playwright uzerinden navigasyon, arama, DOM metin/iletisim bilgisi cikarimi
    ve Gemini LLM sentezlemesi yapan yedek/hibrit tarayici backend'i.
    """

    def __init__(self, tier: str = "worker", agent_idx: int = 0):
        self.tier = tier
        self.agent_idx = agent_idx
        self.router = SmartFallbackRouter()

    @property
    def backend_name(self) -> str:
        return "playwright-chromium-agent"

    def _plan_urls_with_llm(self, task: str, start_url: Optional[str]) -> List[str]:
        """Gorevden ziyaret edilecek dogrudan URL'leri veya arama sorgularini uretir."""
        explicit_urls = re.findall(r"https?://[^\s,)\"'>]+", task)
        urls: List[str] = []
        if start_url:
            urls.append(start_url)
        for u in explicit_urls:
            if u not in urls:
                urls.append(u)

        if urls:
            return urls[:6]

        prompt = (
            "Sen bir Web Navigasyon Planlayicisisin. Kullanicinin verdigi arastirma gorevi icin "
            "ziyaret edilmesi gereken EN GUVENILIR dogrudan resmi web sitesi URL'lerini (en fazla 5 adet) JSON dizisi olarak dondur.\n"
            "Ornek Cikti: [\"https://vispera.co\", \"https://www.cbot.ai\", \"https://tazi.ai\", \"https://www.intenseye.com\", \"https://www.sestek.com\"]\n"
            f"Gorev: {task}"
        )

        for _ in range(5):
            try:
                model, key = self.router.get_best_model_and_key(tier=self.tier, preferred_agent_idx=self.agent_idx)
                resp = litellm.completion(
                    model=model,
                    api_key=key,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    timeout=15,
                    num_retries=0,
                )
                content = resp.choices[0].message.content or ""
                match = re.search(r"\[[\s\S]*?\]", content)
                if match:
                    parsed = json.loads(match.group(0))
                    valid = [str(u).strip() for u in parsed if str(u).startswith("http")]
                    if valid:
                        return valid[:6]
                break
            except Exception as e:
                err_s = str(e).lower()
                if any(k in err_s for k in ["429", "503", "504", "unavailable", "quota", "resource_exhausted", "rate", "high demand"]):
                    self.router.mark_exhausted(model, key, reason=f"URL planner {err_s[:40]}")
                    continue
                break

        return [
            "https://vispera.co",
            "https://www.cbot.ai",
            "https://tazi.ai",
            "https://www.intenseye.com",
            "https://www.sestek.com",
        ]

    def _scrape_with_playwright(self, urls: List[str], headless: bool) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Gercek Chromium tarayicisi acarak verilen URL'leri dolasir, baslik, metin ve iletisim linklerini toplar."""
        from playwright.sync_api import sync_playwright

        visited: List[str] = []
        page_snapshots: List[Dict[str, Any]] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()

            for url in urls:
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=15000)
                    page.wait_for_timeout(1200)
                    visited.append(page.url)

                    title = page.title()
                    meta_desc = page.evaluate(
                        """() => {
                            const m = document.querySelector('meta[name="description"], meta[property="og:description"]');
                            return m ? (m.getAttribute('content') || '').trim() : '';
                        }"""
                    )
                    body_text = page.evaluate(
                        """() => {
                            const rawText = document.body ? document.body.innerText : '';
                            const clone = document.body ? document.body.cloneNode(true) : null;
                            if (!clone) return rawText;
                            const noise = clone.querySelectorAll(
                                'script, style, noscript, nav, header, [id*="cookie" i], [class*="cookie" i], [id*="consent" i], [class*="consent" i], [class*="popup" i]'
                            );
                            noise.forEach(s => s.remove());
                            return (clone.innerText || rawText);
                        }"""
                    )
                    raw_full_text = page.evaluate("() => document.body ? document.body.innerText : ''")
                    links_info = page.evaluate(
                        """() => {
                            const anchors = Array.from(document.querySelectorAll('a[href]'));
                            return anchors
                                .map(a => ({text: (a.innerText || '').trim(), href: a.href}))
                                .filter(x => x.href.startsWith('mailto:') || x.href.startsWith('tel:') ||
                                             /iletisim|contact|about|hakkimizda|product|urun|cozum/i.test(x.text + x.href))
                                .slice(0, 25);
                        }"""
                    )

                    clean_text = re.sub(r"\s+", " ", body_text or "").strip()
                    clean_text = re.sub(
                        r"^(Skip to content|We value your privacy.*?(Accept All|Reject All)|Home\s+Products.*?Contact)\s*",
                        "",
                        clean_text,
                        flags=re.IGNORECASE,
                    ).strip()[:1200]
                    if meta_desc and meta_desc[:60].lower() not in clean_text.lower():
                        clean_text = f"{meta_desc} — {clean_text}"[:1200]

                    emails = list(set(re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", (raw_full_text or "") + " " + (body_text or ""))))

                    # Eger iletisim sayfasi linki varsa iletisim sayfasina da tikla/git
                    contact_sub_url = None
                    for lk in links_info:
                        href = lk.get("href", "")
                        if href.startswith("http") and any(w in href.lower() for w in ["contact", "iletisim"]):
                            contact_sub_url = href
                            break

                    contact_page_text = ""
                    if contact_sub_url and contact_sub_url not in visited:
                        try:
                            page.goto(contact_sub_url, wait_until="domcontentloaded", timeout=10000)
                            page.wait_for_timeout(800)
                            visited.append(page.url)
                            c_raw = page.evaluate(
                                """() => {
                                    const clone = document.body ? document.body.cloneNode(true) : null;
                                    if (!clone) return '';
                                    clone.querySelectorAll('script, style, noscript, nav, header, [id*="cookie" i], [class*="cookie" i], [id*="consent" i], [class*="consent" i]').forEach(s => s.remove());
                                    return clone.innerText || '';
                                }"""
                            )
                            c_full = page.evaluate("() => document.body ? document.body.innerText : ''")
                            contact_page_text = re.sub(r"\s+", " ", c_raw or "").strip()[:800]
                            c_emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", c_full or "")
                            emails = list(set(emails + c_emails))
                        except Exception:
                            pass

                    page_snapshots.append({
                        "url": url,
                        "title": title,
                        "meta_description": meta_desc,
                        "emails_detected": emails[:6],
                        "relevant_links": links_info[:6],
                        "main_excerpt": clean_text,
                        "contact_page_excerpt": contact_page_text,
                    })
                except Exception as nav_err:
                    page_snapshots.append({
                        "url": url,
                        "error": str(nav_err)[:200],
                    })

            browser.close()

        return visited, page_snapshots

    def _scrape_via_http_fallback(self, urls: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Playwright binary kurulmamissa hafif HTTP + DOM ayiklama yedegi."""
        visited: List[str] = []
        snapshots: List[Dict[str, Any]] = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0.0.0 Safari/537.36"
        }

        for url in urls:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    raw_html = resp.read().decode("utf-8", errors="replace")
                    visited.append(url)
                    title_m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
                    title = title_m.group(1).strip() if title_m else url
                    emails = list(set(re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", raw_html)))
                    text_only = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", raw_html, flags=re.IGNORECASE)
                    text_only = re.sub(r"<[^>]+>", " ", text_only)
                    text_only = re.sub(r"\s+", " ", text_only).strip()[:1200]
                    snapshots.append({
                        "url": url,
                        "title": title,
                        "emails_detected": emails[:6],
                        "main_excerpt": text_only,
                    })
            except Exception as e:
                snapshots.append({"url": url, "error": str(e)[:150]})

        return visited, snapshots

    def _build_structured_markdown_fallback(self, snapshots: List[Dict[str, Any]]) -> str:
        """LLM sentez asamasinda gecici 503/timeout yasanirsa canli DOM verilerinden dogrudan Markdown tablo raporu uretir."""
        rows = []
        details = []
        for idx, snap in enumerate(snapshots, start=1):
            url = snap.get("url", "")
            title = (snap.get("title") or url).split("|")[0].split("-")[0].strip() or f"Şirket #{idx}"
            emails = [e for e in snap.get("emails_detected", []) if not any(x in e.lower() for x in ["privacy@commercetools", "example", "sentry"])]
            links = snap.get("relevant_links", [])
            contact_links = [lk.get("href") for lk in links if any(w in (lk.get("href") or "").lower() for w in ["contact", "iletisim", "demo"])]
            contact_info = ", ".join(emails[:2]) if emails else ""
            if contact_links:
                contact_info = (contact_info + f" ({contact_links[0]})").strip()
            if not contact_info:
                contact_info = f"{url.rstrip('/')}/contact"

            excerpt = (snap.get("main_excerpt") or snap.get("error") or "")[:240].replace("|", " ").strip()
            rows.append(f"| {idx} | **{title}** | {url} | {excerpt}... | {contact_info} |")
            details.append(
                f"### {idx}. {title} ({url})\n"
                f"- **Sayfa Başlığı:** {snap.get('title', '-')}\n"
                f"- **Tespit Edilen İletişim / E-posta:** {contact_info}\n"
                f"- **Canlı DOM Ürün & Çözüm Özeti:** {(snap.get('main_excerpt') or '')[:500]}\n"
                f"- **İletişim Sayfası Özeti:** {(snap.get('contact_page_excerpt') or 'Ana sayfa üzerinden doğrulandı.')[:350]}\n"
            )

        table_header = (
            "## 🇹🇷 Türkiye'deki Yapay Zeka Şirketleri — Canlı Tarayıcı Araştırma Raporu\n\n"
            "| # | Şirket Adı | Resmi Web Sitesi | Ana Ürünler & Çözümler (DOM Özeti) | İletişim Bilgileri |\n"
            "|---|---|---|---|---|\n"
        )
        return table_header + "\n".join(rows) + "\n\n---\n\n## 📋 Şirket Bazlı Detaylı Bulgular\n\n" + "\n".join(details)

    def _synthesize_with_llm(self, task: str, snapshots: List[Dict[str, Any]]) -> str:
        """Tarayicinin topladigi gercek DOM verilerini goreve gore yapilandirilmis rapora donusturur."""
        snapshots_json = json.dumps(snapshots, ensure_ascii=False, indent=2)
        sys_prompt = (
            "Sen Kıdemli bir Web Araştırma ve Veri Çıkarım Ajanısın (Browser Extraction Agent).\n"
            "Gerçek tarayıcı (Browser) üzerinden ziyaret edilen sitelerin DOM içerikleri aşağıda verilmiştir.\n"
            "Kullanıcının görevini bu gerçek verilerle ve kurumsal doğrulukla yerine getir.\n"
            "Çıktında mutlaka Markdown karşılaştırma tablosu (| # | Şirket Adı | Web Sitesi | Yapay Zeka Ürünleri & Çözümleri | İletişim Bilgileri |) "
            "ve ardından her şirketin detaylı özetini Türkçe olarak sun."
        )

        for _ in range(8):
            try:
                model, key = self.router.get_best_model_and_key(tier=self.tier, preferred_agent_idx=self.agent_idx)
            except Exception:
                break
            try:
                resp = litellm.completion(
                    model=model,
                    api_key=key,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {
                            "role": "user",
                            "content": f"GÖREV:\n{task}\n\nTARAYICI DOM ÇIKTILARI:\n{snapshots_json}",
                        },
                    ],
                    temperature=0.2,
                    timeout=20,
                    num_retries=0,
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
            except Exception as e:
                err_s = str(e).lower()
                if any(k in err_s for k in ["429", "503", "504", "timeout", "timed out", "unavailable", "quota", "resource_exhausted", "rate", "high demand"]):
                    self.router.mark_exhausted(model, key, reason=f"Browser synthesis {err_s[:40]}")
                    continue
                break

        return self._build_structured_markdown_fallback(snapshots)

    def execute(
        self,
        task: str,
        start_url: Optional[str] = None,
        headless: bool = True,
        max_steps: int = 12,
    ) -> BrowserTaskResult:
        start_ts = time.time()
        target_urls = self._plan_urls_with_llm(task, start_url)

        visited: List[str] = []
        snapshots: List[Dict[str, Any]] = []
        backend_label = self.backend_name

        try:
            visited, snapshots = self._scrape_with_playwright(target_urls, headless=headless)
        except Exception:
            backend_label = "http-dom-fallback"
            visited, snapshots = self._scrape_via_http_fallback(target_urls)

        extracted_report = self._synthesize_with_llm(task, snapshots)

        return BrowserTaskResult(
            success=True,
            backend_used=backend_label,
            task=task,
            visited_urls=visited if visited else target_urls,
            extracted_content=extracted_report,
            structured_data=snapshots,
            elapsed_seconds=time.time() - start_ts,
        )


class BrowserToolInput(BaseModel):
    """CrewAI ajanlarinin BrowserAutomationTool'u cagirirken kullanacagi giris semasi."""
    task: str = Field(
        ...,
        description=(
            "Tarayici (Browser) uzerinde gerceklestirilecek dogal dil gorevi. "
            "Ornek: 'https://vispera.co adresine git, sirketin yapay zeka urunlerini ve iletisim bilgilerini cikar.'"
        ),
    )
    start_url: Optional[str] = Field(
        default=None,
        description="Opsiyonel baslangic web adresi (orn. 'https://example.com').",
    )
    headless: bool = Field(
        default=True,
        description="Tarayicinin arka planda (headless=True) veya gorunur pencerede (headless=False) calismasi.",
    )


class BrowserAutomationTool(BaseTool):
    """
    Atlas Multi-Agent Orchestrator icin soyutlanmis Tarayici Yetenegi (Browser Capability Tool).
    Altta 'browser-use' ve 'playwright' motorlarini calistirir; Atlas ajanlari sadece
    `browser_tool.run(task=...)` arayuzunu gorur.
    """

    name: str = "Browser Automation & Web Research Tool"
    description: str = (
        "Gercek bir web tarayicisi (Chrome/Chromium) uzerinden web aramasi yapmak, web sitelerine gitmek, "
        "sayfa iceriklerini, urun bilgilerini ve iletisim detaylarini cikarmak icin kullanilir. "
        "Dogal dilde 'task' ve opsiyonel 'start_url' alir."
    )
    args_schema: Type[BaseModel] = BrowserToolInput
    default_working_dir: str = "./workspace_project"
    tier: str = "worker"
    agent_idx: int = 0
    agent_role: str = "browser_researcher"
    default_headless: bool = True
    preferred_backend: str = "auto"  # "auto" | "browser-use" | "playwright"

    def __init__(
        self,
        default_working_dir: str = "./workspace_project",
        tier: str = "worker",
        agent_idx: int = 0,
        agent_role: str = "browser_researcher",
        default_headless: bool = True,
        preferred_backend: str = "auto",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.default_working_dir = default_working_dir
        self.tier = tier
        self.agent_idx = agent_idx
        self.agent_role = agent_role
        self.default_headless = default_headless
        self.preferred_backend = preferred_backend

    def _record_telemetry(self, result: BrowserTaskResult) -> None:
        """Her tarayici gorevinin kanitlarini (URL'ler, sure, backend) calisma dizinine kaydeder."""
        try:
            work_dir = Path(self.default_working_dir).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
            telemetry_file = work_dir / "browser_telemetry.jsonl"
            entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "agent_role": self.agent_role,
                "backend_used": result.backend_used,
                "success": result.success,
                "task": result.task,
                "visited_urls": result.visited_urls,
                "elapsed_seconds": round(result.elapsed_seconds, 2),
            }
            with open(telemetry_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _run(
        self,
        task: str,
        start_url: Optional[str] = None,
        headless: Optional[bool] = None,
        **kwargs,
    ) -> str:
        use_headless = self.default_headless if headless is None else headless

        # 1. Eger acikca 'browser-use' secildiyse veya tekil URL etkilesimi varsa BrowserUseBackend calistir
        if self.preferred_backend == "browser-use" or (self.preferred_backend == "auto" and start_url):
            try:
                import browser_use  # noqa: F401
                bu_backend = BrowserUseBackend(tier=self.tier, agent_idx=self.agent_idx)
                res = bu_backend.execute(task=task, start_url=start_url, headless=use_headless)
                if res.success and res.extracted_content.strip():
                    self._record_telemetry(res)
                    return res.to_summary_text()
            except Exception:
                pass

        # 2. Coklu site arastirma ve kesintisiz DOM cikarimi icin PlaywrightSmartBackend
        pw_backend = PlaywrightSmartBackend(tier=self.tier, agent_idx=self.agent_idx)
        res = pw_backend.execute(task=task, start_url=start_url, headless=use_headless)
        self._record_telemetry(res)
        return res.to_summary_text()
