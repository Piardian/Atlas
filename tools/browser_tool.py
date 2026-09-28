import asyncio
import json
import os
import re
import time
import urllib.parse
import urllib.request
import webbrowser
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
    ile calistiran BIRINCIL (default) tarayici backend'i.
    """

    def __init__(self, tier: str = "worker", agent_idx: int = 0):
        self.tier = tier
        self.agent_idx = agent_idx
        self.router = SmartFallbackRouter()

    @property
    def backend_name(self) -> str:
        return "browser-use"

    def _create_rotating_browser_use_llm(self):
        """
        browser_use.ChatGoogle sinifini genisleten ve her adimda 503/429 hatasi alinirsa
        SmartFallbackRouter uzerinden aninda siradaki Gemini anahtarina/modeline gecen sarmalayici.
        """
        from browser_use import ChatGoogle

        router = self.router
        tier = self.tier
        base_idx = self.agent_idx

        class AtlasRotatingChatGoogle(ChatGoogle):
            def __init__(self_llm):
                m, k = router.get_best_model_and_key(tier=tier, preferred_agent_idx=base_idx)
                clean_m = m.replace("gemini/", "")
                if clean_m.startswith("gemma"):
                    clean_m = "gemini-3.6-flash"
                super().__init__(
                    model=clean_m,
                    api_key=k,
                    temperature=0.1,
                    supports_structured_output=True,
                    max_retries=1,
                )
                self_llm.last_used_model = clean_m

            async def ainvoke(self_llm, messages, output_format=None, **kwargs):
                last_exc = None
                for attempt in range(8):
                    m, k = router.get_best_model_and_key(
                        tier=tier, preferred_agent_idx=base_idx + attempt
                    )
                    clean_m = m.replace("gemini/", "")
                    supports_struct = not clean_m.startswith("gemma")
                    delegate = ChatGoogle(
                        model=clean_m,
                        api_key=k,
                        temperature=0.1,
                        supports_structured_output=supports_struct,
                        include_system_in_user=not supports_struct,
                        max_retries=1,
                    )
                    try:
                        res = await delegate.ainvoke(messages, output_format=output_format, **kwargs)
                        self_llm.last_used_model = clean_m
                        return res
                    except Exception as e:
                        last_exc = e
                        err_s = str(e).lower()
                        if any(
                            w in err_s
                            for w in [
                                "429", "503", "504", "404", "unavailable", "not_found",
                                "quota", "resource_exhausted", "rate", "high demand", "overloaded",
                            ]
                        ):
                            router.mark_exhausted(m, k, reason=f"BrowserUse Step LLM: {err_s[:50]}")
                            continue
                        raise e
                raise last_exc

        return AtlasRotatingChatGoogle()

    async def _run_async(
        self,
        task: str,
        start_url: Optional[str],
        headless: bool,
        max_steps: int,
    ) -> BrowserTaskResult:
        from browser_use import Agent as BrowserUseAgent, BrowserProfile, BrowserSession

        start_ts = time.time()
        helper = PlaywrightSmartBackend(tier=self.tier, agent_idx=self.agent_idx)
        target_urls = helper._plan_urls_with_llm(task, start_url)
        first_url = start_url or (target_urls[0] if target_urls else "https://vispera.co")
        urls_list_str = ", ".join(target_urls[:5])

        full_task = (
            f"Navigate first to {first_url}. "
            f"Target websites to inspect for this mission: {urls_list_str}. "
            f"Do NOT waste steps creating or editing todo.md files; directly navigate to each website, extract their core AI products/solutions and official contact details (emails, contact URLs), "
            f"and finish by calling 'done' with a comprehensive Turkish Markdown report including a comparison table covering all {len(target_urls)} targets. "
            f"Mission details: {task}"
        )

        rotating_llm = self._create_rotating_browser_use_llm()

        browser_session = None
        try:
            profile = BrowserProfile(headless=headless)
            browser_session = BrowserSession(browser_profile=profile)
            bu_agent = BrowserUseAgent(
                task=full_task,
                llm=rotating_llm,
                page_extraction_llm=rotating_llm,
                fallback_llm=rotating_llm,
                browser_session=browser_session,
                use_vision=False,
                use_judge=False,
                flash_mode=True,
                max_clickable_elements_length=15000,
            )
        except Exception as init_err:
            raise RuntimeError(f"BrowserUseAgent baslatilamadi: {init_err}") from init_err

        try:
            history = await bu_agent.run(max_steps=max_steps)

            final_res = ""
            visited: List[str] = []
            extracted_chunks: List[str] = []

            if hasattr(history, "final_result") and callable(history.final_result):
                final_res = str(history.final_result() or "").strip()

            if hasattr(history, "urls") and callable(history.urls):
                visited = list(dict.fromkeys(
                    str(u) for u in (history.urls() or [])
                    if u and str(u) != "about:blank"
                ))

            if hasattr(history, "extracted_content") and callable(history.extracted_content):
                extracted_chunks = [
                    str(c).strip() for c in (history.extracted_content() or [])
                    if c and str(c).strip()
                ]

            if not visited and not final_res and not extracted_chunks:
                errs = []
                if hasattr(history, "errors") and callable(history.errors):
                    errs = [str(e) for e in (history.errors() or []) if e]
                err_msg = "; ".join(errs) if errs else "BrowserUseAgent hicbir URL ziyaret edemedi."
                raise RuntimeError(err_msg)

            # Eger BrowserUseAgent tum hedef siteleri gezmeden max_steps sinirina ulastiysa veya
            # ciktiyi henuz tam Markdown tablosuna donusturmediyse, kalan sitelerin DOM ozetleriyle
            # BrowserUseAgent bulgularini birlestirerek eksiksiz tablo raporu olustur.
            visited_domains = {urllib.parse.urlparse(u).netloc.replace("www.", "").lower() for u in visited if u.startswith("http")}
            target_domains = {urllib.parse.urlparse(u).netloc.replace("www.", "").lower() for u in target_urls if u.startswith("http")}
            all_targets_visited = len(visited_domains.intersection(target_domains)) >= len(target_domains)
            has_complete_table = ("|" in final_res and "---" in final_res and len(final_res) >= 250 and all_targets_visited)
            structured_snapshots: List[Dict[str, Any]] = []

            if not has_complete_table:
                # BrowserUseAgent'in gezdigi ve kalan hedef sitelerin DOM verilerini zenginlestir
                extra_visited, structured_snapshots = await asyncio.to_thread(
                    helper._scrape_with_playwright, target_urls, True
                )
                for ev in extra_visited:
                    if ev not in visited:
                        visited.append(ev)

                bu_notes = "\n".join(extracted_chunks[-4:] + ([final_res] if final_res else [])).strip()
                if bu_notes and structured_snapshots:
                    structured_snapshots[0]["browser_use_agent_notes"] = bu_notes[:800]

                final_res = await asyncio.to_thread(helper._synthesize_with_llm, task, structured_snapshots)

            # Eger kullanici buldugu siteleri tarayicida acip karsisina getirmesini istediyse
            t_low = task.lower()
            if any(
                phrase in t_low
                for phrase in ["karşıma getir", "karsima getir", "tarayıcıdan aç", "tarayicidan ac", "tarayıcıda aç", "ekranda aç"]
            ):
                urls_to_open: List[str] = []
                for snap in structured_snapshots:
                    for lk in snap.get("relevant_links", []):
                        href = lk.get("href", "")
                        if href.startswith("http") and "/jobs/view/" in href.lower():
                            if href not in urls_to_open:
                                urls_to_open.append(href)
                if not urls_to_open:
                    urls_to_open = [u for u in visited if "github.com" not in u and "contact" not in u][:3]
                for open_u in urls_to_open[:3]:
                    try:
                        webbrowser.open_new_tab(open_u)
                    except Exception:
                        pass

            active_model = getattr(rotating_llm, "last_used_model", "gemini-3.1-flash-lite")
            return BrowserTaskResult(
                success=True,
                backend_used=f"browser-use ({active_model})",
                task=task,
                visited_urls=visited,
                extracted_content=final_res,
                structured_data=structured_snapshots if structured_snapshots else None,
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
        max_steps: int = 12,
    ) -> BrowserTaskResult:
        try:
            return asyncio.run(
                self._run_async(
                    task=task,
                    start_url=start_url,
                    headless=headless,
                    max_steps=max_steps,
                )
            )
        except Exception as e:
            return BrowserTaskResult(
                success=False,
                backend_used=self.backend_name,
                task=task,
                error=str(e),
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

        t_low = task.lower()
        is_job_search = any(
            k in t_low
            for k in ["iş ilan", "is ilan", "staj", "kariyer", "linkedin", "cv", "job", "intern"]
        )

        if urls and not is_job_search:
            return urls[:6]

        prompt = (
            "Sen bir Web Navigasyon Planlayicisisin. Kullanicinin verdigi arastirma gorevi icin "
            "ziyaret edilmesi gereken EN GUVENILIR dogrudan resmi web sitesi URL'lerini (en fazla 5 adet) JSON dizisi olarak dondur.\n"
            "Eger gorev is/staj ilani aramasi ise LinkedIn Public Jobs (https://www.linkedin.com/jobs/search/?keywords=AI%20Engineer%20Python&location=Turkey), "
            "LinkedIn Staj/Junior (https://www.linkedin.com/jobs/search/?keywords=Python%20Developer%20AI&location=Turkey), "
            "Youthall (https://www.youthall.com/tr/jobs/) veya Coderspace (https://coderspace.io/etkinlikler) URL'lerini kullan. "
            "ASLA kariyer.net veya glassdoor.com ekleme (CAPTCHA engeli vardir).\n"
            f"Gorev: {task}"
        )

        blocked_domains = ["kariyer.net", "glassdoor.com", "indeed.com"]
        planned_urls: List[str] = []
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
                    valid = [
                        str(u).strip()
                        for u in parsed
                        if str(u).startswith("http") and not any(bd in str(u).lower() for bd in blocked_domains)
                    ]
                    if valid:
                        planned_urls = valid[:5]
                        break
                break
            except Exception as e:
                err_s = str(e).lower()
                if any(k in err_s for k in ["429", "503", "504", "unavailable", "quota", "resource_exhausted", "rate", "high demand"]):
                    self.router.mark_exhausted(model, key, reason=f"URL planner {err_s[:40]}")
                    continue
                break

        if not planned_urls:
            if is_job_search:
                planned_urls = [
                    "https://github.com/Piardian",
                    "https://www.linkedin.com/jobs/search/?keywords=AI%20Engineer%20Python&location=Turkey",
                    "https://www.linkedin.com/jobs/search/?keywords=Python%20Developer%20LLM&location=Turkey",
                    "https://www.youthall.com/tr/jobs/",
                    "https://coderspace.io/etkinlikler",
                ]
            else:
                planned_urls = [
                    "https://vispera.co",
                    "https://www.cbot.ai",
                    "https://tazi.ai",
                    "https://www.intenseye.com",
                    "https://www.sestek.com",
                ]

        for pu in planned_urls:
            if pu not in urls and not any(bd in pu.lower() for bd in blocked_domains):
                urls.append(pu)

        return urls[:6]

    def _scrape_with_playwright(self, urls: List[str], headless: bool) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Gercek Chromium tarayicisi acarak verilen URL'leri dolasir, baslik, metin, is kartlari ve iletisim linklerini toplar."""
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
                    page.wait_for_timeout(1400)

                    title = page.title()
                    raw_full_text = page.evaluate("() => document.body ? document.body.innerText : ''")

                    # CAPTCHA / Bot koruma sayfalarini tespit et ve rapora kirli veri olarak sokma
                    if any(
                        bot_sig in (title + " " + (raw_full_text or "")[:500]).lower()
                        for bot_sig in ["access to this page has been denied", "px-captcha", "just a moment...", "humans only"]
                    ):
                        continue

                    visited.append(page.url)
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
                                'script, style, noscript, nav, header, [id*="cookie" i], [class*="cookie" i], [id*="consent" i], [class*="consent" i], [class*="popup" i], .jobs-search__filter-list, .filters'
                            );
                            noise.forEach(s => s.remove());
                            return (clone.innerText || rawText);
                        }"""
                    )

                    # Ozel LinkedIn & Is Ilani Kart Cikarimi
                    job_cards = page.evaluate(
                        """() => {
                            const cards = Array.from(document.querySelectorAll('.base-search-card, .job-search-card, li .base-card'));
                            const results = [];
                            for (const c of cards) {
                                const titleEl = c.querySelector('.base-search-card__title, h3');
                                const compEl = c.querySelector('.base-search-card__subtitle, h4');
                                const locEl = c.querySelector('.job-search-card__location');
                                const linkEl = c.querySelector('a.base-card__full-link, a[href*="/jobs/view/"]');
                                if (titleEl && linkEl && linkEl.href) {
                                    const t = (titleEl.innerText || '').trim();
                                    const comp = compEl ? (compEl.innerText || '').trim() : '';
                                    const loc = locEl ? (locEl.innerText || '').trim() : '';
                                    results.push({
                                        text: `${t} — ${comp} (${loc})`.replace(/\\s+/g, ' ').trim(),
                                        href: linkEl.href.split('?')[0]
                                    });
                                }
                            }
                            return results.slice(0, 10);
                        }"""
                    )

                    links_info = page.evaluate(
                        """() => {
                            const anchors = Array.from(document.querySelectorAll('a[href]'));
                            return anchors
                                .map(a => ({text: (a.innerText || '').replace(/\\s+/g, ' ').trim(), href: a.href}))
                                .filter(x => {
                                    const combined = (x.text + ' ' + x.href).toLowerCase();
                                    if (/skip to|sign in|join now|uas\\/login|authwall|report-abuse|help\\.github|forgot-password|#main-content|javascript:/i.test(combined)) {
                                        return false;
                                    }
                                    return x.href.startsWith('mailto:') || x.href.startsWith('tel:') ||
                                           /iletisim|contact|about|hakkimizda|product|urun|cozum|jobs\\/view|ilan|kariyer|career|intern|staj|etkinlik|repositories/i.test(combined);
                                })
                                .slice(0, 20);
                        }"""
                    )

                    # Eger LinkedIn is kartlari bulunduysa relevant_links'in en basina koy
                    merged_links = job_cards + [lk for lk in links_info if lk.get("href") not in {jc.get("href") for jc in job_cards}]

                    clean_text = re.sub(r"\s+", " ", body_text or "").strip()
                    clean_text = re.sub(
                        r"(You signed in with another tab or window.*?Dismiss alert|Skip to (main )?content|We value your privacy.*?(Accept All|Reject All)|Date posted.*?Sign in to create job alert)\s*",
                        " ",
                        clean_text,
                        flags=re.IGNORECASE,
                    ).strip()[:1200]

                    if job_cards:
                        cards_summary = " | ".join(f"{idx}. {jc['text']}" for idx, jc in enumerate(job_cards[:6], start=1))
                        clean_text = f"Canlı İlanlar: {cards_summary} — {clean_text}"[:1200]
                    elif meta_desc and meta_desc[:60].lower() not in clean_text.lower():
                        clean_text = f"{meta_desc} — {clean_text}"[:1200]

                    emails = list(set(re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", (raw_full_text or "") + " " + (body_text or ""))))

                    # Eger kurumsal iletisim sayfasi linki varsa (github report-abuse haric) ziyaret et
                    contact_sub_url = None
                    if "github.com" not in url.lower() and "linkedin.com" not in url.lower():
                        for lk in merged_links:
                            href = lk.get("href", "")
                            if (
                                href.startswith("http")
                                and any(w in href.lower() for w in ["contact", "iletisim"])
                                and not any(bad in href.lower() for bad in ["report-abuse", "help.github"])
                            ):
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
                        "relevant_links": merged_links[:8],
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
            title = (snap.get("title") or url).split("|")[0].split("-")[0].strip() or f"Kaynak #{idx}"
            emails = [e for e in snap.get("emails_detected", []) if not any(x in e.lower() for x in ["privacy@commercetools", "example", "sentry", "onedocs"])]
            links = snap.get("relevant_links", [])
            action_links = [
                lk.get("href")
                for lk in links
                if any(w in (lk.get("href") or "").lower() for w in ["jobs/view", "ilan", "jobs", "etkinlik", "contact", "iletisim", "demo"])
                and "report-abuse" not in (lk.get("href") or "").lower()
            ]
            contact_info = ", ".join(emails[:2]) if emails else ""
            if action_links:
                contact_info = (contact_info + f" ({action_links[0]})").strip()
            if not contact_info:
                contact_info = url

            excerpt = (snap.get("main_excerpt") or snap.get("error") or "")[:260].replace("|", " ").strip()
            rows.append(f"| {idx} | **{title}** | {url} | {excerpt}... | {contact_info} |")
            top_links_md = "\n".join(
                f"  - [{(lk.get('text') or lk.get('href'))[:85]}]({lk.get('href')})"
                for lk in links[:6]
                if lk.get("href")
            )
            details.append(
                f"### {idx}. {title} ({url})\n"
                f"- **Sayfa Başlığı:** {snap.get('title', '-')}\n"
                f"- **Doğrudan Bağlantı / İletişim:** {contact_info}\n"
                f"- **Canlı DOM İçerik Özeti:** {(snap.get('main_excerpt') or '')[:550]}\n"
                + (f"- **Öne Çıkan İlanlar / Alt Bağlantılar:**\n{top_links_md}\n" if top_links_md else "")
            )

        table_header = (
            "## 🌐 Canlı Tarayıcı Araştırma & Eşleşme Raporu\n\n"
            "| # | Kaynak / Platform | Web Adresi | İçerik & İlan Özeti (DOM) | Doğrudan Link / İletişim |\n"
            "|---|---|---|---|---|\n"
        )
        return table_header + "\n".join(rows) + "\n\n---\n\n## 📋 Detaylı Bulgular ve Bağlantılar\n\n" + "\n".join(details)

    def _synthesize_with_llm(self, task: str, snapshots: List[Dict[str, Any]]) -> str:
        """Tarayicinin topladigi gercek DOM verilerini goreve gore yapilandirilmis rapora donusturur."""
        snapshots_json = json.dumps(snapshots, ensure_ascii=False, indent=2)
        sys_prompt = (
            "Sen Kıdemli bir Web Araştırma ve Veri Çıkarım Ajanısın (Browser Extraction Agent).\n"
            "Gerçek tarayıcı (Browser) üzerinden ziyaret edilen sitelerin DOM içerikleri aşağıda verilmiştir.\n"
            "Kullanıcının görevini bu gerçek verilerle ve kurumsal doğrulukla yerine getir.\n"
            "Çıktında mutlaka Markdown karşılaştırma tablosu ve ardından detaylı bulguları / doğrudan başvuru veya iletişim linklerini Türkçe olarak sun."
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

        # Eger kullanici buldugu siteleri tarayicida acip karsisina getirmesini istediyse veya --show-browser aktifse
        t_low = task.lower()
        should_open_for_user = (not headless) or any(
            phrase in t_low
            for phrase in ["karşıma getir", "karsima getir", "tarayıcıdan aç", "tarayicidan ac", "tarayıcıda aç", "ekranda aç"]
        )
        if should_open_for_user:
            urls_to_open: List[str] = []
            # Oncelik 1: Dogrudan tekil is ilani sayfalari (/jobs/view/...)
            for snap in snapshots:
                for lk in snap.get("relevant_links", []):
                    href = lk.get("href", "")
                    if href.startswith("http") and "/jobs/view/" in href.lower():
                        if href not in urls_to_open:
                            urls_to_open.append(href)
            # Oncelik 2: Genel is arama sayfalari
            for snap in snapshots:
                for lk in snap.get("relevant_links", []):
                    href = lk.get("href", "")
                    if href.startswith("http") and any(w in href.lower() for w in ["linkedin.com/jobs", "youthall.com/tr", "coderspace.io"]):
                        if href not in urls_to_open:
                            urls_to_open.append(href)
            if not urls_to_open:
                urls_to_open = [u for u in (visited or target_urls) if "github.com" not in u][:3]
            for open_u in urls_to_open[:3]:
                try:
                    webbrowser.open_new_tab(open_u)
                except Exception:
                    pass

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
    preferred_backend: str = "browser-use"  # "browser-use" (default) | "auto" | "playwright"

    def __init__(
        self,
        default_working_dir: str = "./workspace_project",
        tier: str = "worker",
        agent_idx: int = 0,
        agent_role: str = "browser_researcher",
        default_headless: bool = True,
        preferred_backend: str = "browser-use",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.default_working_dir = default_working_dir
        self.tier = tier
        self.agent_idx = agent_idx
        self.agent_role = agent_role
        self.default_headless = default_headless
        self.preferred_backend = preferred_backend

    def _record_telemetry(
        self,
        result: BrowserTaskResult,
        fallback_triggered: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> None:
        """Her tarayici gorevinin kanitlarini (URL'ler, sure, backend, varsa fallback nedeni) calisma dizinine kaydeder."""
        try:
            work_dir = Path(self.default_working_dir).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
            telemetry_file = work_dir / "browser_telemetry.jsonl"
            entry: Dict[str, Any] = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "agent_role": self.agent_role,
                "backend_used": result.backend_used,
                "success": result.success,
                "task": result.task,
                "visited_urls": result.visited_urls,
                "elapsed_seconds": round(result.elapsed_seconds, 2),
                "fallback_triggered": fallback_triggered,
            }
            if fallback_reason:
                entry["fallback_reason"] = fallback_reason
            with open(telemetry_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as tel_err:
            print(f"[BrowserAutomationTool] Telemetri kayit uyarisi: {tel_err}")

    def _run(
        self,
        task: str,
        start_url: Optional[str] = None,
        headless: Optional[bool] = None,
        **kwargs,
    ) -> str:
        use_headless = self.default_headless if headless is None else headless
        fallback_reason: Optional[str] = None

        # 1. Birincil (Default) Motor: BrowserUseBackend (browser-use + Chromium)
        if self.preferred_backend in ("browser-use", "auto"):
            try:
                import browser_use  # noqa: F401
                bu_backend = BrowserUseBackend(tier=self.tier, agent_idx=self.agent_idx)
                res = bu_backend.execute(task=task, start_url=start_url, headless=use_headless)
                if res.success and res.extracted_content.strip():
                    self._record_telemetry(res, fallback_triggered=False)
                    return res.to_summary_text()
                fallback_reason = res.error or "BrowserUseBackend bos veya tamamlanmamis cikti dondurdu."
                print(
                    f"⚠️ [BrowserAutomationTool] BrowserUseBackend tamamlanamadi ({fallback_reason}). "
                    f"PlaywrightSmartBackend fallback devreye aliniyor..."
                )
            except Exception as bu_exc:
                fallback_reason = f"{type(bu_exc).__name__}: {bu_exc}"
                print(
                    f"⚠️ [BrowserAutomationTool] BrowserUseBackend istisna firlatti ({fallback_reason}). "
                    f"PlaywrightSmartBackend fallback devreye aliniyor..."
                )

        # 2. Yedek (Fallback) Motor: PlaywrightSmartBackend
        pw_backend = PlaywrightSmartBackend(tier=self.tier, agent_idx=self.agent_idx)
        res = pw_backend.execute(task=task, start_url=start_url, headless=use_headless)
        if fallback_reason:
            res.error = f"[Fallback Sebebi: {fallback_reason}]"
        self._record_telemetry(
            res,
            fallback_triggered=bool(fallback_reason),
            fallback_reason=fallback_reason,
        )
        return res.to_summary_text()

