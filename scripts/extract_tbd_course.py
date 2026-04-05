#!/usr/bin/env python3
"""Extract TBD Course transcripts via Firecrawl Browser API + Vimeo Player JS.

Prerequisites:
    1. Set your Firecrawl API key:
       export FIRECRAWL_API_KEY=fc-your-key-here
    2. Run:
       python3 scripts/extract_tbd_course.py

The script creates a Firecrawl cloud browser session, logs into the course,
plays each Vimeo video muted, captures captions via cuechange events,
and saves transcripts to docs/tbd-course/.

Resumable: re-run to pick up where you left off (reads progress.json).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import urllib.request
import urllib.error


class _SimpleHTTP:
    """Minimal HTTP client using urllib (no external deps)."""

    class _Response:
        def __init__(self, resp):
            self.status_code = resp.status
            self._data = resp.read()

        def json(self):
            return json.loads(self._data)

    def post(self, url, headers=None, **kwargs):
        body = kwargs.get("json")
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=headers or {})
        req.add_header("Content-Type", "application/json")
        try:
            resp = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            resp = e
        return self._Response(resp)

    def delete(self, url, headers=None):
        req = urllib.request.Request(url, method="DELETE", headers=headers or {})
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            resp = e
        return self._Response(resp)


http = _SimpleHTTP()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COURSE_URL = "https://tradetravelchill.club/courses/tbd-system-pt1/"
LOGIN_URL = "https://tradetravelchill.club/membership-account/"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "tbd-course"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
FIRECRAWL_API = "https://api.firecrawl.dev/v2"
POLL_INTERVAL = 10  # seconds between progress polls
SAFETY_MULTIPLIER = 1.8  # timeout = duration * this

# Course credentials
COURSE_USERNAME = "davina"
COURSE_PASSWORD = "9#4Bsa8CT%ZT@"
DEFAULT_API_KEY = "fc-8d93287682984229859187d9752ada6d"

# Vimeo transcript capture JS (injected into page via page.evaluate)
CAPTURE_JS = r"""
return new Promise((resolve) => {
    const script = document.createElement('script');
    script.src = 'https://player.vimeo.com/api/player.js';
    script.onload = () => {
        const iframe = document.querySelector('iframe[src*="vimeo"]');
        if (!iframe) { resolve(JSON.stringify({error: 'no_vimeo_iframe', cues: []})); return; }
        const player = new Vimeo.Player(iframe);
        window._allCues = [];
        window._captureComplete = false;
        player.setVolume(0);
        player.enableTextTrack('en', 'captions').catch(() => player.enableTextTrack('en', 'subtitles')).catch(() => {});
        player.on('cuechange', (data) => {
            if (data.cues && data.cues.length > 0) {
                data.cues.forEach(cue => {
                    const exists = window._allCues.some(c => c.startTime === cue.startTime && c.text === cue.text);
                    if (!exists) window._allCues.push({text: cue.text, startTime: cue.startTime, endTime: cue.endTime});
                });
            }
        });
        player.on('ended', () => { window._captureComplete = true; });
        player.getDuration().then(d => {
            window._videoDuration = d;
            setTimeout(() => { window._captureComplete = true; }, d * SAFETY_MULT * 1000);
        });
        player.play();
        // Try 2x speed (cloud browser, no autoplay restrictions)
        try { player.setPlaybackRate(2); } catch(e) {}
        setInterval(async () => {
            if (window._captureComplete) return;
            try {
                const p = await player.getPaused();
                if (p) { await player.play(); try { player.setPlaybackRate(2); } catch(e) {} }
            } catch(e){}
        }, 2000);
        resolve(JSON.stringify({status: 'started'}));
    };
    script.onerror = () => resolve(JSON.stringify({error: 'sdk_load_failed', cues: []}));
    document.head.appendChild(script);
});
""".replace("SAFETY_MULT", str(SAFETY_MULTIPLIER))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract TBD Course transcripts")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("FIRECRAWL_API_KEY", DEFAULT_API_KEY),
        help="Firecrawl API key (or set FIRECRAWL_API_KEY env var)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel browser sessions (default: 4, max 5 on most plans)",
    )
    parser.add_argument(
        "--course-url",
        type=str,
        default=None,
        help="Course URL to extract (default: TBD System pt1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: docs/tbd-course)",
    )
    parser.add_argument(
        "--lesson",
        type=str,
        default=None,
        help="Extract only this lesson URL (for testing a single lesson)",
    )
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Skip lesson discovery, use cached lesson list from progress.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract lessons even if already in progress.json",
    )
    return parser.parse_args()


class FirecrawlBrowser:
    """Wrapper around Firecrawl v2 Browser API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.session_id = None

    def create_session(self, ttl: int = 3600, save_changes: bool = True) -> str:
        """Create a browser session with persistent profile."""
        resp = http.post(
            f"{FIRECRAWL_API}/browser",
            json={"profile": {"name": "tbd-course", "saveChanges": save_changes}, "ttl": ttl},
            headers=self.headers,
        )
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Failed to create session: {data}")
        self.session_id = data["id"]
        print(f"  Session: {self.session_id}")
        if data.get("liveViewUrl"):
            print(f"  Live view: {data['liveViewUrl']}")
        return self.session_id

    def execute(self, code: str, language: str = "node") -> dict:
        """Execute code in the browser session."""
        if not self.session_id:
            raise RuntimeError("No active session")
        resp = http.post(
            f"{FIRECRAWL_API}/browser/{self.session_id}/execute",
            json={"code": code, "language": language},
            headers=self.headers,
        )
        return resp.json()

    def destroy_session(self):
        """Clean up the browser session."""
        if self.session_id:
            http.delete(
                f"{FIRECRAWL_API}/browser/{self.session_id}",
                headers=self.headers,
            )
            print(f"  Session {self.session_id} destroyed.")
            self.session_id = None

    def goto(self, url: str, wait: int = 3000) -> dict:
        """Navigate to URL and return page info."""
        code = f'await page.goto("{url}"); await page.waitForTimeout({wait}); JSON.stringify({{url: page.url(), title: await page.title()}});'
        result = self.execute(code)
        if result.get("success") and result.get("result"):
            return json.loads(result["result"])
        return {"url": url, "title": "", "error": result.get("error", result.get("stderr", ""))}

    def login(self) -> bool:
        """Log into tradetravelchill.club."""
        print("  Logging in...")
        info = self.goto(LOGIN_URL)
        if "login" not in info.get("url", "").lower() and "log-in" not in info.get("url", "").lower():
            # Already logged in (profile cookies)
            if "membership" in info.get("title", "").lower() or "account" in info.get("title", "").lower():
                print("  Already logged in (profile cookies)!")
                return True

        # Navigate to login page
        info = self.goto("https://tradetravelchill.club/login/")
        pwd_escaped = COURSE_PASSWORD.replace("\\", "\\\\").replace('"', '\\"')
        code = f'''
            await page.fill('input[name="log"]', '{COURSE_USERNAME}');
            await page.fill('input[name="pwd"]', '{pwd_escaped}');
            await page.click('input[type="submit"], button[type="submit"], #wp-submit');
            await page.waitForTimeout(5000);
            JSON.stringify({{url: page.url(), title: await page.title()}});
        '''
        result = self.execute(code)
        if result.get("success") and result.get("result"):
            data = json.loads(result["result"])
            if "account" in data.get("title", "").lower() or "membership" in data.get("url", "").lower():
                print(f"  Logged in! -> {data['title']}")
                return True
            print(f"  Login may have failed: {data}")
        return False


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"lessons": [], "completed": {}}


def save_progress(progress: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"[{m:02d}:{s:02d}]"


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def discover_lessons(browser: FirecrawlBrowser) -> list[dict]:
    """Scrape course overview pages for all lesson URLs.

    The course uses JS-based pagination (click '>' arrow), not URL params.
    """
    lessons = []
    seen_urls = set()

    print(f"\n  Loading course: {COURSE_URL}")
    browser.goto(COURSE_URL, wait=5000)

    SCRAPE_CODE = """
        const links = [...document.querySelectorAll('a')].filter(a => a.href.includes('/lessons/'));
        const unique = [];
        const seen = new Set();
        links.forEach(l => {
            if (!seen.has(l.href)) {
                seen.add(l.href);
                // Try to find section heading above this lesson
                let section = 'Unknown';
                let el = l.closest('.ld-item-list-item, .ld-table-list-item, tr, li');
                if (el) {
                    let prev = el;
                    while (prev) {
                        prev = prev.previousElementSibling;
                        if (prev && (prev.tagName === 'H2' || prev.tagName === 'H3' || prev.classList.contains('ld-lesson-section-heading'))) {
                            section = prev.textContent.trim();
                            break;
                        }
                    }
                }
                unique.push({
                    title: l.textContent.trim().split('\\n')[0].trim(),
                    url: l.href,
                    section: section
                });
            }
        });
        return JSON.stringify(unique);
    """

    for page_num in range(1, 10):
        print(f"  Scraping page {page_num}...")

        result = browser.execute(f"JSON.stringify(await page.evaluate(() => {{ {SCRAPE_CODE} }}));")
        new_on_page = 0
        total_on_page = 0
        if result.get("success") and result.get("result"):
            raw = result["result"]
            items = json.loads(raw)
            if isinstance(items, str):
                items = json.loads(items)
            total_on_page = len(items)
            for item in items:
                if item["url"] not in seen_urls and item["title"]:
                    seen_urls.add(item["url"])
                    lessons.append(item)
                    new_on_page += 1
                    print(f"    Found: {item['title'][:60]}")

        print(f"    ({new_on_page} new, {total_on_page} total links on page {page_num})")

        if new_on_page == 0:
            break

        # Click the LearnDash 'next' pagination link and wait for AJAX content update
        next_page = page_num + 1
        next_code = f"""
            // LearnDash pagination: a.next inside div.ld-pages
            const nextBtn = document.querySelector('.ld-pages a.next:not(.disabled)');
            if (!nextBtn) return 'no_next_button';
            nextBtn.click();
            return 'clicked_page_{next_page}';
        """
        nav_result = browser.execute(f"JSON.stringify(await page.evaluate(() => {{ {next_code} }}));")
        nav_text = str(nav_result.get("result", ""))
        print(f"    Pagination: {nav_text}")
        if "no_next_button" in nav_text:
            print("    No next page button found.")
            break

        # LearnDash uses AJAX to load new content — wait for it
        time.sleep(3)
        browser.execute('await page.waitForTimeout(3000); JSON.stringify("waited");')

    print(f"\n  Total lessons discovered: {len(lessons)}")

    return lessons


def extract_transcript(browser: FirecrawlBrowser, lesson_url: str) -> dict:
    """Navigate to lesson, play video, collect cues."""
    browser.goto(lesson_url, wait=10000)

    # Check for Vimeo iframe (with retry — iframe may load late)
    found = False
    for attempt in range(3):
        check = browser.execute(
            'JSON.stringify((await page.$$("iframe[src*=vimeo]")).length > 0);'
        )
        if check.get("result") == "true":
            found = True
            break
        print(f"    Waiting for Vimeo iframe (attempt {attempt + 1})...")
        time.sleep(5)

    if not found:
        print("    No Vimeo iframe found.")
        return {"cues": [], "error": "no_vimeo_iframe"}

    # Click iframe for autoplay
    browser.execute(
        'await page.locator("iframe[src*=vimeo]").first().click(); await page.waitForTimeout(1500); JSON.stringify("clicked");'
    )

    # Inject capture JS
    print("    Starting capture...")
    inject_code = f'const _cap = await page.evaluate(() => {{ {CAPTURE_JS} }}); _cap;'
    result = browser.execute(inject_code)

    if result.get("success") and result.get("result"):
        data = json.loads(result["result"])
        if data.get("error"):
            print(f"    Error: {data['error']}")
            return {"cues": [], "error": data["error"]}

    # Poll for completion
    print("    Video playing, capturing captions...")
    last_count = 0
    stall_count = 0
    max_stall = 30  # 30 * POLL_INTERVAL = 5 min stall limit

    while True:
        time.sleep(POLL_INTERVAL)
        try:
            status_result = browser.execute(
                'JSON.stringify(await page.evaluate(() => ({cueCount: (window._allCues || []).length, complete: window._captureComplete || false, duration: window._videoDuration || 0})));'
            )
        except Exception as e:
            print(f"    Poll error: {e}")
            stall_count += 1
            if stall_count >= max_stall:
                print(f"    Too many poll errors, stopping.")
                break
            continue

        if not status_result.get("success") or not status_result.get("result"):
            err = status_result.get("error", status_result.get("stderr", "unknown"))
            print(f"    Poll failed: {err}")
            stall_count += 1
            if stall_count >= max_stall:
                print(f"    Too many poll failures, stopping.")
                break
            continue

        try:
            status = json.loads(status_result["result"])
        except (json.JSONDecodeError, TypeError) as e:
            print(f"    Poll parse error: {e} - raw: {status_result.get('result', '')[:100]}")
            stall_count += 1
            if stall_count >= max_stall:
                break
            continue

        current_count = status.get("cueCount", 0)

        if current_count != last_count:
            print(f"    Cues: {current_count}")
            last_count = current_count
            stall_count = 0
        else:
            stall_count += 1

        if status.get("complete"):
            print(f"    Capture complete! {current_count} cues")
            break

        if stall_count >= max_stall:
            print(f"    Stalled for {max_stall * POLL_INTERVAL}s, stopping.")
            break

    # Retrieve final cues — try multiple times
    print(f"    Retrieving cues (last seen: {last_count})...")
    for attempt in range(3):
        try:
            final = browser.execute(
                'JSON.stringify(await page.evaluate(() => ({cues: (window._allCues || []).sort((a,b) => a.startTime - b.startTime), duration: window._videoDuration || 0})));'
            )
            if final.get("success") and final.get("result"):
                data = json.loads(final["result"])
                if data.get("cues"):
                    return data
                print(f"    Retrieval attempt {attempt + 1}: 0 cues in result")
            else:
                print(f"    Retrieval attempt {attempt + 1} failed: {final.get('error', 'unknown')}")
        except Exception as e:
            print(f"    Retrieval attempt {attempt + 1} error: {e}")
        time.sleep(3)

    # Last resort: return whatever count we saw during polling
    print(f"    WARNING: Could not retrieve cues, but saw {last_count} during polling")
    return {"cues": [], "error": "retrieval_failed", "last_seen_count": last_count}


def extract_page_text(browser: FirecrawlBrowser) -> str:
    """Extract written content from the lesson page."""
    code = """
        const selectors = [
            '.learndash-wrapper .ld-tab-content',
            '.learndash_post_sfwd-lessons .entry-content',
            '.ld-lesson-content', 'article .entry-content', '.lesson-content'
        ];
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el) {
                const clone = el.cloneNode(true);
                clone.querySelectorAll('script, style, iframe, nav').forEach(e => e.remove());
                const text = clone.textContent.trim();
                if (text.length > 20) return text;
            }
        }
        return '';
    """
    result = browser.execute(
        f'JSON.stringify(await page.evaluate(() => {{ {code} }}));'
    )
    if result.get("success") and result.get("result"):
        return result["result"].strip()
    return ""


def save_lesson(
    lesson_num: int, title: str, section: str, url: str,
    page_text: str, cues: list[dict], duration: float | None = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(title)
    filename = f"{lesson_num:02d}_{slug}.md"
    filepath = OUTPUT_DIR / filename

    transcript_lines = []
    for cue in cues:
        ts = format_timestamp(cue.get("startTime", 0))
        text = cue.get("text", "").strip()
        if text:
            transcript_lines.append(f"{ts} {text}")

    now = datetime.now(timezone.utc).isoformat()
    content = f"""---
lesson: {lesson_num}
title: "{title}"
section: "{section}"
url: {url}
cues: {len(cues)}
duration_seconds: {int(duration) if duration else 'null'}
extracted: {now}
---

# {title}

"""
    if page_text:
        content += f"## Page Content\n\n{page_text}\n\n"
    content += "## Transcript\n\n"
    if transcript_lines:
        content += "\n".join(transcript_lines) + "\n"
    else:
        content += "_No captions captured for this lesson._\n"
    filepath.write_text(content)
    return filepath


def generate_index(progress: dict) -> None:
    index_path = OUTPUT_DIR / "_index.md"
    lines = ["# TBD System Course — Extracted Transcripts\n"]
    current_section = None
    for entry in progress.get("completed_list", []):
        if entry.get("section") != current_section:
            current_section = entry["section"]
            lines.append(f"\n## {current_section}\n")
        num = entry["num"]
        title = entry["title"]
        slug = slugify(title)
        filename = f"{num:02d}_{slug}.md"
        cues = entry.get("cues", 0)
        lines.append(f"- [{num:02d}. {title}]({filename}) ({cues} cues)")
    index_path.write_text("\n".join(lines) + "\n")


_progress_lock = threading.Lock()


def _worker_extract(
    api_key: str,
    worker_id: int,
    jobs: list[tuple[int, dict]],
    progress: dict,
    force: bool,
) -> list[dict]:
    """Worker thread: create own session, login, extract assigned lessons."""
    tag = f"[W{worker_id}]"
    browser = FirecrawlBrowser(api_key)
    results = []

    try:
        print(f"{tag} Creating session...")
        # All workers read-only — profile cookies already saved from initial login
        browser.create_session(ttl=3600, save_changes=False)
        if not browser.login():
            print(f"{tag} Login failed!")
            return results

        for num, lesson in jobs:
            title = lesson["title"]
            url = lesson["url"]
            section = lesson.get("section", "Unknown")

            with _progress_lock:
                if url in progress.get("completed", {}) and not force:
                    print(f"{tag} [{num}] SKIP (done): {title}")
                    continue

            print(f"{tag} [{num}] Extracting: {title}")

            # Check session alive
            check = browser.execute('JSON.stringify("alive");')
            if not check.get("success"):
                print(f"{tag} Session expired, recreating...")
                try:
                    browser.destroy_session()
                except Exception:
                    pass
                browser.create_session(ttl=3600, save_changes=False)
                browser.login()

            try:
                result = extract_transcript(browser, url)
                page_text = extract_page_text(browser)
                cues = result.get("cues", [])
                filepath = save_lesson(num, title, section, url, page_text, cues, result.get("duration"))

                # Only mark as completed if we actually got cues
                if len(cues) > 0:
                    with _progress_lock:
                        completed = progress.setdefault("completed", {})
                        completed_list = progress.setdefault("completed_list", [])
                        completed[url] = {
                            "num": num, "title": title, "cues": len(cues),
                            "file": str(filepath.name),
                            "extracted": datetime.now(timezone.utc).isoformat(),
                        }
                        completed_list[:] = [e for e in completed_list if e.get("url") != url]
                        completed_list.append({"num": num, "title": title, "section": section, "url": url, "cues": len(cues)})
                        completed_list.sort(key=lambda e: e["num"])
                        save_progress(progress)

                print(f"{tag} [{num}] Captured {len(cues)} cues -> {filepath.name}")
                if len(cues) == 0:
                    print(f"{tag} [{num}] WARNING: 0 cues — will retry on next run")
                results.append({"num": num, "title": title, "cues": len(cues)})

            except Exception as e:
                print(f"{tag} [{num}] ERROR: {e}")
                continue

    finally:
        browser.destroy_session()

    return results


def main() -> None:
    global COURSE_URL, OUTPUT_DIR, PROGRESS_FILE
    args = parse_args()

    if not args.api_key:
        print("ERROR: No Firecrawl API key.")
        print("Set FIRECRAWL_API_KEY env var or use --api-key")
        sys.exit(1)

    # Allow overriding course URL and output dir
    if args.course_url:
        COURSE_URL = args.course_url
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        # Auto-derive output dir from course URL slug
        slug = COURSE_URL.rstrip("/").split("/")[-1]
        OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / slug
    PROGRESS_FILE = OUTPUT_DIR / "progress.json"

    progress = load_progress()
    num_workers = min(args.workers, 4)  # Firecrawl plan limit: 5 concurrent, leave 1 free

    print("=" * 60)
    print("  TBD Course Transcript Extractor (Firecrawl)")
    print(f"  Course: {COURSE_URL}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Workers: {num_workers}")
    print("=" * 60)

    # Use one session for discovery and single-lesson mode
    browser = FirecrawlBrowser(args.api_key)

    try:
        print("\nCreating browser session...")
        browser.create_session(ttl=3600, save_changes=False)

        if not browser.login():
            print("ERROR: Login failed.")
            return

        # Single lesson mode (no parallelism needed)
        if args.lesson:
            print(f"\nExtracting single lesson: {args.lesson}")
            result = extract_transcript(browser, args.lesson)
            page_text = extract_page_text(browser)
            cues = result.get("cues", [])
            filepath = save_lesson(0, "single-lesson", "Test", args.lesson, page_text, cues, result.get("duration"))
            print(f"\nCaptured {len(cues)} cues -> {filepath}")
            return

        # Discover lessons
        if args.skip_discovery and progress.get("lessons"):
            lessons = progress["lessons"]
            print(f"\nUsing cached lesson list ({len(lessons)} lessons).")
        else:
            print("\nDiscovering lessons...")
            lessons = discover_lessons(browser)
            if not lessons:
                print("ERROR: No lessons found.")
                return
            progress["lessons"] = lessons
            save_progress(progress)

    finally:
        browser.destroy_session()

    # Build job list (skip already completed)
    completed = progress.get("completed", {})
    all_jobs = []
    for i, lesson in enumerate(lessons):
        num = i + 1
        if lesson["url"] in completed and not args.force:
            print(f"SKIP (done): [{num}] {lesson['title']}")
        else:
            all_jobs.append((num, lesson))

    total = len(lessons)
    remaining = len(all_jobs)
    print(f"\n{total - remaining}/{total} already done, {remaining} remaining.")

    if remaining == 0:
        print("Nothing to do!")
        generate_index(progress)
        return

    # Cap workers to remaining jobs
    actual_workers = min(num_workers, remaining)
    est_minutes = (remaining / actual_workers) * 8  # ~8 min per lesson at 2x
    print(f"Running {actual_workers} parallel workers. Estimated time: ~{int(est_minutes)} minutes.\n")

    # Distribute jobs round-robin across workers
    worker_jobs: list[list[tuple[int, dict]]] = [[] for _ in range(actual_workers)]
    for idx, job in enumerate(all_jobs):
        worker_jobs[idx % actual_workers].append(job)

    # Launch workers
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = []
        for wid in range(actual_workers):
            if worker_jobs[wid]:
                f = executor.submit(
                    _worker_extract, args.api_key, wid + 1,
                    worker_jobs[wid], progress, args.force,
                )
                futures.append(f)
                time.sleep(2)  # stagger session creation to avoid profile lock race

        # Wait for all workers
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Worker error: {e}")

    generate_index(progress)
    done_final = len(progress.get("completed", {}))
    print(f"\n{'=' * 60}")
    print(f"  Extraction complete: {done_final}/{total} lessons")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
