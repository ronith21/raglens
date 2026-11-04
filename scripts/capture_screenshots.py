"""Capture real screenshots and a demo video of the running RAGLens UI with Playwright.

The API and the UI are started automatically when they are not already running (pass
``--no-start`` to disable). Requires Chromium for Playwright (``python -m playwright install
chromium``) and, for the video, ``ffmpeg`` on PATH.

Usage::

    python scripts/capture_screenshots.py [--ui http://localhost:8501] [--api http://localhost:8000]
                                          [--no-video] [--no-start]
                                          [--browser-path /path/to/chrome]

Outputs:
    docs/screenshots/*.png         ui_ask, ui_citations, ui_refusal, ui_multi_query, ui_eval,
                                   ui_eval_charts, ui_live_eval, ui_analytics, ui_documents,
                                   api_docs
    docs/proof/raglens_demo.mp4    screen recording of the whole walkthrough
    docs/proof/raglens_demo.gif    small looping preview for the README

The video is recorded with Chrome's DevTools screencast (timestamped JPEG frames) and assembled
by ffmpeg, so it does not depend on Playwright's own bundled ffmpeg build.
"""

import argparse
import base64
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from playwright.sync_api import CDPSession, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

SHOTS = Path("docs/screenshots")
PROOF = Path("docs/proof")
FRAMES = PROOF / "_video_raw"
VIEWPORT = {"width": 1440, "height": 900}

ANSWERABLE = "How much is protected per depositor if my bank fails?"
UNANSWERABLE = "What was Deutsche Bank's CET1 ratio in 2024?"
MULTI_QUERY = "What does SCA require for online payments?"
# The demo runs on the banking corpus, with its held-out questions and benchmark results.
DEMO_ENV = {
    "CORPUS_DIR": "data/corpora/banking",
    "INDEX_DIR": "data/index_banking",
    "EVAL_DATASET": "data/eval/banking_test.jsonl",
    "EVAL_RESULTS_PATH": "docs/proof/banking/eval_results.json",
}
DEMO_QUERY_LOG = Path("data/logs/demo_queries.db")


class ScreencastRecorder:
    """Collects Chrome DevTools screencast frames with their capture timestamps."""

    def __init__(self, page: Page, out_dir: Path) -> None:
        self.page = page
        self.out_dir = out_dir
        self.frames: list[tuple[Path, float]] = []
        self._cdp: CDPSession | None = None

    def start(self) -> None:
        """Begin streaming frames (Chrome sends one whenever the page repaints)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._cdp = self.page.context.new_cdp_session(self.page)
        self._cdp.on("Page.screencastFrame", self._on_frame)
        self._cdp.send(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": 85,
                "maxWidth": VIEWPORT["width"],
                "maxHeight": VIEWPORT["height"],
                "everyNthFrame": 1,
            },
        )

    def _on_frame(self, params: dict[str, Any]) -> None:
        path = self.out_dir / f"frame_{len(self.frames):05d}.jpg"
        path.write_bytes(base64.b64decode(params["data"]))
        self.frames.append((path, float(params["metadata"]["timestamp"])))
        if self._cdp is not None:
            self._cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})

    def stop(self, hold_last_s: float = 2.0) -> Path | None:
        """Stop streaming and write an ffmpeg concat list; returns its path (None if no frames)."""
        if self._cdp is not None:
            self.page.wait_for_timeout(500)  # flush pending frames
            self._cdp.send("Page.stopScreencast")
        if not self.frames:
            return None
        lines = []
        for (path, ts), (_, next_ts) in zip(self.frames, self.frames[1:], strict=False):
            lines += [f"file '{path.resolve()}'", f"duration {max(next_ts - ts, 0.001):.3f}"]
        last = self.frames[-1][0].resolve()
        lines += [f"file '{last}'", f"duration {hold_last_s:.3f}", f"file '{last}'"]
        listing = self.out_dir / "frames.txt"
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return listing


def wait_for_http(url: str, timeout_s: float = 60.0) -> None:
    """Block until ``url`` answers 200, or exit with a clear message."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=3.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    sys.exit(f"Timed out waiting for {url}. Is the server running?")


def pause(page: Page, seconds: float) -> None:
    """Sleep while still letting Playwright deliver events (screencast frames)."""
    page.wait_for_timeout(int(seconds * 1000))


def settle(page: Page, pause_s: float = 0.6) -> None:
    """Wait for Streamlit to finish re-running the script."""
    page.wait_for_load_state("networkidle")
    page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=30_000)
    pause(page, pause_s)


def scroll(page: Page, dy: int, steps: int = 12) -> None:
    """Scroll smoothly so the recording is easy to follow."""
    for _ in range(steps):
        page.mouse.wheel(0, dy / steps)
        pause(page, 0.04)
    pause(page, 0.6)


def shot(page: Page, name: str) -> Path:
    """Save a screenshot under docs/screenshots."""
    path = SHOTS / f"{name}.png"
    page.screenshot(path=str(path))
    print(f"screenshot: {path}")
    return path


def ask(page: Page, question: str) -> None:
    """Type a question (visibly, for the video) and submit it."""
    box = page.get_by_label("Question")
    box.click()
    box.fill("")
    box.type(question, delay=22)
    box.press("Control+Enter")  # commits the text area value in Streamlit
    settle(page, 0.3)
    page.get_by_role("button", name="Ask", exact=True).click()
    page.locator(".answer-card").wait_for(timeout=30_000)
    settle(page, 1.2)


def walkthrough(page: Page, ui_url: str, api_url: str) -> None:
    """Scripted demo: ask with evidence, refuse, multi-query, evaluate, analytics, documents."""
    page.goto(ui_url)
    page.get_by_text("API online").wait_for(timeout=60_000)
    settle(page, 1.5)

    ask(page, ANSWERABLE)
    shot(page, "ui_ask")
    scroll(page, 560)
    shot(page, "ui_citations")
    scroll(page, -2000, steps=8)

    ask(page, UNANSWERABLE)
    shot(page, "ui_refusal")

    page.get_by_text("Multi-query rewriting").click()
    settle(page)
    ask(page, MULTI_QUERY)
    shot(page, "ui_multi_query")

    page.get_by_role("tab", name="Evaluation").click()
    settle(page, 1.5)
    shot(page, "ui_eval")
    scroll(page, 700)
    shot(page, "ui_eval_charts")
    page.get_by_role("button", name="Run evaluation").click()
    page.get_by_text("Answer F1", exact=True).wait_for(timeout=180_000)
    settle(page, 1.0)
    scroll(page, 900)
    shot(page, "ui_live_eval")

    scroll(page, -4000, steps=8)
    page.get_by_role("tab", name="Analytics").click()
    settle(page, 1.5)
    shot(page, "ui_analytics")

    page.get_by_role("tab", name="Documents").click()
    settle(page, 1.5)
    shot(page, "ui_documents")
    pause(page, 1.5)

    # Swagger UI is captured in a separate, unrecorded tab and skipped (never faked) if its
    # assets cannot load.
    docs_page = page.context.new_page()
    try:
        docs_page.goto(f"{api_url}/docs")
        docs_page.get_by_text("RAGLens API").first.wait_for(timeout=15_000)
        docs_page.get_by_text("/ask", exact=True).first.click()
        pause(docs_page, 1.0)
        shot(docs_page, "api_docs")
    except PlaywrightTimeoutError:
        print("skipped api_docs: Swagger UI assets could not be loaded")
    finally:
        docs_page.close()


def _is_up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


def start_servers(api_url: str, ui_url: str) -> list[subprocess.Popen[bytes]]:
    """Start the API and UI as subprocesses unless they are already running.

    Servers started here log questions to a fresh demo query log, so the analytics screenshot
    shows exactly this walkthrough.
    """
    procs: list[subprocess.Popen[bytes]] = []
    env = dict(os.environ, API_URL=api_url, QUERY_LOG_PATH=str(DEMO_QUERY_LOG), **DEMO_ENV)
    if not _is_up(f"{api_url}/health"):
        for suffix in ("", "-wal", "-shm"):  # SQLite write-ahead-log side files
            Path(f"{DEMO_QUERY_LOG}{suffix}").unlink(missing_ok=True)
        port = str(urlparse(api_url).port or 8000)
        procs.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "raglens.api.main:app", "--port", port],
                env=env,
            )
        )
        print(f"started API on :{port}")
    if not _is_up(ui_url):
        port = str(urlparse(ui_url).port or 8501)
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "src/raglens/ui/app.py",
                    "--server.headless=true",
                    f"--server.port={port}",
                ],
                env=env,
            )
        )
        print(f"started UI on :{port}")
    return procs


def encode_video(listing: Path) -> list[Path]:
    """Assemble frames into an H.264 MP4 and a README-sized GIF."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found on PATH; skipping video encoding")
        return []
    mp4 = PROOF / "raglens_demo.mp4"
    gif = PROOF / "raglens_demo.gif"
    concat = ["-f", "concat", "-safe", "0", "-i", str(listing)]
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            *concat,
            "-vf",
            "fps=25,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
    )
    scale = "fps=6,scale=960:-1:flags=lanczos"
    palette = FRAMES / "palette.png"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-vf",
            f"{scale},palettegen=max_colors=96",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-i",
            str(palette),
            "-lavfi",
            f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
            str(gif),
        ],
        check=True,
    )
    return [mp4, gif]


def main() -> None:
    """Parse arguments, run the walkthrough, and write screenshots and video."""
    parser = argparse.ArgumentParser(description="Capture RAGLens screenshots and demo video.")
    parser.add_argument("--ui", default="http://localhost:8501")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--no-start", action="store_true", help="Do not start the API/UI if they are not running"
    )
    parser.add_argument(
        "--browser-path",
        default=None,
        help="Use an existing Chromium/Chrome binary instead of Playwright's download",
    )
    args = parser.parse_args()

    SHOTS.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(FRAMES, ignore_errors=True)
    servers = [] if args.no_start else start_servers(args.api, args.ui)
    try:
        wait_for_http(f"{args.api}/health")
        wait_for_http(args.ui)
        capture(args)
    finally:
        for proc in servers:
            proc.terminate()
            proc.wait(timeout=20)


def capture(args: argparse.Namespace) -> None:
    """Run the browser walkthrough and encode the recording."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.browser_path)
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        page = context.new_page()
        recorder = None if args.no_video else ScreencastRecorder(page, FRAMES)
        if recorder:
            recorder.start()
        walkthrough(page, args.ui, args.api)
        listing = recorder.stop() if recorder else None
        context.close()
        browser.close()

    if listing is not None:
        for out in encode_video(listing):
            print(f"video: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    shutil.rmtree(FRAMES, ignore_errors=True)


if __name__ == "__main__":
    main()
