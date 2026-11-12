"""Capture real screenshots of the running Streamlit UI with Playwright (headless Chrome).

Usage: start `streamlit run src/raglens/ui/app.py` first, then run this script.
Set CHROME_PATH to use an installed Chrome instead of Playwright's own Chromium.
"""

import os
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

URL = os.environ.get("RAGLENS_UI_URL", "http://localhost:8501")
OUT = Path("docs/screenshots")


def ask(page: Page, question: str, retriever: str, name: str) -> None:
    """Type a question, pick a retriever, submit and save a full-page screenshot."""
    page.get_by_role("tab", name="Ask").click()
    page.get_by_text(retriever, exact=True).click()
    box = page.get_by_label("Ask a question about the corpus")
    box.fill(question)
    box.press("Enter")
    page.get_by_role("button", name="Ask").click()
    page.get_by_text("Retrieved passages").wait_for(timeout=30000)
    page.wait_for_timeout(800)
    page.screenshot(path=str(OUT / name), full_page=True)


def main() -> None:
    """Capture the Ask tab (answered, refused, expanded evidence) and the Evaluation tab."""
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=os.environ.get("CHROME_PATH") or None)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(URL)
        page.get_by_text("RAGLens").first.wait_for(timeout=60000)
        ask(
            page,
            "How does Reciprocal Rank Fusion combine BM25 and dense rankings?",
            "hybrid",
            "ask_hybrid.png",
        )
        page.locator("summary").filter(has_text="cited").filter(has_text="✅").first.click()
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "ask_evidence.png"), full_page=True)
        ask(page, "What is the capital of Australia?", "hybrid", "ask_refused.png")
        page.get_by_role("tab", name="Evaluation").click()
        page.get_by_text("Unanswerable refused").wait_for(timeout=30000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "evaluation.png"), full_page=True)
        browser.close()
    print(f"Screenshots written to {OUT}/")


if __name__ == "__main__":
    main()
