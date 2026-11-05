"""Ask the Streamlit UI about a booby-trapped document; report script runs and outside requests.

Usage: python scripts/security_ui_probe.py UI_URL SCREENSHOT_PATH
Index docs/proof/security/payload_document.md first (it contains HTML, Markdown and URL payloads).
"""

# ruff: noqa: E501

import asyncio
import sys

from playwright.async_api import async_playwright

UI = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8813"
OUT = sys.argv[2] if len(sys.argv) > 2 else "xss_ui.png"


async def main() -> None:
    dialogs: list[str] = []
    outside: list[str] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1400, "height": 1800})

        async def on_dialog(d):
            dialogs.append(d.message)
            await d.dismiss()

        page.on("dialog", lambda d: asyncio.ensure_future(on_dialog(d)))
        page.on("request", lambda r: outside.append(r.url) if "evil.example" in r.url else None)
        await page.goto(UI)
        box = page.get_by_label("Question")
        await box.wait_for(timeout=60000)
        await box.fill("What is the xsscanary leverage ratio minimum exposure measure?")
        await page.get_by_role("button", name="Ask").click()
        await page.get_by_text("Retrieved passages").wait_for(timeout=60000)
        await page.wait_for_timeout(1500)
        for exp in await page.locator("[data-testid='stExpander'] summary").all():
            if await exp.is_visible():
                await exp.click()
        await page.wait_for_timeout(2000)
        dom = await page.evaluate(
            """() => ({
              onerror: document.querySelectorAll('[onerror],[onload]').length,
              scripts_in_app: [...document.querySelectorAll('[data-testid=stMain] script')].length,
              js_links: [...document.querySelectorAll('a')].filter(a => (a.getAttribute('href')||'').toLowerCase().startsWith('javascript')).map(a => a.outerHTML),
              iframes_evil: [...document.querySelectorAll('iframe')].filter(f => (f.src||'').includes('evil')).length,
              escaped_payload_visible: document.body.innerText.includes("onerror=alert('body-xss')"),
              canary_shown: document.body.innerText.includes('xsscanary'),
            })"""
        )
        await page.screenshot(path=OUT, full_page=False)
        await browser.close()
    print("dialogs fired:", dialogs)
    print("requests to evil.example:", outside)
    print("dom:", dom)


asyncio.run(main())
