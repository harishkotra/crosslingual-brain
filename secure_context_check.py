import sys
from playwright.sync_api import sync_playwright

# Usage: python secure_context_check.py http://127.0.0.1:4173/ http://<lan-ip>:4173/
urls = sys.argv[1:] or ["http://127.0.0.1:4173/"]
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    for url in urls:
        pg = b.new_page()
        pg.goto(url, wait_until="load")
        print(url)
        print("  isSecureContext:", pg.evaluate("window.isSecureContext"))
        print("  navigator.gpu:", pg.evaluate("!!navigator.gpu"))
        print("  title:", pg.title())
        pg.close()
    b.close()
