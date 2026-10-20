import time, sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5173/"
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True,
                          args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--use-angle=metal"])
    pg = b.new_page()
    errs = []
    pg.on("console", lambda m: errs.append(f"[{m.type}] {m.text}"[:220]))
    pg.on("pageerror", lambda e: errs.append(f"[pageerror] {str(e)[:220]}"))
    pg.goto(URL, wait_until="load")
    pg.get_by_role("button", name="Download and verify model").click()
    try:
        pg.wait_for_selector("text=Load check passed", timeout=420_000)
        print("RESULT: model loaded in dev/preview")
    except Exception:
        print("RESULT: did not reach load check within 150s")
        try:
            print("visible fail text:", pg.inner_text(".fail")[:300])
        except Exception:
            pass
    for e in errs[-8:]:
        print(" ", e)
    b.close()
