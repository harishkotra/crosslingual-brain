import time
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:4173/"
log = []

def p(*a):
    line = " ".join(str(x) for x in a)
    log.append(line)
    print(line, flush=True)

def main():
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True,
                               args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--use-angle=metal"])
        page = b.new_page()
        reqs = []
        page.on("request", lambda r: reqs.append((r.method, r.url)))

        page.goto(URL, wait_until="load")
        t0 = time.time()
        page.get_by_role("button", name="Download and verify model").click()
        page.wait_for_selector("text=Load check passed", timeout=1_200_000)
        p(f"model ready {time.time()-t0:.0f}s")

        origins = {}
        for m, u in reqs:
            o = u.split("/")[2] if "://" in u else u
            origins[o] = origins.get(o, 0) + 1
        p("origins during load:", origins)

        page.get_by_role("button", name="Discharge summary (3 pages)").click()
        page.wait_for_function("() => document.body.innerText.includes('discharge_summary.pdf') && !document.body.innerText.includes('indexing')", timeout=600_000)
        p("indexed sample")

        reqs.clear()
        page.fill(".ask", "గది అద్దె రోజుకి ఎంత")
        page.get_by_role("button", name="Ask").click()
        page.wait_for_selector(".results li", timeout=300_000)
        page.wait_for_timeout(700)
        p(f"requests during one query: {len(reqs)}", reqs[:5])
        p("results:", page.inner_text(".results li:first-child .resHead").replace("\n", " "))
        b.close()
    open("netcheck.log", "w").write("\n".join(log))

main()
