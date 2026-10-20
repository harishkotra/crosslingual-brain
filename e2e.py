import time, re
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:4173/"
OUT = []

def log(*a):
    line = " ".join(str(x) for x in a)
    OUT.append(line)
    print(line, flush=True)

CHECKS = [
    ("te", "గది అద్దె రోజుకి ఎంత", "5000"),
    ("te", "కంటిశుక్లం ఆపరేషన్ కి ఎంత కవర్ ఉంటుంది", "40,000"),
    ("hi", "पहले से मौजूद बीमारी के लिए कितने महीने इंतज़ार करना पड़ता है", "36months"),
    ("roman", "motiyabind operation ki kitna milta hai", "40,000"),
    ("roman", "room rent limit per day entha", "5000"),
    ("en", "what is covered after I get discharged from hospital", "60 days"),
]

norm = lambda s: re.sub(r"\s+", "", s).lower()

def main():
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True,
                              args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--use-angle=metal"])
        page = b.new_page(viewport={"width": 1180, "height": 1000})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)[:160]))

        page.goto(URL, wait_until="load")
        page.screenshot(path="shots/30-landing.png", full_page=True)

        t0 = time.time()
        page.get_by_role("button", name="Download and verify model").click()
        page.wait_for_selector("text=Load check passed", timeout=1_200_000)
        log(f"MODEL READY {time.time()-t0:.0f}s | {page.inner_text('.cardHead').replace(chr(10),' | ')}")

        t1 = time.time()
        page.get_by_role("button", name="Health policy (29 pages)").click()
        page.wait_for_function(
            "() => document.body.innerText.includes('health_policy.pdf') && !document.body.innerText.includes('indexing')",
            timeout=1_500_000)
        log(f"INDEXED {time.time()-t1:.0f}s | {page.inner_text('.card:nth-of-type(2) .cardHead').replace(chr(10),' | ')}")
        page.screenshot(path="shots/31-indexed.png", full_page=True)

        any_of_5 = 0
        rank1 = 0
        for lang, q, fact in CHECKS:
            page.fill(".ask", q)
            page.get_by_role("button", name="Ask").click()
            page.wait_for_selector(".results li", timeout=300_000)
            page.wait_for_timeout(200)
            n = page.locator(".results li").count()
            pages, texts = [], []
            for i in range(n):
                li = page.locator(".results li").nth(i)
                head = li.locator(".resHead").inner_text().replace("\n", " ")
                pages.append(head)
                li.locator(".tiny").click()
                page.wait_for_timeout(60)
                texts.append(li.locator(".snippet").inner_text())
            hits = [i for i, t in enumerate(texts) if norm(fact) in norm(t)]
            ok = bool(hits)
            any_of_5 += ok
            rank1 += (0 in hits)
            log(f"{'PASS' if ok else 'FAIL'} [{lang}] {q[:42]}")
            log(f"     results {n} | fact on result {[h+1 for h in hits] or 'none'} | {page.inner_text('.card:nth-of-type(3) .cardHead').replace(chr(10),' ')}")
            log(f"     top: {pages[0]}")
            page.screenshot(path=f"shots/32-{lang}.png", full_page=True)

        log(f"FACT ON ONE OF TOP-5 PAGES {any_of_5}/{len(CHECKS)}   FACT ON RANK-1 PAGE {rank1}/{len(CHECKS)}")
        log("pageerrors:", errs[:3])
        b.close()
    open("e2e_final.log", "w").write("\n".join(OUT))

main()
