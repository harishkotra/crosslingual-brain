from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    for w, h, name in [(390, 844, "iphone"), (820, 1180, "ipad")]:
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        pg.goto("http://127.0.0.1:4173/", wait_until="load")
        sw = pg.evaluate("document.scrollingElement.scrollWidth")
        iw = pg.evaluate("window.innerWidth")
        h1 = pg.evaluate("getComputedStyle(document.querySelector('h1')).fontSize")
        print(f"{name} {w}x{h}: scrollWidth {sw} vs innerWidth {iw} | overflow {'YES' if sw > iw + 1 else 'no'} | h1 {h1}")
        pg.screenshot(path=f"shots/mobile-{name}.png", full_page=True)
        pg.close()
    b.close()
