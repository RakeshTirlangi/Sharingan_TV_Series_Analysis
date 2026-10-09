"""Capture the README screenshots and GIFs from a running Sharingan site.

    sharingan app &                      # the site must be serving on :8000 with all stages computed
    python scripts/capture_media.py      # writes docs/media/*.png and *.gif

Requires `pip install playwright pillow && python -m playwright install chromium`.
GIFs are assembled from viewport frames with Pillow (no ffmpeg needed). The README's first image,
hero-live.svg, is a screenshot with the real eyes overlaid and animated in SVG, so it keeps spinning
even where GitHub pauses GIFs for viewers who prefer reduced motion.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

URL = "http://127.0.0.1:8000/"
OUT = Path(__file__).resolve().parents[1] / "docs" / "media"
W, H = 1280, 800
GIF_WIDTH = 960
HIDE_NAV = "nav{display:none!important}"


def frame(page: Page) -> Image.Image:
    return Image.open(io.BytesIO(page.screenshot())).convert("RGB")


def record(page: Page, seconds: float, fps: float = 8) -> list[Image.Image]:
    frames, step = [], 1 / fps
    end = time.time() + seconds
    while time.time() < end:
        t = time.time()
        frames.append(frame(page))
        time.sleep(max(0.0, step - (time.time() - t)))
    return frames


def save_gif(frames: list[Image.Image], name: str, fps: float = 8, hold_last: float = 1.5) -> None:
    scale = GIF_WIDTH / frames[0].width
    size = (GIF_WIDTH, round(frames[0].height * scale))
    small = [f.resize(size, Image.LANCZOS) for f in frames]
    if hold_last:
        small += [small[-1]] * int(hold_last * fps)
    # One shared palette, built from a mosaic of frames across the whole clip, keeps every
    # colour that appears anywhere (e.g. all community colours) and stays stable frame to frame.
    picks = [small[i] for i in range(0, len(small), max(1, len(small) // 8))][:8]
    mosaic = Image.new("RGB", (size[0], size[1] * len(picks)))
    for k, f in enumerate(picks):
        mosaic.paste(f, (0, k * size[1]))
    palette = mosaic.quantize(colors=224, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in small]
    path = OUT / f"{name}.gif"
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=int(1000 / fps),
                      loop=0, optimize=True, disposal=1)
    print(f"{path.name}: {len(quantized)} frames, {path.stat().st_size / 1e6:.1f} MB")


def hero_live_svg(page: Page) -> None:
    """Landing page as an animated SVG: a screenshot with the real eyes overlaid, spinning via SMIL.

    GitHub pauses GIFs for viewers whose system asks for reduced motion, but renders SVG
    animations in <img> normally, so this keeps the README's first image alive for everyone.
    """
    import base64
    import re

    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(2500)
    page.evaluate("document.querySelectorAll('svg').forEach(s => { s.pauseAnimations(); s.setCurrentTime(0) })")
    eyes = page.evaluate("""() => [...document.querySelectorAll('svg')]
        .filter(s => s.querySelector('animateTransform'))
        .map(s => { const r = s.getBoundingClientRect();
                    return {x: r.x, y: r.y, w: r.width, h: r.height, html: s.outerHTML}; })
        .filter(e => e.y + e.h > 0 && e.y < innerHeight)""")
    png = base64.b64encode(page.screenshot(type="png")).decode()
    page.evaluate("document.querySelectorAll('svg').forEach(s => s.unpauseAnimations())")
    layers = []
    for i, e in enumerate(eyes):
        inner = re.sub(r"^<svg[^>]*>|</svg>$", "", e["html"])
        inner = re.sub(r'id="(\w+)"', lambda m: f'id="{m[1]}{i}"', inner)        # unique gradient ids
        inner = re.sub(r"url\(#(\w+)\)", lambda m: f"url(#{m[1]}{i})", inner)
        layers.append(f'<svg x="{e["x"]:.1f}" y="{e["y"]:.1f}" width="{e["w"]:.1f}" height="{e["h"]:.1f}" '
                      f'viewBox="0 0 100 100">{inner}</svg>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<title>Sharingan landing page</title>'
           f'<image width="{W}" height="{H}" href="data:image/png;base64,{png}"/>{"".join(layers)}</svg>')
    path = OUT / "hero-live.svg"
    path.write_text(svg)
    print(f"{path.name}: {len(eyes)} animated eyes, {path.stat().st_size / 1e6:.1f} MB")


def focus_section(page: Page, selector: str, offset: int = 0) -> None:
    page.evaluate(f"""() => {{
        const el = document.querySelector({selector!r});
        window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - {offset});
    }}""")
    page.wait_for_timeout(700)


def focus_text(page: Page, text: str, offset: int = 24) -> None:
    """Scroll so the card whose heading is ``text`` starts ``offset`` px below the viewport top."""
    box = page.get_by_text(text, exact=True).first.bounding_box()
    page.evaluate(f"window.scrollBy(0, {box['y'] - offset})")
    page.wait_for_timeout(700)


def shot(page: Page, name: str) -> None:
    path = OUT / f"{name}.png"
    page.screenshot(path=str(path))
    print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()

        # ---------------------------------------------------------------- hero
        page = browser.new_page(viewport={"width": W, "height": H})
        page.goto(URL, wait_until="domcontentloaded")
        hero = record(page, 3.2, fps=10)               # entrance animation + count-up + spinning eye
        save_gif(hero, "hero", fps=10)
        page.wait_for_load_state("networkidle")
        shot(page, "hero")
        hero_live_svg(page)

        page.add_style_tag(content=HIDE_NAV)

        # -------------------------------------------------------------- themes
        focus_section(page, "#themes", offset=-150)     # skip the section header
        page.wait_for_timeout(1200)
        shot(page, "themes")
        focus_text(page, "Theme intensity across 220 episodes", offset=48)
        frames = record(page, 0.8)
        for chip in ["sacrifice", "betrayal", "loneliness", "revenge"]:
            page.locator("#themes button", has_text=chip).first.click()
            frames += record(page, 1.1)
        save_gif(frames, "themes")

        # ------------------------------------------------------------- network
        focus_section(page, "#network", offset=-330)
        page.wait_for_timeout(6500)                     # let the force simulation settle
        shot(page, "network")
        # Taller viewport: arc pills, graph and sidebar fit, so no click ever scrolls the page.
        page.set_viewport_size({"width": W, "height": 1200})
        focus_section(page, "#network", offset=-318)   # arc pills + graph + sidebar all in frame
        page.wait_for_timeout(1500)
        frames = record(page, 0.8)
        for name in ["Orochimaru", "Hinata", "Rock Lee", "Naruto"]:
            # dispatch_event clicks without Playwright's scroll-into-view, so the frame never moves
            page.locator("#network ol button", has_text=name).first.dispatch_event("click")
            frames += record(page, 1.6)
        page.locator("#network button", has_text="Chūnin Exams").first.dispatch_event("click")
        frames += record(page, 5.0)
        save_gif(frames, "network")
        page.set_viewport_size({"width": W, "height": H})

        # --------------------------------------------------------------- jutsu
        page.set_viewport_size({"width": W, "height": 900})
        focus_text(page, "Classify a technique", offset=28)
        page.get_by_role("button", name="Classify").dispatch_event("click")
        page.get_by_text("Illusion: hijacks").wait_for(timeout=60000)
        page.wait_for_timeout(1200)
        shot(page, "jutsu")
        frames = []
        for example in ["Example 2", "Example 3", "Example 1"]:
            page.get_by_role("button", name=example).dispatch_event("click")
            frames += record(page, 0.6)
            page.get_by_role("button", name="Classify").dispatch_event("click")
            frames += record(page, 2.0)
        save_gif(frames, "jutsu")
        page.set_viewport_size({"width": W, "height": H})

        # ---------------------------------------------------------------- chat
        focus_section(page, "#chat", offset=-330)
        page.locator("#chat button", has_text="Kakashi Hatake").first.click()
        page.wait_for_timeout(600)
        frames = record(page, 0.8, fps=6)
        page.get_by_role("button", name="Why are you always late?").click()
        start = time.time()
        while time.time() - start < 180:                # stream until the send button returns
            frames.append(frame(page))
            if page.locator("form button[type=submit]").count():
                break
            page.wait_for_timeout(330)
        page.get_by_text("memories used").first.click()
        frames += record(page, 1.5, fps=6)
        # Long CPU generations: keep at most ~60 frames so the GIF stays small.
        if len(frames) > 60:
            keep = sorted(set(range(0, len(frames) - 10, max(1, (len(frames) - 10) // 50))) | set(range(len(frames) - 10, len(frames))))
            frames = [frames[i] for i in keep]
        save_gif(frames, "chat", fps=6, hold_last=2.5)
        shot(page, "chat")

        browser.close()


if __name__ == "__main__":
    sys.exit(main())
