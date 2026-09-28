"""Record the static shared-clock player. Requires Playwright + Chromium + ffmpeg.

First serve site/ locally, then run this script. It reads existing timestamped
inference records; it does not generate tokens or change recorded durations.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765/')
    parser.add_argument('--output', default='site/assets/progdraft-synchronized.mp4')
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='progdraft-video-') as directory:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
            context = browser.new_context(viewport=dict(width=1440, height=920),
                                          record_video_dir=directory,
                                          record_video_size=dict(width=1440, height=920))
            page = context.new_page()
            page.goto(args.url.rstrip('/')+'/?record=1', wait_until='networkidle')
            page.wait_for_function('window.progdraftReplay')
            page.evaluate('progdraftReplay.setRate(0.05)')
            page.wait_for_timeout(1600)
            page.get_by_role('button', name='▶ Play together').click()
            page.wait_for_function('!progdraftReplay.playing && progdraftReplay.time > 0', timeout=60000)
            assert page.locator('.exact.verified').count() == 2
            page.wait_for_timeout(2500)
            data = page.evaluate('progdraftReplay.data')
            video = page.video
            context.close()
            webm = Path(directory)/'comparison.webm'
            video.save_as(str(webm))
            browser.close()
        subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(webm),
                        '-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',
                        '-movflags','+faststart','-an',str(output)], check=True)
    # A representative frame shows both arms in flight; no synthetic UI mockup.
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport=dict(width=1440, height=920))
        page.goto(args.url.rstrip('/')+'/?record=1', wait_until='networkidle')
        page.wait_for_function('window.progdraftReplay')
        page.evaluate('progdraftReplay.setRate(0.05)')
        page.evaluate('(t)=>progdraftReplay.seek(t)', data['methods']['ours']['rounds'][8]['observed_decode_s'])
        page.screenshot(path=str(output.parent/'comparison-poster.jpg'), quality=92, type='jpeg')
        browser.close()
    print('Saved', output, 'with original timestamps and uniform 0.05x playback.')


if __name__ == '__main__':
    main()
