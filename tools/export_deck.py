"""Turn the claude.ai Slides files into a standalone HTML deck: python export_deck.py <artifact-files dir> <images dir> <out dir>"""
import json, re, shutil, sys
from pathlib import Path

src, images, out = map(Path, sys.argv[1:4])
blobs = {'c6ca2ae1f564e99786549ca4aac85136': 'froc.png',
         'a83cfd29e8bb5d801f860b10115d1513': 'false_alarms.png',
         '4723acdf71ac0576cb2d25fa8225ea6f': 'missed.png'}

deck = json.loads((src / 'project/deck.json').read_text(encoding='utf-8'))
(out / 'img').mkdir(parents=True, exist_ok=True)
for name in blobs.values():
    shutil.copy(images / name, out / 'img' / name)

slides = []
for i, sid in enumerate(deck['order'], 1):
    html = (src / f'project/slides/{sid}.html').read_text(encoding='utf-8').strip()
    for blob, name in blobs.items():
        html = html.replace(f'/_blob/{blob}', f'img/{name}')
    html = re.sub(r'<x-connector style="width:(\d+)px;color:([^"]+)"></x-connector>',
                  r'<div style="width:\1px;flex:none;text-align:center;font-size:56px;color:\2">&#8594;</div>', html)
    notes = re.search(r'<aside>(.*?)</aside>', html, re.S)
    html = re.sub(r'<aside>.*?</aside>', '', html, flags=re.S)
    slides.append(f'<div class="frame"><div class="fit">{html}</div></div>\n'
                  f'<details class="notes"><summary>Slide {i} speaker notes</summary><p>{notes.group(1) if notes else ""}</p></details>')

fonts = '&'.join(f['href'].split('?', 1)[1].replace('&display=swap', '') for f in deck['faces'].values())
page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{deck['title']}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?{fonts}&display=swap">
<style>
body {{ margin: 0; background: #2b2f36; font-family: 'IBM Plex Sans', Arial, sans-serif; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 24px 16px 64px; }}
.frame {{ position: relative; width: 100%; aspect-ratio: 16 / 9; overflow: hidden; margin-top: 24px; box-shadow: 0 4px 24px rgba(0,0,0,.4); }}
.fit {{ position: absolute; top: 0; left: 0; width: 1920px; height: 1080px; transform-origin: 0 0; }}
section {{ position: relative; box-sizing: border-box; width: 1920px; height: 1080px; overflow: hidden; }}
section * {{ margin: 0; box-sizing: border-box; }}
section ol {{ padding-left: 1.2em; }}
section table {{ border-collapse: collapse; width: 100%; }}
section th, section td {{ padding: .35em .6em; text-align: left; vertical-align: top; border-bottom: 1px solid #CFC9BC; }}
section th {{ font-weight: 600; border-bottom: 2px solid #14233A; }}
.notes {{ color: #D5DAE1; font-size: 15px; line-height: 1.5; margin-top: 8px; }}
.notes summary {{ cursor: pointer; color: #9FB0C3; }}
@page {{ size: 1920px 1080px; margin: 0; }}
@media print {{
  body {{ background: none; }} main {{ max-width: none; padding: 0; }} .notes {{ display: none; }}
  .frame {{ width: 1920px; height: 1080px; margin: 0; box-shadow: none; break-after: page; }}
  .fit {{ transform: none !important; }}
}}
</style></head>
<body><main>
{chr(10).join(slides)}
</main>
<script>
function fit() {{ document.querySelectorAll('.frame').forEach(f => f.firstElementChild.style.transform = `scale(${{f.clientWidth / 1920}})`); }}
addEventListener('resize', fit); fit();
</script>
</body></html>
'''
(out / 'index.html').write_text(page, encoding='utf-8')
print('slides:', len(slides))
