#!/usr/bin/env python3
"""Build the static case index from the same task identities as the live gallery."""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'


def build():
    static = [c for c in json.loads((PUB/'data/case-studies.json').read_text()) if c.get('predictions')]
    dynamic = json.loads((PUB/'data/case-comparisons.json').read_text())
    cases = static + dynamic['groups']
    assert len({c['id'] for c in cases}) == len(cases)
    model_count = len(dynamic['models'])
    h = html.escape
    cards = []
    for c in cases:
        media = ''
        record = ''
        if c.get('runIds'):
            runs = [dynamic['runs'][rid] for rid in c['runIds']]
            assert len(runs) == model_count
            assert all(r['taskId'] == c['taskId'] for r in runs)
            # A pre-start timeout has an assignment ID but no native scene or media.
            assert all(r['identitySha256'] == c['identitySha256'] or (
                r.get('executionStarted') is False and r['identitySha256'] is None
                and not r.get('video') and not r.get('animation')
            ) for r in runs)
            preview = next((r for r in runs if r.get('poster') and (r.get('video') or r.get('animation'))), None)
            if preview:
                assert preview['sourceReportSha256'] == preview['mediaReportSha256']
                assert (PUB/preview['poster']).is_file()
                record = f' data-preview-record="{h(preview["id"])}"'
                media = f'<img src="{h(preview["poster"])}" alt="{h(c["benchmark"])} · {h(preview["model"])} recording" loading="lazy"><span class="study-play" aria-hidden="true">▶</span>'
            else:
                media = '<span class="study-result">Recorded outcomes · Open to compare ↗</span>'
            result = f'{model_count} model outcomes'
        else:
            assert len(c['predictions']) == model_count
            src = c['images'][0]
            assert (PUB/src).is_file()
            media = f'<img src="{h(src)}" alt="{h(c["benchmark"])} task input" loading="lazy">'
            result = 'Polygon predictions' if c['domainKey'] == 'affordance' else 'First-run predictions'
        title = c.get('title') or c.get('shortTask') or c['benchmark']
        description = c.get('shortTask') or c['question']
        search = ' '.join(' '.join([c['benchmark'], title, c['question'], c.get('taskId', '')]).split())
        cards.append(f'<article class="study-card" data-domain="{h(c["domainKey"])}" data-search="{h(search)}">'
                     f'<button class="study-open" data-case="{h(c["id"])}"{record} aria-label="Compare {model_count} models: {h(title)}">'
                     f'<div class="study-media">{media}</div><div class="study-card-body"><span class="study-benchmark">{h(c["benchmark"])}</span>'
                     f'<h3>{h(title)}</h3><p>{h(description)}</p><span class="study-result">{h(result)} ↗</span></div></button></article>')
    path = PUB/'case-atlas.html'
    page = path.read_text()
    start = page.index('<div class="study-grid atlas-grid">')
    end = page.index('<p class="atlas-note">Source images', start)
    page = page[:start] + '<div class="study-grid atlas-grid">\n' + '\n'.join(cards) + '\n</div>' + page[end:]
    page = re.sub(r'<p id="study-count" aria-live="polite">.*?</p>', f'<p id="study-count" aria-live="polite">{len(cases)} cases</p>', page)
    page = re.sub(r'Ten models · Five capabilities · [^<]+', f'Ten models · Five capabilities · {len(cases)} cases', page)
    page = page.replace('Browse tasks across five capabilities. Each comparison groups recordings of the same task.',
                        'Compare ten models on the same input or task, including Gemini 4 Argon. Open a case for predictions, available recordings and individual results.')
    path.write_text(page)
    print(f'Built {len(static)} static cases + {len(dynamic["groups"])} execution tasks = {len(cases)} exact-case cards.')


if __name__ == '__main__':
    build()
