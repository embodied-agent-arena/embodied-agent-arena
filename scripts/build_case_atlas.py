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
    featured = ['showcase-calvin', 'W5-w5_umd-03', 'W2-w2_mindcube-01',
                'showcase-alfworld_visual', 'W1-w1_vlm_motion_real-01', 'showcase-robocasa']
    cases.sort(key=lambda c: featured.index(c['id']) if c['id'] in featured else len(featured))
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
            preview = dynamic['runs'][c['sceneContext']['sourceRunId']]
            if preview:
                assert preview['sourceReportSha256'] == preview['mediaReportSha256']
                assert (PUB/preview['poster']).is_file()
                record = f' data-preview-record="{h(preview["id"])}"'
                label = f'{preview["model"]} · {"Action log" if preview["mediaKind"]=="Animated execution log" else "Preview"}'
                if preview.get('video'):
                    media = f'<video muted loop playsinline preload="none" poster="{h(preview["poster"])}" data-video="{h(preview["video"])}" aria-label="{h(label)}"></video>'
                else:
                    media = f'<img src="{h(preview["poster"])}" data-animation="{h(preview["animation"])}" data-poster="{h(preview["poster"])}" alt="{h(label)}" loading="lazy">'
                media += f'<span class="atlas-preview-label">{h(label)}</span>'
            else:
                media = '<span class="study-result">Recorded outcomes · Open to compare ↗</span>'
            result = f'{model_count} model outcomes'
        else:
            assert len(c['predictions']) == model_count
            sources = c['images'][:2]
            assert sources and all((PUB/src).is_file() for src in sources)
            media = ''.join(f'<img src="{h(src)}" alt="{h(c["benchmark"])} task input {i+1}" loading="lazy">' for i,src in enumerate(sources))
            result = 'Polygon predictions' if c['domainKey'] == 'affordance' else 'First-run predictions'
        title = c.get('title') or c.get('shortTask') or c['benchmark']
        description = c.get('shortTask') or c['question']
        search = ' '.join(' '.join([c['benchmark'], title, c['question'], c.get('taskId', '')]).split())
        cards.append(f'<article class="study-card" data-domain="{h(c["domainKey"])}" data-search="{h(search)}">'
                     f'<button class="study-open" data-case="{h(c["id"])}"{record} aria-label="Compare {model_count} models: {h(title)}">'
                     f'<div class="study-media{" study-pair" if not c.get("runIds") and len(c["images"])>1 else ""}">{media}</div>'
                     f'<div class="study-card-body"><span class="study-benchmark">{h(c["benchmark"])}<span aria-hidden="true">↗</span></span>'
                     f'<h3>{h(title)}</h3></div></button></article>')
    path = PUB/'case-atlas.html'
    page = path.read_text()
    start = page.index('<div class="study-grid atlas-grid">')
    end = page.index('<p class="atlas-note">Source images', start)
    page = page[:start] + '<div class="study-grid atlas-grid">\n' + '\n'.join(cards) + '\n</div>' + page[end:]
    intro_start = page.index('<div class="atlas-heading">')
    intro_end = page.index('<div class="study-grid atlas-grid">', intro_start)
    filters = ''.join(f'<button data-study-filter="{key}" aria-pressed="{str(key=="all").lower()}">{label}</button>' for key,label in [
        ('all','All'),('geometry','Geometry'),('spatial','Spatial'),('affordance','Affordance'),('planning','Planning'),('manipulation','Manipulation')])
    intro = (f'<div class="atlas-heading"><div><h1>Case Atlas</h1><p>Same task. Ten models.</p></div>'
             '<button class="atlas-motion" data-atlas-motion aria-pressed="false">Pause previews</button></div>'
             '<div class="atlas-tools"><label class="study-search"><span class="sr-only">Find a case</span>'
             '<input type="search" id="study-search" placeholder="Search a task, object or benchmark…"></label>'
             f'<div class="study-filters" role="group" aria-label="Filter cases by capability">{filters}</div></div>'
             f'<p id="study-count" aria-live="polite">{len(cases)} cases · Select a card to compare</p>')
    page = page[:intro_start]+intro+page[intro_end:]
    page = re.sub(r'<p id="study-count" aria-live="polite">.*?</p>', f'<p id="study-count" aria-live="polite">{len(cases)} cases</p>', page)
    page = re.sub(r'Ten models · Five capabilities · [^<]+', f'Ten models · Five capabilities · {len(cases)} cases', page)
    page = page.replace('Browse tasks across five capabilities. Each comparison groups recordings of the same task.',
                        'Compare ten models on the same input or task, including Gemini 4 Argon. Open a case for predictions, available recordings and individual results.')
    path.write_text(page)
    print(f'Built {len(static)} static cases + {len(dynamic["groups"])} execution tasks = {len(cases)} exact-case cards.')


if __name__ == '__main__':
    build()
