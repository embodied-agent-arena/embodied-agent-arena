"""Check rendered manuscript pagination after compilation (requires PyMuPDF).

Usage: python scripts/check_manuscript_layout.py public/paper.pdf --output audit.json
The JSON identifies pages to inspect; visual review remains necessary.
"""
import argparse
import json
import re
from pathlib import Path

import pymupdf


def audit(path, max_gap):
    doc = pymupdf.open(path)
    pages, placements, short_tails = [], {}, []
    for number, page in enumerate(doc, 1):
        # This manuscript uses a running header and a separate page-number footer.
        content_bottom = page.rect.height - 57
        body = [b for b in page.get_text('blocks') if b[1] > 48 and b[3] < page.rect.height - 50]
        bottoms = [b[3] for b in body]
        bottoms += [i['bbox'][3] for i in page.get_image_info()
                    if i['bbox'][1] > 48 and i['bbox'][3] < page.rect.height - 50]
        gap = max(0, content_bottom - max(bottoms, default=48))
        pages.append({'page': number, 'extra_bottom_space_pt': round(gap, 1)})
        for b in body:
            m = re.match(r'(Figure|Table)\s+(\d+)\.', b[4])
            if m:
                placements[m[0][:-1]] = {'page': number, 'y': round(b[1], 1)}
        if number <= 13:
            for b in page.get_text('dict')['blocks']:
                if b['type'] or len(b['lines']) < 2 or b['bbox'][2] - b['bbox'][0] < 350:
                    continue
                line = b['lines'][-1]
                text = ''.join(s['text'] for s in line['spans']).strip()
                if (1 <= len(text.split()) <= 3 and line['spans'][0]['size'] > 10
                        and line['bbox'][1] > 48 and line['bbox'][3] < content_bottom
                        and not re.search(r'https?://|github\.com|^\(\d+\)$', text)):
                    short_tails.append({'page': number, 'text': text})
    issues = [f"Page {p['page']}: {p['extra_bottom_space_pt']} pt of extra bottom space"
              for p in pages if p['extra_bottom_space_pt'] > max_gap]
    t, f = placements.get('Table 2'), placements.get('Figure 3')
    if not t or not f or t['page'] != f['page'] or t['y'] >= f['y']:
        issues.append('Table 2 must precede Figure 3 on the same page.')
    return {'pages': len(doc), 'threshold_pt': max_gap, 'page_spacing': pages,
            'main_float_placements': placements, 'short_main_text_tails_for_review': short_tails,
            'issues': issues}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--max-gap', type=float, default=120)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.pdf, args.max_gap)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{result['pages']} pages; {len(result['issues'])} pagination issues; "
          f"{len(result['short_main_text_tails_for_review'])} short prose tails to inspect.")
    for issue in result['issues']:
        print(issue)
    raise SystemExit(bool(result['issues']))
