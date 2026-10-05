#!/usr/bin/env python3
"""Render checked-in, crawlable research metadata and results from study.json."""
import csv
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'public'
study = json.loads((PUBLIC / 'data/study.json').read_text())
esc = html.escape


def replace_block(path, name, content):
    text = path.read_text()
    pattern = rf'<!-- BEGIN {name} -->.*?<!-- END {name} -->'
    replacement = f'<!-- BEGIN {name} -->\n{content}\n<!-- END {name} -->'
    text, count = re.subn(pattern, lambda _: replacement, text, flags=re.S)
    if count != 1:
        raise ValueError(f'{path}: expected one {name} block, found {count}')
    path.write_text(text)


best = [min if m['better'] == 'lower' else max for m in study['metrics']]
best = [f(float(model['scores'][i].split(' ± ')[0]) for model in study['models'])
        for i, f in enumerate(best)]
html_rows, md_rows = [], []
for model in study['models']:
    cells, md_cells = [], []
    for i, value in enumerate(model['scores']):
        parts = value.split(' ± ')
        winner = float(parts[0]) == best[i]
        primary = f'<strong>{parts[0]}</strong>' if winner else parts[0]
        cells.append('<td>' + primary + (' ± ' + parts[1] if len(parts) > 1 else '') + '</td>')
        md_cells.append(('**' + parts[0] + '**' if winner else parts[0]) + (' ± ' + parts[1] if len(parts) > 1 else ''))
    html_rows.append(f'<tr><th scope="row">{esc(model["name"])}</th>{"".join(cells)}</tr>')
    model_label = model['name'].replace('-', '&#8209;').replace(' ', '&nbsp;')
    md_rows.append('| ' + model_label + ' | ' + ' | '.join(cell.replace(' ', '&nbsp;') for cell in md_cells) + ' |')
replace_block(PUBLIC / 'index.html', 'RESULT_ROWS', '\n'.join(html_rows))
replace_block(ROOT / 'README.md', 'RESULT_TABLE', '\n'.join([
    '| Model | Rot.&nbsp;↓ | Trans.&nbsp;↓ | Spatial&nbsp;↑ | AbsRel&nbsp;↓ | Mask&nbsp;↑ | Contact&nbsp;↑ | Plan&nbsp;↑ | Manip.&nbsp;↑ |',
    '| :--- | ' + ' | '.join(['---:'] * len(study['metrics'])) + ' |',
    *md_rows]))

with (PUBLIC / 'data/model-comparison.csv').open('w', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['model', 'model_identifier', 'domain', 'metric', 'value', 'run_sd', 'unit', 'better_direction', 'protocol', 'manuscript_date', 'manuscript_version', 'source_table', 'source_url'])
    for model in study['models']:
        for metric, score in zip(study['metrics'], model['scores']):
            parts = score.split(' ± ')
            writer.writerow([model['name'], model['identifier'], metric['domain'], metric['name'], parts[0], parts[1] if len(parts)>1 else '', metric['unit'], metric['better'], study['source']['protocol'], study['manuscript_date'], study['manuscript_version'], 'Table 2', study['url']+'paper.pdf#page=7'])

author_html = ''.join(f'<span>{esc(a["name"])}<sup>{a["affiliations"]}</sup></span>' for a in study['authors'])
replace_block(PUBLIC / 'index.html', 'AUTHORS', author_html)
bibtex = '@misc{huang2026embodiedagentarena,\n'
bibtex += '  title = {' + study['title'] + '},\n'
bibtex += '  author = {' + ' and '.join(a['name'].rsplit(' ', 1)[1] + ', ' + a['name'].rsplit(' ', 1)[0] for a in study['authors']) + '},\n'
bibtex += '  year = {2026},\n  eprint = {2610.00854},\n  archivePrefix = {arXiv},\n  primaryClass = {cs.RO},\n  url = {https://arxiv.org/abs/2610.00854}\n}'
(PUBLIC / 'data/citation.bib').write_text(bibtex+'\n')
replace_block(PUBLIC / 'index.html', 'CITATION', '<pre id="bibtex">'+esc(bibtex)+'</pre>')
replace_block(ROOT / 'README.md', 'CITATION', '```bibtex\n'+bibtex+'\n```')
replace_block(PUBLIC / 'index.html', 'ABSTRACT', '<p>'+esc(study['abstract'])+'</p>')

metadata = [f'<meta name="citation_title" content="{esc(study["title"], quote=True)}">']
metadata += [f'<meta name="citation_author" content="{a["name"]}">' for a in study['authors']]
metadata += [f'<meta name="citation_publication_date" content="{study["date_published"].replace("-", "/")}">',
             f'<meta name="citation_pdf_url" content="{study["url"]}paper.pdf">',
             '<meta name="citation_doi" content="10.48550/arXiv.2610.00854">']
schema = {'@context':'https://schema.org', '@type':'ScholarlyArticle', '@id':study['url']+'#paper',
          'name':study['title'], 'headline':study['title'], 'url':study['url'], 'sameAs':'https://arxiv.org/abs/2610.00854',
          'identifier':'arXiv:2610.00854', 'datePublished':study['date_published'], 'dateModified':study['manuscript_date'],
          'version':study['manuscript_version']+' ('+study['manuscript_date']+')', 'description':study['description'],
          'abstract':study['abstract'], 'author':[{'@type':'Person','name':a['name']} for a in study['authors']],
          'encoding':{'@type':'MediaObject','contentUrl':study['url']+'paper.pdf','encodingFormat':'application/pdf'}}
metadata += ['<script type="application/ld+json">'+json.dumps(schema,ensure_ascii=False).replace('<','\\u003c')+'</script>']
replace_block(PUBLIC / 'index.html', 'SCHOLAR_METADATA', '\n'.join(metadata))

cff = ['cff-version: 1.2.0', 'message: "Please cite the paper in preferred-citation when using Embodied Agent Arena."',
       'type: software', 'title: "Embodied Agent Arena"',
       'authors:', '  - name: "Embodied Agent Arena"',
       'repository-code: "https://github.com/embodied-agent-arena/embodied-agent-arena"',
       'url: '+json.dumps(study['url']), 'preferred-citation:', '  type: article',
       '  title: '+json.dumps(study['title']), '  authors:']
for a in study['authors']:
    first, last = a['name'].rsplit(' ', 1)
    cff += ['    - family-names: '+json.dumps(last), '      given-names: '+json.dumps(first)]
cff += ['  doi: 10.48550/arXiv.2610.00854', '  url: "https://arxiv.org/abs/2610.00854"',
        '  year: 2026', '  status: preprint', '  date-released: "2026-10-01"']
(ROOT / 'CITATION.cff').write_text('\n'.join(cff)+'\n')
print('Rendered 7 models × 8 metrics, authors, abstract, citation, metadata, and CSV.')
