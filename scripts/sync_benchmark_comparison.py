"""Publish Table 1 directly from the released manuscript source archive."""
from html import escape
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'public'

LINKS = {
    'OpenEQA': 'https://github.com/facebookresearch/open-eqa',
    'VLABench': 'https://github.com/OpenMOSS/VLABench',
    'BEAR': 'https://github.com/yqi19/BEAR-official',
    'EmbodiedBench': 'https://github.com/EmbodiedBench/EmbodiedBench',
    'EmbodiedEval': 'https://github.com/thunlp/EmbodiedEval',
    'Embodied Arena': 'https://github.com/EmbodiedArena/embodied-arena',
    'CaP-X': 'https://github.com/capgym/cap-x',
    'WorldLines': 'https://arxiv.org/html/2606.18847v2',
    'Hard/Easy Vision': 'https://mbzuai-oryx.github.io/frontier-vision/',
    'Embodied Agent Arena': '#arena',
}


def manuscript_rows():
    with zipfile.ZipFile(PUBLIC / 'author-source.zip') as archive:
        source = archive.read('sections/table_benchmark_comparison.tex').decode()
    rows = []
    for line in source.splitlines():
        if '&' not in line or r'\benchyes' not in line:
            continue
        cells = line.removesuffix(r'\\').split('&')
        name = re.sub(r'~\\citeyearpar\{[^}]+\}', '', cells[0]).strip()
        name = re.sub(r'\\textbf\{([^}]+)\}', r'\1', name)
        scale = re.sub(r'\\textbf\{([^}]+)\}', r'\1', cells[1]).strip()
        coverage = ['yes*' if '$^{*}$' in cell else 'yes' if r'\benchyes' in cell else 'no'
                    for cell in cells[2:]]
        assert len(coverage) == 7, name
        rows.append((name, scale, coverage))
    assert len(rows) == 10
    return rows


def render():
    lines = ['<table class="benchmark-table" id="benchmark-coverage" aria-label="Evaluation coverage, diagnostic analysis, and sample scale; manuscript Table 1" aria-describedby="benchmark-key">',
             '<colgroup><col class="benchmark-name-col"><col class="benchmark-scale-col"><col span="7"></colgroup>',
             '<thead><tr class="benchmark-groups"><th colspan="2" scope="colgroup">Benchmark &amp; scale</th><th colspan="5" scope="colgroup">Evaluation coverage</th><th colspan="2" scope="colgroup">Diagnostic analysis</th></tr>',
             '<tr><th scope="col">Benchmark</th><th scope="col">Instances</th><th scope="col">Fixed<br>views</th><th scope="col">Region<br>grounding</th><th scope="col">Navigation</th><th scope="col">Arm /<br>gripper</th><th scope="col">Cross-session<br>memory</th><th scope="col">Calibration<br>/ pose</th><th scope="col">Task<br>breakdowns</th></tr></thead><tbody>']
    for name, scale, coverage in manuscript_rows():
        cls = ' class="benchmark-own"' if name == 'Embodied Agent Arena' else ''
        row = f'<tr{cls}><th scope="row"><a href="{LINKS[name]}">{escape(name)}</a></th><td data-scale="{escape(scale)}">{escape(scale) if scale != "NR" else "Not reported"}</td>'
        for column, item in enumerate(coverage):
            label = 'Evaluated; separate BEAR-Agent online validation' if name == 'BEAR' and column == 3 else 'Evaluated' if item.startswith('yes') else 'Not evaluated'
            symbol = '<span class="benchmark-yes" aria-hidden="true">✓</span>' if item.startswith('yes') else '<span class="benchmark-no" aria-hidden="true">—</span>'
            star = '<sup aria-hidden="true">*</sup>' if item == 'yes*' else ''
            row += f'<td data-coverage="{item}" aria-label="{label}">{symbol}{star}</td>'
        lines.append(row + '</tr>')
    lines.append('</tbody></table>')
    return '\n'.join('      ' + line for line in lines)


if __name__ == '__main__':
    page = PUBLIC / 'index.html'
    html = page.read_text()
    start, end = '<!-- BEGIN BENCHMARK COMPARISON -->', '<!-- END BENCHMARK COMPARISON -->'
    before, rest = html.split(start)
    _, after = rest.split(end)
    page.write_text(before + start + '\n' + render() + '\n      ' + end + after)
    print('Updated benchmark table: 10 benchmarks, 70 coverage cells, all instance counts from manuscript Table 1.')
