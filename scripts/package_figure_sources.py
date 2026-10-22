#!/usr/bin/env python3
"""Package current editable manuscript figures and their existing exports."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ['figure3-tradeoffs', 'figure4-geometry', 'figure5-spatial',
        'figure6-affordance', 'figure7-planning', 'figure8-manipulation']
SUPPLEMENT = ['supplement-figure6-affordance', 'supplement-figure7-planning']
ANALYSIS = ['execution-progress', 'planning-execution-profile', 'model-task-comparison']
FORMATS = ['drawio', 'svg', 'pdf', 'png']
QUALITATIVE = ['execution-cases', 'figure-cases-spatial-reasoning', 'figure-cases-affordance']


def build(figures):
    out = ROOT/'public/assets/analysis'
    contents = {}
    for stem in MAIN + SUPPLEMENT + ['cases-affordance']:
        for ext in FORMATS:
            path = figures/f'{stem}.{ext}'
            if not path.is_file():
                raise FileNotFoundError(path)
            contents[path.name] = path.read_bytes()
        if stem in MAIN:
            assert b'Gemini 4' in contents[f'{stem}.drawio'], f'{stem}: stale model list'
    for stem in ANALYSIS:
        for ext in FORMATS:
            name = f'{stem}.{ext}'
            contents[name] = (out/name).read_bytes()
    # Attribution travels with the existing logo assets embedded in the figures.
    with zipfile.ZipFile(out/'editable-figures.zip') as previous:
        for name in previous.namelist():
            if (name.startswith('model-logos/') or Path(name).stem in QUALITATIVE) and not name.endswith('/'):
                replacement = figures/name
                contents[name] = replacement.read_bytes() if replacement.is_file() else previous.read(name)
    for path in (ROOT/'public/assets/model-logos').iterdir():
        if path.is_file(): contents['model-logos/'+path.name] = path.read_bytes()
    scopes = json.loads((out/'figure-manifest.json').read_text())['scopes']
    scopes.update({stem: 'Original fine-grained diagnostics retained in the appendix.' for stem in SUPPLEMENT})
    scopes.update({stem: 'Ten-model Planning logs; selected exact-task progress comparisons include Gemini 4 Argon.' for stem in ANALYSIS})
    scopes.pop('figure-protocol-cases', None)
    scopes['figure5-spatial'] = 'All ten models: 11 task-level spatial diagnostics and six native source scores, with repeated-run means and SD.'
    scopes['figure8-manipulation'] = 'All 100 adopted CALVIN episodes: verified consecutive goals and execution-error recovery.'
    scopes['execution-progress'] = 'Exact Archaeology Dating task: native progress for Opus 5.5, Gemini 4 Argon, and DeepSeek-1017.'
    scopes['model-task-comparison'] = 'Table 2 Planning and Manipulation means; source-specific observation schedules and budgets.'
    scopes['cases-affordance'] = 'Selected qualitative polygon examples; ten-model comparisons are in the online Case Atlas.'
    manifest = {'manuscriptDate': '2026-10-22', 'modelsInMainComparison': 10,
                'scopes': scopes, 'files': [{'path': n, 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)} for n,b in sorted(contents.items())]}
    manifest['naming'] = {'full': ['Gemini 3.8 Flash', 'Gemini 4 Argon'], 'compact': ['Gemini 3.8', 'Gemini 4']}
    contents['figure-manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    contents['README.md'] = b'''# Editable manuscript and analysis figures

Updated for the 22 October 2026 author manuscript.

- Figures 3-8: ten-model comparisons. Figure 5 includes all models in its task-level heatmaps. Figure 8 examines goal completion and execution recovery in all 100 adopted CALVIN episodes.
- Figures 6-8 use rounded vertical bars; supplement-figure6/7 retain the fine-grained diagnostics in the appendix.
- Model identity colours follow the supplied logo families, with shades and line styles distinguishing models. Numerical heatmaps retain common scales. See model-logos/palette.json.
- model-task-comparison: the README comparison, using the unchanged Table 2 means.
- cases-affordance: selected qualitative polygon examples. The online Case Atlas compares all ten models on each image.
- execution-progress: exact-task Archaeology Dating progress; planning-execution-profile: ten-model Planning resources. Data and definitions for the new Spatial and CALVIN diagnostics are in public/data/spatial-recovery/.

Model names distinguish Gemini 3.8 Flash (Gemini 3.8) from Gemini 4 Argon (Gemini 4).

Open .drawio files in draw.io / diagrams.net to edit the vector objects. SVG and PDF are vector exports; PNG is a preview. Keep the Comic Sans MS typeface installed when re-exporting to preserve the manuscript typography. Photographs and model logos retain their source formats and attribution; see model-logos/.

figure-manifest.json records exact file hashes and experiment scopes. These are editable sources and exports, not a complete raw-log plotting pipeline.
'''
    # Fixed timestamps/order make repeated packaging byte-for-byte reproducible.
    with zipfile.ZipFile(out/'editable-figures.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026,10,22,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    (out/'figure-manifest.json').write_bytes(contents['figure-manifest.json'])
    print(f'Packaged {len(contents)} files; six current manuscript figures plus scoped qualitative/execution figures.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', type=Path, required=True)
    build(parser.parse_args().figures.resolve())
