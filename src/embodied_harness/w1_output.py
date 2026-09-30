"""W1-local presentation and model-authored Python extraction (no answer repair)."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import json
from pathlib import Path
import re

FORMAT_GUIDE = (
    'Return only executable Python, without explanatory prose or Markdown. '
    'The requested answer schema describes the dictionary passed to p.submit(answer); '
    'do not return a bare JSON object. Use Python True, False and None, not true, false or null. '
    'First call print(p.observe()) and yield to receive the images, then submit in a later cell. '
    'Observation keys are task_summary, transport_images and remaining_budget.'
)
FORMAT_ERROR = (
    'W1 output format: return one unambiguous executable Python cell. '
    'Import primitive_api as p and call p.submit(answer) after receiving images. '
    'Do not return JSON, explanation text or multiple alternative code blocks. '
    'Use Python True/False/None.'
)
_FENCE = re.compile(r'^```([^\n`]*)\n(.*?)^```[ \t]*(?:\n|$)', re.MULTILINE | re.DOTALL)
_IMPORT = re.compile(r'(?m)^import primitive_api(?: as [A-Za-z_]\w*)?[ \t]*$')


def public_question(question: str) -> str:
    """Only change response-format wording; the private source catalog stays intact."""
    text = question.replace('只输出', '传给 p.submit 的字典要求：')
    text = text.replace('JSON 对象', 'Python 字典').replace('JSON', 'Python 字典')
    for source, target in [('true', 'True'), ('false', 'False'), ('null', 'None')]:
        text = re.sub(r'\b' + source + r'\b', target, text)
    return text + '\n\n' + FORMAT_GUIDE


def _tree(code):
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError):
        return None


def _bare_answer(tree):
    return (len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, (ast.Dict, ast.List, ast.Constant)))


def extract_cell(text: str) -> dict:
    """Select existing Python verbatim; never synthesize a submission or fix values.

    Invalid/ambiguous responses consume a normal code turn with an explicit
    formatting error. No extra completion request or budget is introduced.
    """
    raw = text if isinstance(text, str) else ''
    stripped = raw.strip()
    def reject(reason):
        return dict(code='raise ValueError(' + repr(FORMAT_ERROR) + ')',
                    selection='format_error', reason=reason, synthetic_feedback=True)
    tree = _tree(stripped)
    if tree is not None and tree.body:
        if _bare_answer(tree):
            return reject('bare_answer')
        return dict(code=stripped, selection='whole_python', synthetic_feedback=False)
    candidates = []
    fences = list(_FENCE.finditer(stripped))
    for fence in fences:
        if fence.group(1).strip().lower() in {'python', 'py', ''}:
            code = fence.group(2).strip()
            tree = _tree(code)
            if tree is None or not tree.body or _bare_answer(tree):
                return reject('invalid_python_block')
            candidates.append((code, 'python_fence'))
    imports = [match for match in _IMPORT.finditer(stripped)
               if not any(f.start() <= match.start() < f.end() for f in fences)]
    if len(imports) > 1:
        return reject('ambiguous_python')
    for match in imports:
        code = stripped[match.start():].strip()
        if _tree(code) is not None:
            candidates.append((code, 'python_suffix'))
    if len(candidates) != 1:
        return reject('ambiguous_python' if candidates else 'no_python')
    code, selection = candidates[0]
    return dict(code=code, selection=selection, synthetic_feedback=False)


@contextmanager
def author_output(extraction_log: Path):
    """Install only inside the dedicated W1 runner process, then restore exactly.

    The shared author and other benchmark runners remain byte-for-byte unchanged.
    """
    import codex_cell_author
    previous = codex_cell_author.usable_python_cell
    def extract(text, extract=None):
        result = extract_cell(text)
        with extraction_log.open('a') as stream:
            stream.write(json.dumps(dict(raw_reply=text, **result), ensure_ascii=False) + '\n')
        return result['code']
    codex_cell_author.usable_python_cell = extract
    try:
        yield
    finally:
        codex_cell_author.usable_python_cell = previous
