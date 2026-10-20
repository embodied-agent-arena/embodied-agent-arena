#!/usr/bin/env python3
"""Render the overview from audited exact-task executions, never list positions."""
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "public"
MODELS = ["astra", "sol", "fable", "gemini", "qwen", "qwen35-397b",
          "qwen35-27b", "opus55", "deepseek", "gemini4argon"]
SHORT_NAMES = ["GPT-6 Astra", "GPT-6 Sol", "Fable 5.1", "Gemini 3.8 Flash",
               "Qwen Max", "Qwen 397B", "Qwen 27B", "Opus 5.5", "DeepSeek-1017", "Gemini 4 Argon"]
# Fixed native tasks; the choice is explicit and independent of model order.
COLUMNS = [
    ("showcase-alfred_official_visual", "ALFRED", "Clean the soap"),
    ("showcase-alfworld_visual", "ALFWorld", "Clean & place soap"),
    ("showcase-discoveryworld", "DiscoveryWorld", "Date the artifact"),
    ("showcase-humanclaw", "HumanCLAW", "Reach the chair"),
    ("showcase-calvin", "CALVIN", "Sequence 0526"),
    ("task-video-1fd3d0c0ff", "RLBench", "Close the laptop"),
    ("showcase-maniskill", "ManiSkill", "Insert the charger"),
    ("task-video-cefac4d024", "CLIPort", "Pack matching objects"),
    ("showcase-robocasa", "RoboCasa", "Open the left drawer"),
    ("showcase-robocasa365", "RoboCasa365", "Clear the cutting board"),
    ("showcase-capx_behavior1k", "BEHAVIOR-1K", "Pack the toolbox"),
]
OUTCOMES = {
    "success": "Completed",
    "budget_exhausted": "Budget reached",
    "task_failure": "Not completed",
    "agent_failure": "Agent error",
    "runtime_failure": "Runtime error",
    "timeout": "Timed out",
}


def build():
    data = json.loads((PUB / "data/case-comparisons.json").read_text())
    groups = {g["id"]: g for g in data["groups"]}
    cells, rows, columns = [], [], []
    h = html.escape
    for gid, label, short in COLUMNS:
        g = groups[gid]
        assert len(g["runIds"]) == len(MODELS)
        columns.append({"groupId": gid, "benchmark": label, "title": short,
                        "taskId": g["taskId"], "nativeIdentity": g["nativeIdentity"],
                        "identitySha256": g["identitySha256"]})
    for row, (model, name) in enumerate(zip(MODELS, SHORT_NAMES)):
        rendered = [f'<tr><th scope="row" title="{h(data["models"][row])}">{h(name)}</th>']
        for col, (gid, benchmark, title) in enumerate(COLUMNS):
            g = groups[gid]
            r = data["runs"][g["runIds"][row]]
            assert r["modelKey"] == model
            assert r["taskId"] == g["taskId"]
            assert r["identitySha256"] == g["identitySha256"]
            detail = json.loads((PUB / r["detailUrl"]).read_text())
            assert detail["taskId"] == r["taskId"]
            assert detail["modelKey"] == model
            assert detail["reportSha256"] == r["sourceReportSha256"]
            # An animated plot of a log is not footage of the environment.
            # Keep it in the detailed record, but do not use it as a scene tile.
            has_footage = r["mediaKind"] != "Animated execution log"
            video = r["video"] if has_footage else None
            animation = r["animation"] if has_footage else None
            media = video or animation
            if media:
                assert r["sourceReportSha256"] == r["mediaReportSha256"]
                assert (PUB / media).is_file() and (PUB / r["poster"]).is_file()
            status = OUTCOMES.get(r["outcome"], "Not completed")
            cid = f"arena-{row+1}-{col+1}"
            cell = {
                "id": cid, "row": row + 1, "column": col + 1, "groupId": gid,
                "modelKey": model, "model_label": r["model"], "benchmark": benchmark,
                "display_instruction": g["question"], "taskId": r["taskId"],
                "identitySha256": r["identitySha256"], "recordId": r["recordId"],
                "reportSha256": r["sourceReportSha256"], "detailUrl": r["detailUrl"],
                "outcome": r["outcome"], "status": status, "sourceVideo": video,
                "sourceAnimation": animation, "poster": r["poster"] if media else None,
                "mediaKind": r["mediaKind"],
                "mediaReportSha256": r["mediaReportSha256"] if media else None,
                "mediaSha256": hashlib.sha256((PUB/media).read_bytes()).hexdigest() if media else None,
            }
            cells.append(cell)
            attrs = (f'class="overview-cell {"has-media" if media else "log-only"}" '
                     f'data-case="{h(gid)}" data-model="{row}" data-scene="{cid}" '
                     f'data-record="{h(r["recordId"])}" aria-describedby="scene-tooltip" '
                     f'aria-label="{h(r["model"])} · {h(benchmark)} · {h(title)}: {status}. '
                     f'{"Play recording and inspect execution" if media else "No video; inspect execution log"}"')
            if video:
                visual = (f'<video muted loop playsinline preload="none" '
                          f'poster="{h(r["poster"])}" data-preview-src="{h(video)}" '
                          f'aria-hidden="true"></video>')
            elif animation:
                visual = (f'<img loading="lazy" src="{h(r["poster"])}" '
                          f'data-preview-src="{h(animation)}" '
                          f'data-poster="{h(r["poster"])}" alt="">')
            else:
                visual = (f'<span class="overview-log-status">{status}</span>'
                          '<span class="overview-log-link">Result · No video ↗</span>')
            if media:
                visual += f'<span class="overview-media-status">{status} ↗</span>'
            rendered.append(f'<td><button type="button" {attrs}>{visual}</button></td>')
        rendered.append("</tr>")
        rows.append("".join(rendered))
    heads = ['<th scope="col" class="overview-corner">Model / Task</th>']
    for gid, label, short in COLUMNS:
        heads.append(f'<th scope="col"><button type="button" data-case="{h(gid)}">'
                     f'<strong>{h(label)}</strong><span>{h(short)}</span></button></th>')
    block = (
        '<!-- BEGIN EXACT_TASK_OVERVIEW -->\n'
        '<figure class="execution-showcase task-overview" aria-labelledby="overview-title">\n'
        '<div class="overview-toolbar"><div><h3 id="overview-title">Same Task. Ten Models.</h3>'
        '<p>One fixed task per column · Hover to play · Click for details</p></div>'
        '<div class="overview-nav"><button type="button" data-overview-scroll="-1" '
        'aria-label="Show previous task columns">←</button><button type="button" '
        'data-overview-scroll="1" aria-label="Show next task columns">→</button></div></div>\n'
        '<div class="overview-scroll" tabindex="0" role="region" aria-label="Ten-model task comparison">'
        '<table class="overview-table"><caption class="sr-only">Ten models evaluated on the same '
        'native task in each column. Results remain visible when footage is unavailable.</caption>'
        f'<thead><tr>{"".join(heads)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>\n'
        '<figcaption>Every cell links to the same task and its own execution record. '
        '“No video” means the recording is absent; the result is retained from the evaluation record.</figcaption>\n'
        '</figure>\n<!-- END EXACT_TASK_OVERVIEW -->'
    )
    index = PUB / "index.html"
    text = index.read_text()
    if "<!-- BEGIN EXACT_TASK_OVERVIEW -->" in text:
        text = re.sub(r'<!-- BEGIN EXACT_TASK_OVERVIEW -->.*?<!-- END EXACT_TASK_OVERVIEW -->',
                      lambda _: block, text, flags=re.S)
    else:
        text, n = re.subn(r'<figure class="execution-showcase".*?</figure>',
                         lambda _: block, text, count=1, flags=re.S)
        assert n == 1
    index.write_text(text)
    manifest = {"updated": "2026-10-21", "rowOrder": MODELS, "columns": columns,
                "tiles": cells, "taskCount": len(columns), "recordCount": len(cells),
                "recordingCount": sum(bool(c["mediaSha256"]) for c in cells),
                "source": "data/case-comparisons.json",
                "matching": ["taskId", "modelKey", "identitySha256", "reportSha256"]}
    (PUB / "data/report/overview-provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    app = PUB / "app.js"
    js = app.read_text()
    js, n = re.subn(r"const sceneTasks = \[.*?\];", lambda _: "const sceneTasks = " +
                   json.dumps(cells, ensure_ascii=False, separators=(",", ":")) + ";", js, count=1)
    assert n == 1
    app.write_text(js)
    print(f'{len(columns)} exact tasks × {len(MODELS)} models; {manifest["recordingCount"]} recordings; '
          f'{len(cells)-manifest["recordingCount"]} log-only cells.')


if __name__ == "__main__":
    build()
