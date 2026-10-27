"""Serving-only progress/preview page; never opens evaluation references."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import read_json
from benchmark.report import document, esc, image, link, number, table
from paper_reconstruction_stage4.outputs import read_result

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--follow", action="store_true")
args = parser.parse_args()
root = args.run.resolve()
request = read_json(root / "request.json")
cache = {}
while True:
    progress = read_json(root / "progress.json")
    rows, previews = [], []
    done = 0
    for cid in request["case_ids"]:
        case = progress["cases"].get(cid, {})
        status = case.get("status", "pending")
        done += int(status == "predicted")
        path = root / "cases" / cid / "stage4"
        if cid not in cache and (path / "manifest.json").is_file():
            cache[cid] = read_result(path)
        result = cache.get(cid)
        components = [len(c) for c in result["layouts"][0]["components"]] if result else None
        attempts = case.get("attempts", [])
        times = {stage: next((a.get("elapsed_seconds") for a in reversed(attempts)
                              if a["stage"] == stage and a.get("returncode") == 0), None)
                 for stage in ("stage2", "stage3", "pool", "stage4")}
        logs = root / "cases" / cid / "logs"
        rows.append((esc(cid), esc(status), esc(case.get("active_stage")), esc(components),
                     *[number(times[s]) for s in times], link(logs, root, "Stage logs")))
        if result:
            previews.append(image(path / "preview.png", root, cid + " — primary, construction review pending"))
    content = f'<h1>Live reconstruction benchmark: {done}/{len(request["case_ids"])} predictions</h1>'
    content += '<p>Phase: '+esc(progress["phase"])+'. Updated '+esc(datetime.now(timezone.utc).isoformat())+'</p>'
    content += '<p>Generator cuts; Stage 1 inference bypassed. This page reads serving outputs only. Connectivity is not a correctness verdict. Full offline diagnoses follow the prediction phase.</p>'
    content += table(["Case", "Status", "Active stage", "Components", "Stage 2 seconds", "Stage 3 seconds", "Pool seconds", "Stage 4 seconds", "Files"], rows)
    content += '<div class="visuals">'+''.join(previews)+'</div>'
    content += '<p>'+link(root / "report/index.html", root, "Full diagnosis after completion")+'</p>'
    page = document("Live reconstruction benchmark", content)
    (root / "live.html").write_text(page, encoding="utf-8")
    if not args.follow or progress["phase"] in {"finished", "diagnostics_failed", "interrupted"}:
        break
    time.sleep(15)
