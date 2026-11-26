"""Proves that uploading the files in the "sample test sets" folder gives the recorded test results.

Close the demo first (its window), then run from the demo folder with the thesis venv:

    C:\\corecs3-dual-stream\\venv\\Scripts\\python.exe tools\\check_sample_folder.py

Every file in the folder is sent through the demo's real upload and run requests (the same code the browser calls),
then the four models score it live on the GPU. For a test input, each score is compared with the score the same
model gave it when the test sets were evaluated, and every frame of its evaluation batch is compared too. For a file
from "Outside our datasets" (no recorded evaluation exists) the scores are compared with the earlier saved run of the
same file, for information. The uploads this check makes are deleted at the end.
"""
import json
import os
import shutil
import sys

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
sys.path.insert(0, APP)
os.environ.setdefault("DEMO_MODE", "live")
os.environ.setdefault("DEMO_UPLOAD_BATCH", "reference")   # as run_demo_hpc.bat sets it
os.environ.setdefault("DEMO_THESIS", r"C:\corecs3-dual-stream")
import engine as E   # noqa: E402
import server as S   # noqa: E402

FOLDER = os.environ.get("DEMO_SAMPLE_FOLDER", os.path.join(E.ROOT, "sample test sets"))
TOL = 0.01
client = S.app.test_client()
made, bad, runs, rows, worst = [], 0, 0, 0, 0.0
print(f"{'file':34s} {'model':>5s} {'live':>8s} {'recorded':>9s} {'largest diff, whole batch':>26s}  check")
for sub in sorted(os.listdir(FOLDER)):
    for name in sorted(os.listdir(os.path.join(FOLDER, sub))):
        path = os.path.join(FOLDER, sub, name)
        with open(path, "rb") as f:
            up = client.post("/api/upload", data={"file": (f, name)}, content_type="multipart/form-data")
        meta = up.get_json()
        label = f"{sub[:20]}\\{name}"
        if up.status_code != 200:
            print(f"{label[:34]:34s}     -  upload refused: {meta.get('error')}")
            continue
        if not meta.get("recognized"):
            made.append(meta["id"])
        for k in (1, 2, 3, 4):
            r = client.post("/api/run", json={"sample": meta["id"], "model": k}).get_json()
            if "error" in r:
                print(f"{label[:34]:34s} {k:5d}  ERROR: {r['error']}")
                bad += 1
                continue
            if r.get("fallback") or r.get("replayed"):
                print(f"{label[:34]:34s} {k:5d}  NOT LIVE (saved run shown): {r.get('fallback')}")
                bad += 1
                continue
            runs += 1
            if meta.get("recognized"):
                fd = max([abs(a - b) for a, b in zip(r["frame_p"], r["frame_eval_p"]) if b is not None]
                         + [r.get("batch_max_diff", 0.0)])
                worst, rows = max(worst, fd), rows + r.get("batch_rows", 0)
                ev = r["eval_p"]
                note = ("DIFFERENT VERDICT" if (r["p"] >= .5) != (ev >= .5) else
                        "score differs by more than 0.01" if abs(r["p"] - ev) > TOL else "match")
                bad += note != "match"
                print(f"{label[:34]:34s} {k:5d} {r['p']:8.4f} {ev:9.4f} {fd:26.6f}  {note}  ({meta['tag']} {meta['title']})", flush=True)
            else:
                old = os.path.join(E.CACHE, "own-" + meta["tag"].lower(), f"model{k}.json")
                prev = json.load(open(old))["p"] if os.path.exists(old) else None
                info = "no earlier run" if prev is None else f"earlier run {prev:.4f}, difference {abs(prev - r['p']):.6f}"
                print(f"{label[:34]:34s} {k:5d} {r['p']:8.4f} {'-':>9s} {'':26s}  new upload ({info})", flush=True)
for sid in made:   # this check's own uploads are removed again
    shutil.rmtree(os.path.join(S.UPLOADS, sid), ignore_errors=True)
    shutil.rmtree(os.path.join(E.CACHE, sid), ignore_errors=True)
print(f"\n{runs} live runs. Test inputs: {rows} frame scores compared; largest difference from a recorded score: {worst:.6f}")
print("ALL UPLOADED TEST INPUTS MATCH THE RECORDED TEST RESULTS." if bad == 0 else
      f"{bad} run(s) do not match or did not run live. Do not present yet: send me this whole output.")
