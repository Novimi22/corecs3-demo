"""Runs every input in the list through the four models once, saves the runs (cache/), and checks every score
against the score the same model gave the same frame when the test sets were evaluated. Each test frame is run
inside its whole evaluation batch, so all 32 rows of every batch are checked, not only the curated frames.
Run from the demo folder:   .venv\\Scripts\\python.exe tools\\prerun.py
Add --fresh to delete the saved runs first.
"""
import os
import shutil
import sys

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
sys.path.insert(0, APP)
os.environ.setdefault("DEMO_MODE", "live")
import engine as E   # noqa: E402

if "--fresh" in sys.argv and os.path.isdir(E.CACHE):
    shutil.rmtree(E.CACHE)
import server as S   # noqa: E402

TOL = 0.01
worst, bad, n, rows = 0.0, 0, 0, 0
print(f"{'input':30s} {'model':>5s} {'live':>8s} {'recorded':>9s} {'largest diff, whole batch':>26s}  check")
for it in S.manifest():
    if it.get("group") == "upload":
        continue
    for k in (1, 2, 3, 4):
        r = S.engine.run(it["id"], it["dir"], k, it.get("label"), S.eval_p(it, k))
        n += 1
        fd = max([abs(a - b) for a, b in zip(r["frame_p"], r["frame_eval_p"]) if b is not None]
                 + [r.get("batch_max_diff", 0.0)])
        worst = max(worst, fd)
        rows += r.get("batch_rows", 0)
        ev = r["eval_p"]
        same = (r["p"] >= 0.5) == (ev >= 0.5)
        if not same:
            note, bad = "DIFFERENT VERDICT", bad + 1
        elif abs(r["p"] - ev) > TOL:
            note, bad = "score differs by more than 0.01", bad + 1
        else:
            note = "match"
        print(f"{(it['tag'] + ' ' + it['title'])[:30]:30s} {k:5d} {r['p']:8.4f} {ev:9.4f} {fd:26.6f}  {note}", flush=True)
print(f"\n{n} runs, {rows} frame scores compared. Largest difference from a recorded score: {worst:.6f}")
print("ALL RESULTS MATCH THE RECORDED TEST RESULTS." if bad == 0 else
      f"{bad} run(s) do not match. Do not present yet: send me this whole output.")
