"""Decides how uploads should be scored. Run on the HPC from C:\\corecs3-demo (thesis venv active):
    python -u tools\\compare_upload_modes.py
Scores every curated test input as if it were an upload, in two ways, and compares with its recorded test result:
  alone      the input's own frames are the whole batch
  reference  the batch is filled to 32 with a fixed, balanced set of real and fake Test Set A frames
             (never frames of the input itself)
Uses the random-number state of the first evaluation batch, exactly as uploads do. Changes no saved runs.
"""
import os
import sys

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
sys.path.insert(0, APP)
os.environ.setdefault("DEMO_MODE", "live")
import numpy as np   # noqa: E402
import torch         # noqa: E402
import engine as E   # noqa: E402

eng = E.Engine()
items = E.samples_meta()["items"]
bs = eng.order["batch_size"]
stats = {m: {"agree": 0, "correct": 0, "diff": [], "n": 0} for m in ("alone", "reference")}
print(f"{'input':30s} {'model':>5s} {'recorded':>9s} {'alone':>8s} {'reference':>10s}")
for k in (1, 2, 3, 4):
    m, agg, states = eng.model(k)
    for it in items:
        d = os.path.join(E.BUNDLE, "samples", it["id"])
        frames = E.load_sample(d)
        for f in frames:
            f["file"], f["lm_file"] = os.path.join(d, f["file"]), os.path.join(d, f["lm_file"])
        own = {(f["set"], f["row"]) for f in frames}
        out = {}
        for mode in ("alone", "reference"):
            batch = frames + (eng.reference_frames(bs - len(frames), exclude=own) if mode == "reference" else [])
            ds = E.thesis_dataset("", batch, k >= 3)
            x = {n: v.to(eng.device) for n, v in ds.collate_fn([ds[j] for j in range(len(batch))]).items()}
            if states is not None:
                torch.set_rng_state(states[("A", 0)])
            with torch.no_grad():
                pb = torch.softmax(m(x)["cls"].float(), 1)[:, 1].cpu().numpy()
            p = float(np.mean(pb[:len(frames)]))
            rec = it["eval_p"][str(k)]
            st = stats[mode]
            st["n"] += 1
            st["agree"] += int((p >= .5) == (rec >= .5))
            st["correct"] += int((p >= .5) == (it["label"] == "Fake"))
            st["diff"].append(abs(p - rec))
            out[mode] = p
        print(f"{(it['tag'] + ' ' + it['title'])[:30]:30s} {k:5d} {it['eval_p'][str(k)]:9.4f} {out['alone']:8.4f} "
              f"{out['reference']:10.4f}", flush=True)
print()
for mode, st in stats.items():
    print(f"{mode:9s}: same verdict as the recorded result in {st['agree']}/{st['n']} runs | correct vs the truth in "
          f"{st['correct']}/{st['n']} | average score difference {np.mean(st['diff']):.4f}")
