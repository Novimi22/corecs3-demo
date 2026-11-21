"""Run ONCE on the HPC, from the thesis folder with its venv active:

    cd C:\\corecs3-dual-stream
    python C:\\corecs3-demo\\tools\\collect_bundle.py

Copies what the demo needs into C:\\corecs3-demo\\bundle (nothing in the thesis folders is changed):
  weights/      the four trained checkpoints
  samples/      the curated test inputs: the SAME 224 x 224 crops and 81-point landmark files the evaluation read
                (from DATA_ROOT), one folder per input, plus samples.json with, for every frame, its row in the test
                CSV (= its position in the evaluation) and the score each model gave it in the evaluation
  clip_weights/ CLIP ViT-L/14, so the laptop needs no download
  shape_predictor_81_face_landmarks.dat   for new uploads
Then zip the bundle folder and put it on the team Drive.
"""
import json
import math
import os
import random
import shutil
import sys

import pandas as pd

THESIS = os.getcwd()
sys.path.insert(0, THESIS)
from src.config import DATA_ROOT, TEST_A_CSV, TEST_B_CSV, BATCH_SIZE, OUTPUT_DIR, MODEL_DOWNLOAD_ROOT  # noqa: E402

DEMO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(DEMO, "bundle")
CKPT = {1: "model1_baseline.pth", 2: "model2_daid_only.pth", 3: "model3_fixed_srm.pth", 4: "model4_full_learnable.pth"}
OUTDIR = {1: OUTPUT_DIR, 2: os.path.join("outputs", "model2"), 3: os.path.join("outputs", "model3"),
          4: os.path.join("outputs", "model4")}
# tag, generative method folder, sample folder (as printed in Chapter IV), group
CURATED = [
    ("S1", "Real_FFPP", "095_0dbde7de", "main"), ("S2", "Real_FFPP", "824_f45ab9f3", "main"),
    ("S3", "DeepFakes", "444_655_da269e5f", "main"), ("S4", "Face2Face", "427_637_0fac8302", "main"),
    ("S5", "FaceSwap", "901_902_327707a8", "main"), ("S6", "NeuralTextures", "095_053_00e8fcfb", "main"),
    ("S7", "Real_DiffFace", "28830_57498ef9", "main"), ("S8", "ADM", "15440_3210c8e7", "main"),
    ("H1", "Real_FFPP", "543_bf35b651", "hard"), ("H2", "Real_FFPP", "080_d3b22235", "hard"),
    ("H3", "Real_FFPP", "256_c5f08095", "hard"), ("H4", "NeuralTextures", "419_824_83f6d9f9", "hard"),
    ("H5", "NeuralTextures", "522_337_43e97647", "hard"), ("H6", "Real_DiffFace", "18310_91fbb426", "hard"),
    ("H7", "Real_DiffFace", "22314_f67c92ad", "hard"), ("H8", "Real_DiffFace", "21633_acb7292f", "hard"),
    ("H9", "ADM", "2755_052cab31", "hard"), ("H10", "ADM", "8822_0abdd520", "hard"),
]
EXTRA = {"Real_FFPP": 1, "DeepFakes": 1, "Face2Face": 1, "FaceSwap": 1, "NeuralTextures": 1, "Real_DiffFace": 2, "ADM": 2}
NAMES = {"Real_FFPP": "Real video", "DeepFakes": "Deepfakes video", "Face2Face": "Face2Face video",
         "FaceSwap": "FaceSwap video", "NeuralTextures": "NeuralTextures video", "Real_DiffFace": "Real photo",
         "ADM": "Diffusion-generated photo"}


def norm(p):
    return "/" + str(p).replace("\\", "/").strip("/")


def local(rel):
    """A CSV path (either slash) under DATA_ROOT, as a path on this machine."""
    return os.path.join(DATA_ROOT, *norm(rel).strip("/").split("/"))


def parts(img):
    """('Face2Face', '427_637_0fac8302') from 'FFpp\\Face2Face\\427_637_0fac8302\\f00000.png'."""
    s = norm(img).split("/")
    return s[-3], s[-2]


def main():
    for sub in ("weights", "samples", "clip_weights"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    sets = {"A": pd.read_csv(TEST_A_CSV), "B": pd.read_csv(TEST_B_CSV)}
    for t, df in sets.items():
        df["row"] = range(len(df))                       # position in the evaluation (loaders use shuffle=False)
        pp = [parts(i) for i in df["img"]]
        df["method"] = [a for a, _ in pp]
        df["folder"] = [b for _, b in pp]

    # every model's scores by row; each predictions file must be in the order of the test CSV
    frame_p, video_p = {}, {}
    for k in CKPT:
        for t, name in (("A", f"model{k}_predictions_test_a.csv"), ("B", f"model{k}_predictions_test_b.csv")):
            pr = pd.read_csv(os.path.join(THESIS, OUTDIR[k], name))
            if len(pr) != len(sets[t]) or not (pr["id"].map(norm).values == sets[t]["img"].map(norm).values).all():
                raise SystemExit(f"[!] {name} is not in the order of the test CSV. Stop and send me this message.")
            frame_p[(k, t)] = pr["p"].to_numpy()
        vp = pd.read_csv(os.path.join(THESIS, OUTDIR[k], f"model{k}_predictions_test_a_video.csv"))
        video_p[k] = dict(zip(vp["video_id"].astype(str), vp["p"]))
    print("[+] the predictions files of all four models match the test CSVs row by row")

    picks = list(CURATED)
    rng = random.Random(7)
    used = {f for _, _, f, _ in CURATED}
    n = 0
    for m, cnt in EXTRA.items():
        df = sets["B" if m in ("Real_DiffFace", "ADM") else "A"]
        pool = sorted(set(df.loc[df["method"] == m, "folder"]) - used)
        for f in rng.sample(pool, cnt):
            n += 1
            picks.append((f"X{n}", m, f, "extra"))

    items = []
    for tag, m, folder, group in picks:
        t = "B" if m in ("Real_DiffFace", "ADM") else "A"
        df = sets[t]
        sel = df[(df["method"] == m) & (df["folder"] == folder)].sort_values("row")
        if sel.empty:
            print(f"[!] {tag} {m}/{folder}: not in Test Set {t}. Stop and send me this message.")
            continue
        sid = f"{tag}-{m}-{folder}".replace("_", "-")
        d = os.path.join(OUT, "samples", sid)
        os.makedirs(d, exist_ok=True)
        src_manifest = os.path.join(os.path.dirname(local(sel["img"].iloc[0])), "manifest.json")
        faces = {}
        if os.path.exists(src_manifest):
            faces = {int(e["frame_index"]): bool(e.get("face_detected", True)) for e in json.load(open(src_manifest))}
        entries = []
        for _, r in sel.iterrows():
            fi = int(r["frame_index"])
            shutil.copy(local(r["img"]), os.path.join(d, f"f{fi:05d}.png"))
            shutil.copy(local(r["landmark"]), os.path.join(d, f"f{fi:05d}_lm81.npy"))
            entries.append({"frame_index": fi, "frame_path": f"f{fi:05d}.png", "landmark_path": f"f{fi:05d}_lm81.npy",
                            "face_detected": faces.get(fi, True), "set": t, "row": int(r["row"]),
                            "eval_p": {str(k): float(frame_p[(k, t)][int(r["row"])]) for k in CKPT}})
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(entries, f)
        vid = str(sel["video_id"].iloc[0])
        label = "Fake" if sel["label"].iloc[0] == "FAKE" else "Real"
        ev = {str(k): (float(video_p[k][vid]) if t == "A" else entries[0]["eval_p"][str(k)]) for k in CKPT}
        for k in CKPT:   # a video's score is the mean of its frame scores (eval_model.py): check it
            mean = sum(e["eval_p"][str(k)] for e in entries) / len(entries)
            if abs(mean - ev[str(k)]) > 1e-6:
                raise SystemExit(f"[!] {tag}: the frame scores do not average to the video score of Model {k}. "
                                 f"Stop and send me this message.")
        items.append({"id": sid, "tag": tag, "group": group, "method": m, "title": NAMES[m], "folder": folder,
                      "set": t, "label": label, "kind": "video" if t == "A" else "image", "frames": len(entries),
                      "video_id": vid, "eval_p": ev})
        print(f"[+] {tag:4s} {m}/{folder}: {len(entries)} frame(s), evaluation scores "
              f"{[round(ev[str(k)], 4) for k in CKPT]}")
    order = {"batch_size": BATCH_SIZE, "rows": {t: len(df) for t, df in sets.items()},
             "batches": {t: math.ceil(len(df) / BATCH_SIZE) for t, df in sets.items()}}
    with open(os.path.join(OUT, "samples", "samples.json"), "w") as f:
        json.dump({"order": order, "items": items}, f, indent=1)

    for k, name in CKPT.items():
        src = os.path.join(THESIS, OUTDIR[k], name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, "weights", name))
            print(f"[+] weights: {name} ({os.path.getsize(src) / 2**20:.0f} MB)")
        else:
            print(f"[!] missing checkpoint {src}")
    clip = os.path.join(MODEL_DOWNLOAD_ROOT, "ViT-L-14.pt")
    if os.path.exists(clip):
        shutil.copy(clip, os.path.join(OUT, "clip_weights", "ViT-L-14.pt"))
        print("[+] CLIP ViT-L/14 weights copied")
    else:
        print(f"[!] CLIP weights not found at {clip}")
    lp = os.getenv("LANDMARK_PREDICTOR", "shape_predictor_81_face_landmarks.dat")
    for cand in (lp, os.path.join(THESIS, lp)):
        if os.path.exists(cand):
            shutil.copy(cand, os.path.join(OUT, "shape_predictor_81_face_landmarks.dat"))
            print("[+] landmark predictor copied")
            break
    else:
        print("[!] shape_predictor_81_face_landmarks.dat not found: set LANDMARK_PREDICTOR")
    print(f"\nDone: {len(items)} inputs -> {OUT}")


if __name__ == "__main__":
    main()
