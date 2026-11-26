"""Moves the test inputs that are not in the demo's list into a normal folder, so they can be uploaded when asked.

Run once on the HPC, from the demo folder, with the thesis venv:

    C:\\corecs3-dual-stream\\venv\\Scripts\\python.exe tools\\make_sample_folder.py

It makes C:\\corecs3-demo\\sample test sets with four subfolders:
  Sample Inputs          S1, S2, S4, S5, S6       (sample test inputs from Chapter IV)
  Difficult Inputs       H1-H5, H7, H8, H10       (inputs all four models got wrong, from the error analysis)
  More inputs            X1-X9                    (more test inputs drawn at random from the test sets)
  Outside our datasets   A1 ... D5                (our own photos and videos, copied from own_inputs)
The six inputs in the list (X10, S3, S7, S8, H6, H9) are not copied.

Test Set B inputs are photos: the file is the test image itself (its 224 x 224 face crop, as evaluated). Test Set A
inputs are videos: the file is an MP4 of the video's 10 evaluated face crops. When one of these files is uploaded, the
server recognizes it by its SHA-256 fingerprint (bundle/upload_index.json) and runs the four models live on that test
input exactly as the evaluation did, so the scores can be checked against the recorded ones. A file from "Outside our
datasets" is processed like any new upload (frames, face crop, landmarks) and keeps its code and title. Nothing in
the thesis folder or the bundle's test data is changed. Run again any time; files are rewritten identically.
"""
import hashlib
import json
import os
import shutil
import sys

import cv2

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
sys.path.insert(0, APP)
import engine as E   # noqa: E402
from add_own_inputs import TITLES   # noqa: E402  (the codes and titles of our own inputs)

OUT = os.environ.get("DEMO_SAMPLE_FOLDER", os.path.join(E.ROOT, "sample test sets"))
OWN_SRC = os.environ.get("DEMO_OWN_INPUTS", os.path.join(E.ROOT, "own_inputs"))
FEATURED = {"X10", "S3", "S7", "S8", "H6", "H9"}
SUB = {"main": "Sample Inputs", "hard": "Difficult Inputs", "extra": "More inputs"}
OWN_SUB = "Outside our datasets"
MEDIA = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mp4", ".mov", ".avi", ".mkv")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_video(frames, path):
    """The video's evaluated face crops, in order, two per second."""
    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 2, (w, h))
    if not vw.isOpened():
        raise RuntimeError("OpenCV cannot write MP4 files here")
    for fp in frames:
        vw.write(cv2.imread(fp))
    vw.release()


def main():
    index, n = {}, 0
    for sub in list(SUB.values()) + [OWN_SUB]:
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    for it in E.samples_meta()["items"]:
        if it["tag"] in FEATURED or it.get("group") not in SUB:
            continue
        d = os.path.join(E.BUNDLE, "samples", it["id"])
        frames = [os.path.join(d, e["frame_path"]) for e in json.load(open(os.path.join(d, "manifest.json")))]
        if len(frames) == 1:
            dst = os.path.join(OUT, SUB[it["group"]], it["tag"] + ".png")
            shutil.copyfile(frames[0], dst)
        else:
            dst = os.path.join(OUT, SUB[it["group"]], it["tag"] + ".mp4")
            write_video(frames, dst)
        index[sha(dst)] = {"sample": it["id"], "tag": it["tag"]}
        n += 1
        print(f"[+] {SUB[it['group']]}\\{os.path.basename(dst)}  ({it['title']}, actually {it['label'].lower()})")
    if os.path.isdir(OWN_SRC):
        for root, _dirs, names in os.walk(OWN_SRC):
            for name in sorted(names):
                stem, ext = os.path.splitext(name)
                if ext.lower() not in MEDIA:
                    continue
                code = stem.upper()
                dst = os.path.join(OUT, OWN_SUB, code + ext)
                shutil.copyfile(os.path.join(root, name), dst)
                index[sha(dst)] = {"tag": code, "title": TITLES.get(code, stem)[:40]}
                n += 1
                print(f"[+] {OWN_SUB}\\{os.path.basename(dst)}  ({TITLES.get(code, stem)})")
    else:
        print(f"[i] {OWN_SRC} not found: '{OWN_SUB}' stays empty")
    with open(os.path.join(E.BUNDLE, "upload_index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"\nDone: {n} files in {OUT}\nFingerprints saved in {os.path.join(E.BUNDLE, 'upload_index.json')}")


if __name__ == "__main__":
    main()
