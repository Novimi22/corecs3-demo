"""Prepares our own extra photos and videos ONCE, so they appear in the demo without uploading them each time.

Put the files (any subfolders) in C:\\corecs3-demo\\own_inputs, then run from the demo folder with the thesis venv:

    python tools\\add_own_inputs.py

Each file goes through the same preprocessing as an upload (frames, MTCNN face crop, 224 x 224, 81 landmarks) and is
saved in C:\\corecs3-demo\\own. The demo lists them in the closed drop-down "Extra inputs (outside our
datasets)" at the bottom of the list. The four models still score them live when you click Run. Files already prepared are skipped;
add --fresh to prepare everything again. Neither folder is ever committed to GitHub (.gitignore).
"""
import json
import os
import re
import shutil
import sys

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
sys.path.insert(0, APP)
import engine as E   # noqa: E402

SRC = os.environ.get("DEMO_OWN_INPUTS", os.path.join(E.ROOT, "own_inputs"))
OUT = os.path.join(E.ROOT, "own")
VIDEO = (".mp4", ".mov", ".avi", ".mkv")
PHOTO = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
# short titles for the files in the "[PICS] Any Upload" Drive folder (file name = code, e.g. A1.jpg)
TITLES = {"A1": "Real teammate selfie", "A2": "Real teammate video", "A3": "Real, dark and blurry",
          "A4": "Real, beauty filter", "A5": "Real, photo of a screen", "B1": "Our face-swap video",
          "B2": "Diffusion portrait", "B3": "GAN face", "B4": "Celeb-DF clip", "C1": "Group photo",
          "C2": "Face hidden by a hand", "D1": "Food (no face)", "D2": "Anime face", "D3": "Video with no face",
          "D4": "Long video", "D5": "Black-and-white portrait"}


def main():
    if not os.path.isdir(SRC):
        raise SystemExit(f"[!] Folder not found: {SRC}\n    Make it and put our photos and videos inside, then run again.")
    if "--fresh" in sys.argv and os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)
    files = []
    for root, _dirs, names in os.walk(SRC):
        for n in sorted(names):
            if os.path.splitext(n)[1].lower() in VIDEO + PHOTO:
                files.append(os.path.join(root, n))
    if not files:
        raise SystemExit(f"[!] No photos or videos in {SRC}.")
    ok = 0
    for path in sorted(files, key=lambda p: os.path.basename(p).lower()):
        name = os.path.basename(path)
        stem, ext = os.path.splitext(name)
        sid = "own-" + re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
        d = os.path.join(OUT, sid)
        if os.path.exists(os.path.join(d, "upload.json")):
            print(f"[=] {name}: already prepared")
            ok += 1
            continue
        os.makedirs(d, exist_ok=True)
        try:
            entries = E.preprocess_upload(path, d)
        except Exception as e:   # e.g. no face in any frame of a video: listed, never shown
            shutil.rmtree(d, ignore_errors=True)
            print(f"[!] {name}: skipped ({e})")
            continue
        code = stem.upper()
        meta = {"id": sid, "tag": code, "title": TITLES.get(code, stem)[:40], "group": "own", "set": None,
                "label": None, "kind": "video" if ext.lower() in VIDEO else "image", "frames": len(entries),
                "folder": None, "eval_p": None}
        with open(os.path.join(d, "upload.json"), "w") as f:
            json.dump(meta, f)
        faces = sum(1 for e in entries if e["face_detected"])
        print(f"[+] {name}: {len(entries)} frame(s), face found in {faces}")
        ok += 1
    print(f"\nDone: {ok} of {len(files)} files ready in {OUT}. Start the demo; open the drop-down at the bottom of the list.")


if __name__ == "__main__":
    main()
