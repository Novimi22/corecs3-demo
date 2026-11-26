"""CORECS3 system demo: web server. Start with run_demo.bat (Windows) or:  python app/server.py
Then open http://localhost:7860 in Chrome or Edge.
"""
import hashlib
import json
import os
import re
import sys
import time
import uuid

from flask import Flask, abort, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E   # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SAMPLES = os.path.join(E.BUNDLE, "samples")
UPLOADS = os.path.join(E.ROOT, "uploads")
# The list shows these six test inputs. Every other input is a file in the "sample test sets" folder
# (tools/make_sample_folder.py) and appears in the list only after that file is uploaded.
FEATURED = ["X10", "S3", "S7", "S8", "H6", "H9"]
INDEX = os.path.join(E.BUNDLE, "upload_index.json")   # file fingerprint -> test input (written by make_sample_folder.py)
SHOWN = []          # inputs uploaded since the server started, in upload order
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "This file is larger than 200 MB. Use a shorter video."}), 413
engine = E.Engine()


def manifest():
    """Every input the server can run: bundle/samples/samples.json (written by tools/collect_bundle.py), then uploads."""
    items = E.samples_meta()["items"]
    for it in items:
        it["dir"] = os.path.join(SAMPLES, it["id"])
    if os.path.isdir(UPLOADS):
        for sid in sorted(os.listdir(UPLOADS)):
            meta = os.path.join(UPLOADS, sid, "upload.json")
            if os.path.exists(meta):
                it = json.load(open(meta))
                it["dir"] = os.path.join(UPLOADS, sid)
                items.append(it)
    return items


def listed():
    """What the list shows: the six featured test inputs, then whatever was uploaded since the server started."""
    items = {it["id"]: it for it in manifest()}
    feat = [it for t in FEATURED for it in items.values() if it.get("tag") == t and it.get("group") != "upload"]
    return feat + [items[i] for i in SHOWN if i in items and items[i] not in feat]


def fingerprints():
    try:
        return json.load(open(INDEX))
    except (OSError, ValueError):
        return {}


def show(sid):
    if sid in SHOWN:
        SHOWN.remove(sid)
    SHOWN.append(sid)


def eval_p(item, k):
    """The score Model k gave this input in the evaluation of the test sets (None for uploads)."""
    return (item.get("eval_p") or {}).get(str(k))


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:p>")
def static_files(p):
    return send_from_directory(STATIC, p)


@app.get("/media/<sid>/<path:p>")
def media(sid, p):
    if not re.fullmatch(r"[\w\-]+", sid):
        abort(404)
    return send_from_directory(os.path.join(E.CACHE, sid), p)


@app.get("/thumb/<sid>")
def thumb(sid):
    for it in manifest():
        if it["id"] == sid:
            files = sorted(f for f in os.listdir(it["dir"]) if re.fullmatch(r"f\d{5}\.png", f))
            if files:
                return send_from_directory(it["dir"], files[0])
    abort(404)


@app.get("/api/status")
def status():
    return jsonify(engine.status())


@app.get("/api/samples")
def samples():
    feat = set(FEATURED)
    return jsonify([{**{k: v for k, v in it.items() if k != "dir"},
                     "featured": it.get("tag") in feat and it["id"] not in SHOWN and it.get("group") != "upload"}
                    for it in listed()])


@app.post("/api/run")
def run():
    body = request.get_json(force=True)
    sid, k = body["sample"], int(body["model"])
    item = next((it for it in manifest() if it["id"] == sid), None)
    if item is None:
        abort(404)
    try:
        res = engine.run(sid, item["dir"], k, item.get("label"), eval_p(item, k))
    except Exception as e:   # shown on screen, never a blank page
        return jsonify({"error": str(e)}), 500
    inputs = os.path.join(E.CACHE, sid, "inputs.json")
    res["inputs"] = json.load(open(inputs)) if os.path.exists(inputs) else None
    return jsonify(res)


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "No file received."}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mp4", ".mov", ".avi", ".mkv"):
        return jsonify({"error": "This file type is not supported. Use a photo (JPG, JPEG, PNG, WEBP, BMP) "
                                 "or a video (MP4, MOV, AVI, MKV), up to 200 MB."}), 400
    data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    known = fingerprints().get(digest)
    if known and known.get("sample"):
        # One of our test inputs (a file from the "sample test sets" folder): it is run like every test input,
        # on its test-set frames inside its evaluation batch, so the live scores can be checked against the record.
        item = next((it for it in manifest() if it["id"] == known["sample"]), None)
        if item is None:
            return jsonify({"error": "This test input is not in the bundle."}), 404
        show(item["id"])
        return jsonify({**{k: v for k, v in item.items() if k != "dir"}, "recognized": True})
    for it in manifest():   # the same file uploaded again since the server started: show that upload again
        if it["id"] in SHOWN and it.get("sha256") == digest:
            show(it["id"])
            return jsonify({k: v for k, v in it.items() if k != "dir"})
    sid = "upload-" + uuid.uuid4().hex[:8]
    d = os.path.join(UPLOADS, sid)
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "original" + ext)
    with open(src, "wb") as out:
        out.write(data)
    try:
        with engine.lock:   # MTCNN draws random numbers while it loads: never while a model is being built
            entries = E.preprocess_upload(src, d)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    # a file from the folder's "Outside our datasets" keeps its code and title (A1 "Real person, selfie"); others are "New"
    tag, title = (known["tag"], known["title"]) if known else ("New", f.filename[:40])
    meta = {"id": sid, "tag": tag, "title": title, "group": "upload", "set": None, "label": None,
            "kind": "video" if ext in (".mp4", ".mov", ".avi", ".mkv") else "image", "frames": len(entries),
            "folder": None, "eval_p": None, "sha256": digest, "uploaded": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(d, "upload.json"), "w"))
    show(sid)
    return jsonify(meta)


if __name__ == "__main__":
    port = int(os.environ.get("DEMO_PORT", "7860"))
    print(f"\n  CORECS3 demo ({E.MODE} mode, {engine.gpu_name}) -> open http://localhost:{port}\n")
    if E.MODE == "live":
        engine.warm()
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
