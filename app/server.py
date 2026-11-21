"""CORECS3 system demo: web server. Start with run_demo.bat (Windows) or:  python app/server.py
Then open http://localhost:7860 in Chrome or Edge.
"""
import json
import os
import re
import sys
import uuid

from flask import Flask, abort, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E   # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SAMPLES = os.path.join(E.BUNDLE, "samples")
UPLOADS = os.path.join(E.ROOT, "uploads")
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
engine = E.Engine()


def manifest():
    """bundle/samples/samples.json (written by tools/collect_bundle.py), then this session's uploads."""
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
    out = []
    for it in manifest():
        out.append({k: v for k, v in it.items() if k != "dir"})
    return jsonify(out)


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
        return jsonify({"error": "Use a photo (JPG, PNG) or a short video (MP4, MOV)."}), 400
    sid = "upload-" + uuid.uuid4().hex[:8]
    d = os.path.join(UPLOADS, sid)
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "original" + ext)
    f.save(src)
    try:
        with engine.lock:   # MTCNN draws random numbers while it loads: never while a model is being built
            entries = E.preprocess_upload(src, d)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    meta = {"id": sid, "tag": "New", "title": f.filename[:40], "group": "upload", "set": None, "label": None,
            "kind": "video" if ext in (".mp4", ".mov", ".avi", ".mkv") else "image", "frames": len(entries),
            "folder": None, "eval_p": None}
    json.dump(meta, open(os.path.join(d, "upload.json"), "w"))
    return jsonify(meta)


if __name__ == "__main__":
    port = int(os.environ.get("DEMO_PORT", "7860"))
    print(f"\n  CORECS3 demo ({E.MODE} mode, {engine.gpu_name}) -> open http://localhost:{port}\n")
    if E.MODE == "live":
        engine.warm()
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
