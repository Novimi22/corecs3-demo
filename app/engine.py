"""CORECS3 demo engine.

Runs the four trained models on one input with the thesis code itself (folder `thesis`, branch hpc-model4):
  - preprocessing of new uploads: data_loader.py's own functions (Modules i-iv), saved exactly as the cache was
  - tensors: the thesis's TestDataset (src/data/datasets.py), the class the evaluation read Test Sets A and B with
  - models: eval_model.load_model_k (eval_model1.load_model1 for Model 1), called right after torch.manual_seed(SEED),
    exactly as eval_model.py does
  - scoring: p = softmax(cls)[:, 1], Fake when p >= 0.5, eval mode, no_grad, FP32; a video's score = mean of its frames

Why the random-number replay: Liao et al.'s Global Feature Adapter builds a new, untrained Fusion layer (random
weights) on every forward pass, so a frame's score depends on the random numbers drawn before its batch. The
evaluation seeded once (SEED), built the model, and scored the test sets in CSV order, 32 rows per batch. For a
test-set frame the engine restores the random-number state its batch had in the evaluation (row // 32), so the
Fusion layers get the same weights the evaluation used. New uploads use the state of the first evaluation batch.
check_replay() proves on the real model that one forward pass draws exactly what replay() assumes.

Modes (DEMO_MODE):
  live     every click runs the models on the GPU; all four are loaded at start-up (warm()). If a live run fails, the
           run saved for that input by tools/prerun.py is shown instead, so the screen never goes blank.
  replay   only shows saved runs (no GPU needed): the backup
  standin  random-weight test models (code checks only)
Every live run is saved in cache/<sample>/model<k>.json.
"""
import json
import os
import sys
import tempfile
import threading
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.environ.get("DEMO_BUNDLE", os.path.join(ROOT, "bundle"))
THESIS = os.environ.get("DEMO_THESIS", os.path.join(ROOT, "thesis"))
CACHE = os.environ.get("DEMO_CACHE", os.path.join(ROOT, "cache"))
MODE = os.environ.get("DEMO_MODE", "live")
GPU_SLOTS = int(os.environ.get("DEMO_GPU_MODELS", "2"))   # models kept on the GPU at once; the rest wait in RAM
sys.path.insert(0, THESIS)
os.environ.setdefault("MODEL_DOWNLOAD_ROOT", os.path.join(BUNDLE, "clip_weights"))      # read by thesis src/config.py
os.environ.setdefault("LANDMARK_PREDICTOR", os.path.join(BUNDLE, "shape_predictor_81_face_landmarks.dat"))
ENGINE_VERSION = 2

SEED = 42
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CKPT = {1: "model1_baseline.pth", 2: "model2_daid_only.pth", 3: "model3_fixed_srm.pth", 4: "model4_full_learnable.pth"}
MODELS = {1: "Model 1", 2: "Model 2", 3: "Model 3", 4: "Model 4"}


def samples_meta():
    p = os.path.join(BUNDLE, "samples", "samples.json")
    if not os.path.exists(p):
        return {"order": {"batch_size": 32, "rows": {}, "batches": {}}, "items": []}
    with open(p) as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------- inputs
def load_sample(sample_dir):
    """A sample folder as the cache stores it: f00000.png + f00000_lm81.npy + manifest.json."""
    with open(os.path.join(sample_dir, "manifest.json")) as f:
        entries = json.load(f)
    frames = []
    for e in sorted(entries, key=lambda e: e["frame_index"]):
        name = os.path.basename(e["frame_path"])
        crop = np.asarray(Image.open(os.path.join(sample_dir, name)).convert("RGB"))
        lm = np.load(os.path.join(sample_dir, os.path.basename(e["landmark_path"]))).astype(np.float32)
        raw = os.path.join(sample_dir, f"raw_f{e['frame_index']:05d}.jpg")
        frames.append({"index": int(e["frame_index"]), "file": name, "lm_file": os.path.basename(e["landmark_path"]),
                       "crop": crop, "lm": lm, "face": bool(e.get("face_detected", True)),
                       "raw": raw if os.path.exists(raw) else None, "box": e.get("raw_box"),
                       "set": e.get("set"), "row": e.get("row"), "eval_p": e.get("eval_p")})
    return frames


def preprocess_upload(path, out_dir):
    """Modules i-iv on a new photo or video: data_loader._process_sample's steps, with its own functions."""
    import cv2
    import data_loader as dl   # thesis/data_loader.py
    os.makedirs(out_dir, exist_ok=True)
    is_video = os.path.splitext(path)[1].lower() in dl.VIDEO_EXTS
    if is_video:
        frames = dl.extract_frames(path)
    else:
        img = cv2.imread(path)
        if img is None:
            raise ValueError("This file is not an image or video the system can read.")
        frames = [(0, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))]
    landmarker = dl.LandmarkExtractor(os.environ.get("LANDMARK_PREDICTOR",
                                                     os.path.join(BUNDLE, "shape_predictor_81_face_landmarks.dat")))
    entries = []
    for idx, rgb in frames:
        box = dl.detect_largest_face(rgb)
        if box is None:
            if is_video:
                continue                      # videos keep only frames with a face
            crop, box_in_crop = dl.center_square_crop(rgb)
            face = False
        else:
            crop, box_in_crop = dl.crop_with_margin(rgb, box)
            face = True
        if crop.size == 0:
            continue
        img224, box224 = dl.resize_crop(crop, box_in_crop)
        lm = landmarker(img224, box224)
        cv2.imwrite(os.path.join(out_dir, f"f{idx:05d}.png"), cv2.cvtColor(img224, cv2.COLOR_RGB2BGR))  # as the cache
        np.save(os.path.join(out_dir, f"f{idx:05d}_lm81.npy"), lm)
        raw = Image.fromarray(rgb)
        raw.thumbnail((640, 640))
        s = raw.width / rgb.shape[1]
        raw.save(os.path.join(out_dir, f"raw_f{idx:05d}.jpg"), quality=90)
        entries.append({"frame_index": int(idx), "frame_path": f"f{idx:05d}.png", "landmark_path": f"f{idx:05d}_lm81.npy",
                        "face_detected": face, "raw_box": [v * s for v in box] if box else None})
    if not entries:
        raise ValueError("No face was found in any frame of this video.")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(entries, f)
    return entries


def thesis_dataset(sample_dir, frames, srm):
    """The thesis's TestDataset over this sample's frames (same loading, ToTensor + ImageNet Normalize, landmarks,
    srm_image), so the tensors are built by the code the evaluation used."""
    import pandas as pd
    from src.data.datasets import TestDataset
    df = pd.DataFrame({"img": [f["file"] for f in frames], "landmark": [f["lm_file"] for f in frames],
                       "label": ["REAL"] * len(frames), "video_id": ["demo"] * len(frames),
                       "subgroup": [0] * len(frames), "frame_index": [f["index"] for f in frames]})
    fd, csv = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(csv, index=False)
        return TestDataset(sample_dir, csv, srm=srm)
    finally:
        os.remove(csv)


# ----------------------------------------------------------------------------------------------- RNG replay
def _dfa(model):
    return model.dfa if hasattr(model, "dfa") else model


def replay(model, n):
    """Draws exactly what n forward passes draw from the CPU random-number generator: Adapter.fuse builds
    Fusion(clip_dim, adapter_dim) for every fusion_map block the adapter runs (blocks 1-8), in block order."""
    from src.models.DFACLIP.layer import Fusion
    ad = _dfa(model).GlobalContextAdapter
    clip_dim = _dfa(model).clip_model.visual.conv1.out_channels
    blocks = [b for b in sorted(ad.fusion_map) if b <= min(8, len(ad.vit_model.blocks))]
    for _ in range(n):
        for _b in blocks:
            Fusion(clip_dim, ad.num_features)


def loader_iter_draw():
    """iter(DataLoader) draws one base seed; the evaluation iterated Test Set A, then Test Set B."""
    from torch.utils.data import DataLoader
    from src.data.datasets import BaseDataset

    class One(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return {}
    iter(DataLoader(One(), batch_size=32, shuffle=False, num_workers=0, collate_fn=BaseDataset.collate_fn))


def dummy_batch(k, device):
    d = {"image": torch.zeros(1, 3, 224, 224), "landmark": torch.full((1, 81, 2), 112.0),
         "label": torch.zeros(1, dtype=torch.long)}
    if k >= 3:
        d["srm_image"] = torch.zeros(1, 3, 224, 224)
    return {key: v.to(device) for key, v in d.items()}


def check_replay(model, k, device):
    """One real forward pass must leave the generator exactly where replay(model, 1) leaves it."""
    s = torch.get_rng_state()
    with torch.no_grad():
        model(dummy_batch(k, device))
    after_forward = torch.get_rng_state()
    torch.set_rng_state(s)
    replay(model, 1)
    ok = torch.equal(after_forward, torch.get_rng_state())
    torch.set_rng_state(s)
    return ok


# ----------------------------------------------------------------------------------------------- models
class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.models, self.on_gpu = {}, []
        self.device = "cuda" if (MODE == "live" and torch.cuda.is_available()) else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
        self.warm_state = {"done": 0, "error": None, "running": False}
        meta = samples_meta()
        self.order = meta["order"]
        self.needed = {("A", 0)}   # uploads use the first evaluation batch
        for it in meta["items"]:
            for e in json.load(open(os.path.join(BUNDLE, "samples", it["id"], "manifest.json"))):
                if e.get("set") is not None:
                    self.needed.add((e["set"], e["row"] // self.order["batch_size"]))

    def status(self):
        weights = {k: os.path.exists(os.path.join(BUNDLE, "weights", CKPT[k])) for k in MODELS}
        return {"mode": MODE, "device": self.gpu_name, "weights": weights, "loaded": sorted(self.models),
                "warm": self.warm_state}

    def warm(self):
        """Loads the four models in the background at start-up, so the first click is as fast as the rest."""
        def go():
            self.warm_state["running"] = True
            try:
                for k in MODELS:
                    with self.lock:
                        self.model(k)
                    self.warm_state["done"] = k
            except Exception as e:   # shown in the status chip; saved runs still work
                self.warm_state["error"] = str(e)
            self.warm_state["running"] = False
        threading.Thread(target=go, daemon=True).start()

    def _states(self, model, k):
        """The CPU generator state before every evaluation batch the curated inputs sit in."""
        path = os.path.join(CACHE, "rng", f"model{k}.pt")
        start = torch.get_rng_state()
        if os.path.exists(path):
            saved = torch.load(path, weights_only=False)
            if not torch.equal(saved["start"], start):
                raise RuntimeError(f"Model {k} was built in a different random-number state than in the checked "
                                   f"run (tools/prerun.py). Live runs are off; the checked saved runs are shown.")
            if set(saved["states"]) >= self.needed:
                return saved["states"]
        if not check_replay(model, k, self.device):
            raise RuntimeError(f"Model {k}: a forward pass draws different random numbers than the replay assumes. "
                               f"Do not use live mode; send this message.")
        states = {}
        for t in ("A", "B"):
            loader_iter_draw()
            last = max([b for s, b in self.needed if s == t], default=-1)
            for b in range(self.order["batches"].get(t, 0) if t == "A" else last + 1):
                if (t, b) in self.needed:
                    states[(t, b)] = torch.get_rng_state()
                replay(model, 1)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"start": start, "states": states}, path)
        return states

    def _build(self, k):
        if MODE == "standin":
            from src.config import FEATURE_DIM, DAID_RANK
            from src.daid import DomainAgnosticAggregator
            torch.manual_seed(SEED)
            if k == 1:
                from src.utils.cpu_standin import DFACLIPStandIn
                return DFACLIPStandIn().eval(), None, None
            from src.models.dual_stream import DualStreamDFA
            m = DualStreamDFA(k, device="cpu", cpu_test=True)
            agg = DomainAgnosticAggregator(FEATURE_DIM, DAID_RANK).eval()
            m.attach_daid(agg)
            return m.eval(), agg, None
        path = os.path.join(BUNDLE, "weights", CKPT[k])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint {path}. Copy the bundle from the team Drive (README step 3).")
        from eval_model import load_model_k   # thesis/eval_model.py
        torch.manual_seed(SEED)               # eval_model.main: seed, then build the model
        model, _ckpt = load_model_k(k, path, self.device, False)
        states = self._states(model, k)
        return model, getattr(model, "_daid", None), states

    def model(self, k):
        if self.device == "cuda" and k not in self.on_gpu:
            while len(self.on_gpu) >= GPU_SLOTS:            # move the least recently used model to RAM
                old = self.on_gpu.pop(0)
                om, oagg, _ = self.models[old]
                om.to("cpu")
                if oagg is not None:
                    oagg.to("cpu")
                torch.cuda.empty_cache()
        if k not in self.models:
            self.models[k] = self._build(k)
        m, agg, states = self.models[k]
        if self.device == "cuda":
            if k in self.on_gpu:
                self.on_gpu.remove(k)
            m.to("cuda")
            if agg is not None:
                agg.to("cuda")
            self.on_gpu.append(k)
        return m, agg, states

    # ------------------------------------------------------------------------------------------- one run
    def run(self, sample_id, sample_dir, k, label=None, eval_p=None):
        cached = os.path.join(CACHE, sample_id, f"model{k}.json")

        def saved(reason):
            if not os.path.exists(cached):
                raise RuntimeError(reason)
            with open(cached) as f:
                res = json.load(f)
            res["replayed"] = True
            return res
        if MODE == "replay":
            return saved("This input has no saved run. Saved-runs mode only replays; use run_demo.bat to run it live.")
        try:
            return self._run(sample_id, sample_dir, k, label, eval_p)
        except Exception as e:
            if MODE == "live" and os.path.exists(cached):
                res = saved(str(e))
                res["fallback"] = str(e)
                return res
            raise

    def _run(self, sample_id, sample_dir, k, label, eval_p):
        out_dir = os.path.join(CACHE, sample_id)
        cached = os.path.join(out_dir, f"model{k}.json")
        with self.lock:
            t0 = time.time()
            frames = load_sample(sample_dir)
            m, agg, states = self.model(k)
            srm = k >= 3
            ds = thesis_dataset(sample_dir, frames, srm)
            bs = self.order["batch_size"]
            groups = {}
            for i, f in enumerate(frames):    # frames that shared an evaluation batch share its random numbers
                key = (f["set"], f["row"] // bs) if f.get("set") is not None else ("A", 0)
                groups.setdefault(key, []).append(i)
            p = np.zeros(len(frames))
            first = None
            with torch.no_grad():
                for key, idx in groups.items():
                    d = {n: v.to(self.device) for n, v in ds.collate_fn([ds[i] for i in idx]).items()}
                    if states is not None:
                        torch.set_rng_state(states[key])
                    pred = m(d)
                    p[idx] = torch.softmax(pred["cls"].float(), dim=1)[:, 1].cpu().numpy().astype(float)
                    if 0 in idx:
                        first = (pred, d, idx.index(0))
                extra = self._extras(m, agg, first[0], first[1], first[2], srm, out_dir, k)
            secs = time.time() - t0
        os.makedirs(out_dir, exist_ok=True)
        common_inputs(sample_id, frames, out_dir)
        mean = float(np.mean(p))
        res = {"engine": ENGINE_VERSION, "sample": sample_id, "model": k, "frame_index": [f["index"] for f in frames],
               "frame_p": [float(x) for x in p], "p": mean, "verdict": "Fake" if mean >= 0.5 else "Real",
               "label": label, "correct": None if label is None else ((mean >= 0.5) == (label == "Fake")),
               "eval_p": eval_p, "frame_eval_p": [None if not f.get("eval_p") else f["eval_p"][str(k)] for f in frames],
               "seconds": round(secs, 2), "mode": MODE, "replayed": False, **extra}
        with open(cached, "w") as f:
            json.dump(res, f)
        return res

    def _extras(self, m, agg, pred, d, j, srm, out_dir, k):
        """What the 'Inside the system' screen shows for the first frame (j = its place in its batch)."""
        os.makedirs(out_dir, exist_ok=True)
        ex = {}
        h_sp = pred.get("h_spatial", pred.get("features"))
        if h_sp is not None:
            ex["h_spatial"] = strip(h_sp[j])
        if "f" in pred:
            ex["f"] = strip(pred["f"][j])
        if "h" in pred and k >= 3:
            ex["h"] = strip(pred["h"][j])
        if agg is not None and "h" in pred:   # DAID: size of the added term U^T(h/||h||) next to h
            h = pred["h"][j:j + 1].float()
            add = agg(torch.nn.functional.normalize(h, p=2, dim=1))
            ex["daid_ratio"] = round(float(add.norm() / h.norm()), 5)
        mask = pred.get("mask")
        if mask is not None:                  # Liao's landmark region masks (used for visualization only)
            mk = mask[j].float().mean(0).cpu().numpy()
            heat_overlay(d["image"][j].cpu().numpy(), mk, os.path.join(out_dir, f"m{k}_mask.png"))
            ex["mask_img"] = f"m{k}_mask.png"
        if srm:
            fs = m.frequency_stream
            res = fs.residuals(d["srm_image"][j:j + 1]).float()[0].cpu().numpy()     # [3, 224, 224]
            names = []
            for i in range(3):
                fn = f"m{k}_srm{i}.png"
                residual_png(res[i], os.path.join(out_dir, fn))
                names.append(fn)
            ex["srm_imgs"] = names
            w = fs.srm_weight().detach().float().cpu().numpy()[:, 0]               # [3, 5, 5], red channel
            base = fs.srm_base.detach().float().cpu().numpy()[:, 0]
            ex["kernels"] = [np.round(w[i], 4).tolist() for i in range(3)]
            ex["kernel_change"] = [round(float(np.abs(w[i] - base[i]).max()), 4) for i in range(3)]
        return ex


# ----------------------------------------------------------------------------------------------- pictures
def strip(v, bins=96):
    v = v.detach().float().cpu().numpy().ravel()
    pad = (-len(v)) % bins
    v = np.concatenate([v, np.zeros(pad)]).reshape(bins, -1).mean(1)
    s = float(np.abs(v).max()) or 1.0
    return [round(float(x / s), 3) for x in v]


def residual_png(r, path):
    a = np.abs(r)[3:-3, 3:-3]   # zero padding makes the 2-pixel border light up: leave it out
    hi = np.percentile(a, 99) or 1.0
    img = (255 - np.clip(a / hi, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def heat_overlay(img_chw, mk, path):
    base = ((img_chw.transpose(1, 2, 0) * STD + MEAN).clip(0, 1) * 255).astype(np.uint8)
    hm = np.asarray(Image.fromarray((mk / (mk.max() or 1) * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR))
    over = base.astype(np.float32)
    over[..., 0] = over[..., 0] * 0.55 + hm * 0.45
    over[..., 1] *= 0.75
    Image.fromarray(over.clip(0, 255).astype(np.uint8)).save(path)


def common_inputs(sample_id, frames, out_dir):
    """Pictures shared by the four models: crops, landmarks, the raw frame with its MTCNN box, histograms."""
    meta = os.path.join(out_dir, "inputs.json")
    if os.path.exists(meta):
        return
    crops = []
    for f in frames:
        fn = f"crop_{f['index']:05d}.jpg"
        Image.fromarray(f["crop"]).save(os.path.join(out_dir, fn), quality=92)
        crops.append(fn)
    f0 = frames[0]
    im = Image.fromarray(f0["crop"]).copy()
    dr = ImageDraw.Draw(im)
    for x, y in f0["lm"]:
        dr.ellipse([x - 1.6, y - 1.6, x + 1.6, y + 1.6], fill=(255, 214, 0), outline=(30, 30, 30))
    im.save(os.path.join(out_dir, "landmarks.png"))
    raw = None
    if f0.get("raw"):
        r = Image.open(f0["raw"]).convert("RGB")
        if f0.get("box"):
            x1, y1, x2, y2 = f0["box"]
            mw, mh = (x2 - x1) * 0.2, (y2 - y1) * 0.2
            dr = ImageDraw.Draw(r)
            dr.rectangle([x1, y1, x2, y2], outline=(255, 214, 0), width=3)
            dr.rectangle([x1 - mw, y1 - mh, x2 + mw, y2 + mh], outline=(31, 78, 140), width=3)
        r.save(os.path.join(out_dir, "raw.jpg"), quality=90)
        raw = "raw.jpg"
    c = f0["crop"].astype(np.float32) / 255.0
    hist = {}
    for name, arr in (("before", c), ("after", (c - MEAN) / STD)):
        lo, hi = (0.0, 1.0) if name == "before" else (-2.5, 2.8)
        hist[name] = [np.histogram(arr[..., i], bins=32, range=(lo, hi))[0].tolist() for i in range(3)]
        hist[name + "_range"] = [lo, hi]
    with open(meta, "w") as fh:
        json.dump({"crops": crops, "landmarks": "landmarks.png", "raw": raw, "hist": hist,
                   "face": [bool(f["face"]) for f in frames]}, fh)
