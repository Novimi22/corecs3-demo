# CORECS3 system demo: how to run it on the HPC

The demo runs our four trained models live on the HPC's GPU. You can pick a test input or upload a new photo or video, then watch it go through the whole system: frames, face crop, landmarks, both streams, DAID, and the final REAL or FAKE decision.

Everything below happens **on the HPC**. Connect to it with **Chrome Remote Desktop**.

---

## Where everything is (on the HPC)

| What | Where |
|---|---|
| The demo | `C:\corecs3-demo` |
| Start button | `C:\corecs3-demo\run_demo_hpc.bat` |
| Models, test inputs and recorded scores | `C:\corecs3-demo\bundle` (don't edit) |
| Our thesis code and Python environment | `C:\corecs3-dual-stream` (the demo only reads from it) |
| Uploaded files | `C:\corecs3-demo\uploads` |
| Our own extra photos and videos (originals) | `C:\corecs3-demo\own_inputs` (never on GitHub) |
| The same files, prepared for the demo | `C:\corecs3-demo\own` (never on GitHub) |

---

## 1. Before you start: restart the HPC if needed

The HPC's GPU uses a free licence that stops working about 24 hours after a restart. If nobody has restarted it today:

1. Click **Start**, then the power button, then **Restart**.
2. Wait 2–3 minutes, then reconnect with Chrome Remote Desktop.

**On defense day, always restart it in the morning.**

---

## 2. Start the demo

1. Open **File Explorer** and go to `C:\corecs3-demo`.
2. Double-click **`run_demo_hpc.bat`**.
3. A **black window** opens. **Minimize it. Don't close it.** That window *is* the demo; closing it stops everything.
4. Chrome opens the demo at **http://localhost:7860** after about 8 seconds.
5. Wait until the chip at the top right says **"Live · NVIDIA L4-24Q"**. It shows "Loading models 1/4…" for about a minute first.

---

## 3. Use the demo

| Screen | What to do |
|---|---|
| **Detect** | Pick an input on the left, or upload one, then click **Run the four models**. Each card shows that model's fake probability (0 = real, 1 = fake; 0.50 or higher = FAKE). |
| **Inside the system** | After running an input, pick a model (1–4) to see every step for that input. |
| **Batch test** | Click **Run all inputs**. Each model's card should say **"28 of 28 match the recorded test results"**. |
| **Results (Chapter IV)** | The Chapter IV results as reported in the manuscript: performance table, AUC-ROC bars, confusion matrices, threshold tests, model comparison and error analysis. |

**"✓ Matches the recorded test result"** means the live score is the same as the score that model gave that input in our Chapter IV evaluation. Uploads have no recorded result, so they say "New input: no recorded test result".

### Uploading a photo or video
- **Photos:** JPG, JPEG, PNG, WEBP or BMP. **Videos:** MP4, MOV, AVI or MKV. Up to **200 MB**.
- Any video length works. The system always analyses 10 evenly spaced frames. Keep demo clips under about 30 seconds so results come back quickly.
- Only the **largest face** in each frame is analysed.
- iPhone HEIC photos are rejected. iPhone videos may need *Settings → Camera → Formats → Most Compatible*.
- Our tested upload files and their true answers are in the **Demo uploads** Drive folder and the **Demo Upload References** sheet (team Drive). Download the ones you need to the HPC's Desktop before presenting.
- Use photos of teammates, with their permission, or generated faces. Don't use strangers or public figures.

### Our own extra inputs (no uploading during the defense)
Our own photos and videos are outside FF++ and DiffusionFace, so they are hidden in the closed drop-down **"Extra inputs (outside our datasets)"** at the bottom of the input list. Open it only if the panel asks for an input outside our datasets. They are prepared once (face crop, landmarks), so clicking one is as fast as a test input; the four models still score them live.

To prepare them (once, or after adding files): close the demo, put the files in `C:\corecs3-demo\own_inputs`, then in PowerShell:
```
cd C:\corecs3-dual-stream
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
$env:DEMO_THESIS="C:\corecs3-dual-stream"
cd C:\corecs3-demo
python tools\add_own_inputs.py
```
Use photos of teammates (with permission) or generated faces only. Never commit these files to GitHub.

---

## 4. Stop the demo

Close the **black window** with its **X**. If Windows asks "Terminate batch job?", type `Y` and press Enter.

### Clear old uploads (before presenting)
With the demo **stopped**, delete the folder `C:\corecs3-demo\uploads`. The "Your uploads" list then starts empty next time.

---

## 5. If something goes wrong

| What you see | What to do |
|---|---|
| **"Failed to fetch"** everywhere | The black window was closed. Double-click `run_demo_hpc.bat` again. |
| Chip stays on **"Loading models…"**, or says **"models did not load"** | The GPU licence has probably expired. Restart the HPC (step 1), then start the demo again. |
| The page looks old after an update | Click the page and press **Ctrl + F5**. |
| A red message under the upload box | Read it. Usually the format is wrong, the file is too big, or no face was found. |
| A card says **"Saved run shown: the live run failed"** | The GPU had a problem, so the demo showed the checked saved result instead. Restart the demo; if it keeps happening, restart the HPC. |

**Check the GPU** in PowerShell:
```
cd C:\corecs3-dual-stream
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -c "import torch; x = torch.ones(3).cuda(); print('GPU works:', x * 2)"
```
It should print `GPU works: tensor([2., 2., 2.], device='cuda:0')`. If it says "operation not supported", restart the HPC.

---

## 6. Optional: re-check accuracy

This runs every test input through all four models and compares every score with our recorded results. It takes about 5–10 minutes. Close the demo's black window first, then run in PowerShell:
```
cd C:\corecs3-dual-stream
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
$env:DEMO_THESIS="C:\corecs3-dual-stream"
$env:DEMO_GPU_MODELS="4"
cd C:\corecs3-demo
python -u tools\prerun.py
```
The last line should read **ALL RESULTS MATCH THE RECORDED TEST RESULTS.**

---

## 7. Updating the demo code

The code lives in this GitHub repo. To update one file on the HPC, run this in PowerShell (example for `app/engine.py`):
```
Invoke-WebRequest "https://raw.githubusercontent.com/Novimi22/corecs3-demo/main/app/engine.py?v=1" -OutFile C:\corecs3-demo\app\engine.py
```
Change the `?v=` number each time, so GitHub sends a fresh copy. Restart the demo afterwards.

---

## Rules
- **Never upload the `bundle` folder, videos, images or model weights to GitHub.** The FaceForensics++ terms don't allow sharing them. They stay on the team Drive (`corecs3-demo-bundle.zip`).
- Don't change anything in `C:\corecs3-dual-stream`. The demo only reads from it.

---

## Defense-day checklist
- [ ] Restart the HPC in the morning.
- [ ] Delete `C:\corecs3-demo\uploads` (our own extra inputs in `own` stay).
- [ ] Start `run_demo_hpc.bat` and wait for **"Live · NVIDIA L4-24Q"**.
- [ ] Run S4 once: the four scores should be 0.6192 · 0.5632 · 0.4861 · 0.3925, each with ✓.
- [ ] Have the upload files ready on the HPC's Desktop.
- [ ] Keep the black window minimized, not closed.
