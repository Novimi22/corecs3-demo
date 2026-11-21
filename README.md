# CORECS3 system demo

A live demonstration of the four trained models from *Cross-Domain Deepfake Detection on Human Portraits*. It takes an input (a test video, a test photo, or a new upload), runs it through every module, and shows each model's output. It runs on a Windows laptop with an NVIDIA GPU and needs no internet once it is set up.

The thesis code does all the work. The demo uses the `thesis` folder (branch `hpc-model4`) for:

- preprocessing new uploads, with `data_loader.py`
- loading inputs, with `TestDataset`
- building the models, with `eval_model.load_model_k`
- scoring, as `eval_model.py` does

This repository adds only the screens.

**This repository holds no videos, images or weights.** The FaceForensics++ terms do not allow redistribution, so the samples and checkpoints stay on the team Drive, in `corecs3-demo-bundle.zip`.

## Screens

| Screen | What it shows |
|---|---|
| Detect | Choose an input or upload a new photo or video. All four models run on it, one after another. |
| Inside the system | The same input followed through every step: frames, face crop, landmarks, normalization, the two streams, fusion, DAID, the classifier, and the decision. |
| Batch test | Every input in the list through all four models, with a confusion matrix for each model. |

## How the outputs stay identical to the evaluation

Liao et al.'s Global Feature Adapter builds a new layer with random weights on every forward pass. A frame's score therefore depends on the random numbers drawn before its batch during the evaluation.

The demo rebuilds each model exactly as `eval_model.py` did: seed 42, then `load_model_k`. For a test-set frame, it restores the random-number state that frame's batch had in the evaluation. The batch is the frame's row in the test CSV divided by 32. The adapter then draws the same weights, and the frame gets the same score.

For every test input, the Detect screen shows ✓ **Matches the recorded test result** when the live score is within 0.01 of the score in that model's predictions file and gives the same verdict.

`tools/prerun.py` checks this for every input before the defense.

## Setup (once)

**1. HPC: collect the bundle.** Open PowerShell. Go to `C:\corecs3-dual-stream`, activate the venv, and check that you are on branch `hpc-model4`. Then run:

```
git clone https://github.com/<your-account>/corecs3-demo C:\corecs3-demo
python C:\corecs3-demo\tools\collect_bundle.py
```

The script copies the following into `C:\corecs3-demo\bundle`:

- the four checkpoints
- the curated inputs (S1–S8, ten difficult inputs and nine more), with the exact crops and landmark files the evaluation read
- every model's recorded score for every one of their frames
- the CLIP weights
- the landmark predictor

It changes nothing in the thesis folders. Every line should start with `[+]`. Zip the `bundle` folder and upload it to the team Drive as `corecs3-demo-bundle.zip`.

**2. Laptop: get the code.** In PowerShell:

```
cd $HOME\Documents
git clone https://github.com/<your-account>/corecs3-demo
cd corecs3-demo
git clone -b hpc-model4 https://github.com/mejhee/corecs3-dual-stream thesis
```

**3. Laptop: add the bundle.** Download `corecs3-demo-bundle.zip` from the Drive and unzip it inside `corecs3-demo`. Check that `corecs3-demo\bundle\weights\model1_baseline.pth` exists.

**4. Laptop: install.** Use Python 3.12 from python.org and tick "Add to PATH". The versions must match the HPC:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install facenet-pytorch --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `2.6.0+cu126 True` and the name of your RTX card.

**5. Laptop: run every input once and check it.**

```
python tools\prerun.py --fresh
```

This runs every input through the four models, saves each run, and compares every frame's score with its recorded score. The last line must read **ALL RESULTS MATCH THE RECORDED TEST RESULTS.**

## Starting the demo

Double-click `run_demo.bat`. The browser opens at http://localhost:7860; press F11 for full screen. The status chip shows *Loading models 1/4…* for about a minute, then *Live · <your GPU>*. Wait for that before you start.

If a live run ever fails, the screen shows the saved run from step 5 for that input and says so.

**Backup:** if the GPU misbehaves on the day, double-click `run_saved_only.bat`. It shows the runs saved in step 5 and needs no GPU. Uploads don't work in this mode.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `DEMO_MODE` | `live` | `live` runs on the GPU. `replay` only shows saved runs. `standin` uses random-weight test models (for code checks only). |
| `DEMO_GPU_MODELS` | `2` | How many models stay on the GPU at once. The others wait in RAM. |
| `DEMO_PORT` | `7860` | Web port |
