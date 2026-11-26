{
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const f4 = v => (v == null || isNaN(v)) ? '–' : Number(v).toFixed(4);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const TOL = 0.01;   // a live score within 0.01 of the recorded score, with the same verdict, counts as a match
const MODELS = {1: ['Model 1', 'Baseline (DFA)'], 2: ['Model 2', 'DFA + DAID'], 3: ['Model 3', 'DFA + DAID + fixed SRM'], 4: ['Model 4', 'DFA + DAID + learnable SRM']};
const SETS = {A: ['Test Set A', 'FaceForensics++', 'same source as the training data'], B: ['Test Set B', 'DiffusionFace', 'never seen during training']};
const S = {samples: [], sel: null, runs: {}, inModel: 4, busy: false};
// The server lists six test inputs (X10, S3, S7, S8, H6, H9), then every input uploaded since it started.

async function api(url, opt) {
  const r = await fetch(url, opt);
  const j = await r.json().catch(() => ({error: 'The server did not answer.'}));
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
function pipeSteps(k) {
  const s = ['Frames', 'Face crop', 'Landmarks', 'Normalize', 'Spatial stream'];
  if (k >= 3) s.push(k === 4 ? 'Learnable SRM stream' : 'SRM stream', 'Feature fusion');
  if (k >= 2) s.push('DAID');
  s.push('Classifier', 'Decision');
  return s;
}
function match(r) {   // live score vs the score recorded when the test sets were evaluated
  if (r.eval_p == null) return null;
  return Math.abs(r.eval_p - r.p) <= TOL && (r.eval_p >= 0.5) === (r.p >= 0.5);
}

/* ---------------------------------------------------------------- tabs, status, list */
$$('nav.tabs button').forEach(b => b.addEventListener('click', () => show(b.dataset.v)));
function show(v) {
  $$('nav.tabs button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  $$('.view').forEach(x => x.classList.toggle('on', x.id === 'v-' + v));
  if (v === 'inside') renderInside();
  if (v === 'results') renderResults();
  window.scrollTo(0, 0);
}
async function loadStatus() {
  try {
    const s = await api('/api/status');
    const ready = Object.values(s.weights).filter(Boolean).length;
    let t = s.mode === 'live' ? `Live · ${s.device}` : s.mode === 'standin' ? 'Test mode (random weights)' : 'Saved runs';
    let warn = s.mode !== 'live' || ready < 4;
    if (s.mode === 'live' && ready < 4) t += ` · ${ready}/4 models found`;
    else if (s.mode === 'live' && s.warm.error) t += ' · models did not load';
    else if (s.mode === 'live' && s.warm.done < 4) { t = `Loading models ${s.warm.done}/4…`; warn = true; setTimeout(loadStatus, 2500); }
    $('#status span').textContent = t;
    $('#status').classList.toggle('warn', warn);
  } catch (e) { $('#status span').textContent = 'Server not reachable'; }
}
function subline(s) {
  if (!s.set) return s.tag && s.tag !== 'New' ? `${s.tag} · outside our datasets` : 'New input';
  return `${s.tag} · ${SETS[s.set][1]}`;
}
function itemRow(s) {
  return `<button class="item" data-id="${s.id}">
      <img src="/thumb/${s.id}" alt="" loading="lazy"><span><span class="t">${esc(s.title)}</span><span class="s">${esc(subline(s))}</span></span>
      ${s.label ? `<span class="chip ${s.label === 'Real' ? 'real' : 'fake'}">${s.label}</span>` : '<span class="chip unk">?</span>'}</button>`;
}
async function loadSamples(selectId) {
  S.samples = await api('/api/samples');
  const g = $('#groups');
  const feat = S.samples.filter(s => s.featured), ups = S.samples.filter(s => !s.featured);
  g.innerHTML = (feat.map(itemRow).join('') || '<p class="mut small" style="padding:14px">No inputs yet. Copy the bundle folder (README step 3).</p>')
    + (ups.length ? '<div class="gh">Your uploads</div>' + ups.map(itemRow).join('') : '');
  $$('.item', g).forEach(b => b.addEventListener('click', () => select(b.dataset.id)));
  if (selectId) select(selectId); else if (!S.sel && S.samples.length) select(S.samples[0].id);
  else if (S.sel) $$('.item').forEach(b => b.classList.toggle('on', b.dataset.id === S.sel.id));
  if (selectId) { const on = $(`.item[data-id="${selectId}"]`); if (on) on.scrollIntoView({block: 'nearest'}); }
}
$('#up-toggle').addEventListener('click', () => {
  const w = $('#upwrap'); w.hidden = !w.hidden; $('#up-toggle').setAttribute('aria-expanded', !w.hidden);
});

/* ---------------------------------------------------------------- why these results */
// Pre-written notes, chosen by rules from the four scores, the input type and the true answer (when known).
// Every figure comes from Chapter IV at the 0.5 cut-off; nothing here analyses the image itself.
const FACTS = {A: {tpr: [.9888, .9700, .9613, .9700], fpr: [.4050, .1250, .1350, .2400]},
               B: {tpr: [.9382, .7884, .8012, .8277], fpr: [.5135, .2704, .2221, .4631]}};
// Error remarks for the ten inputs that all four models got wrong (Chapter IV error analysis)
const REMARKS = {
  '543_bf35b651': 'a blurry, heavily compressed broadcast frame with the skin texture smoothed out',
  '080_d3b22235': 'a low-resolution news frame with soft focus and a downward gaze',
  '256_c5f08095': 'bright, even lighting and smooth, made-up skin, with the mouth open mid-speech',
  '419_824_83f6d9f9': 'a low-quality webcam frame with the eyes nearly closed and the mouth moving',
  '522_337_43e97647': 'a blurry broadcast frame with the mouth open mid-speech, where the manipulation lies',
  '18310_91fbb426': 'a stylized, retouched portrait with high contrast, heavy makeup and hair over the face',
  '22314_f67c92ad': 'a severely blurred, upscaled close-up with part of the face outside the crop',
  '21633_acb7292f': 'a studio portrait with soft lighting and very smooth, retouched skin',
  '2755_052cab31': 'a sharp image with natural lighting and realistic skin texture and stubble',
  '8822_0abdd520': 'a sharp image with natural wrinkles and realistic skin detail'};
const pct = v => (v * 100).toFixed(1) + '%';
const andList = a => a.length < 2 ? (a[0] || '') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1];
const names = ks => (ks.length === 1 ? 'Model ' : 'Models ') + andList(ks.map(String));
const one = (ks, a, b) => ks.length === 1 ? a : b;
// "the false-positive rate was 40.5% for Model 1 and 24.0% for Model 4" / "the false-positive rate of Model 1 was 40.5%"
const rateOf = (label, ks, rates) => ks.length === 1 ? `the ${label} of Model ${ks[0]} was ${pct(rates[ks[0] - 1])}`
  : `the ${label} was ${andList(ks.map(k => `${pct(rates[k - 1])} for Model ${k}`))}`;
function explain(s, R) {
  const ks = [1, 2, 3, 4], fake = ks.filter(k => R[k].p >= .5), real = ks.filter(k => R[k].p < .5);
  const dom = s.set || (s.kind === 'video' ? 'A' : 'B'), F = FACTS[dom];
  const realWhere = dom === 'A' ? 'the real FaceForensics++ videos of Test Set A' : 'real photos from outside the training data';
  const fakeWhere = dom === 'A' ? 'the fake videos of Test Set A' : 'the diffusion-generated faces of Test Set B';
  const noun = s.kind === 'video' ? 'video' : 'image';
  const remark = REMARKS[s.folder];
  const MECH = 'The error analysis suggests that the models relied partly on skin smoothness and image quality as indicators of manipulation.';
  const out = [];
  out.push(!real.length ? 'All four models classified this face as fake.' : !fake.length ? 'All four models classified this face as real.'
    : `${names(fake)} classified this face as fake, whereas ${names(real)} classified it as real.`);

  if (s.label) {                                   // a test input: the ground truth is known
    const truthFake = s.label === 'Fake';
    const wrong = ks.filter(k => (R[k].p >= .5) !== truthFake), right = ks.filter(k => !wrong.includes(k));
    if (!wrong.length) {
      out.push(dom === 'A'
        ? 'All four models classified this input correctly. This outcome is consistent with the in-domain results in Chapter IV, where every model achieved an AUC-ROC above 0.96 on the FaceForensics++ videos of Test Set A.'
        : 'All four models classified this input correctly. Although the models were less reliable on DiffusionFace than on FaceForensics++, with AUC-ROC values of 0.77 to 0.89 compared with 0.97 to 0.99, they still classified the majority of Test Set B images correctly.');
    } else if (!truthFake) {
      out.push(`Because this face is real, ${wrong.length === 4 ? 'all four fake classifications are false positives'
        : `the fake ${one(wrong, 'classification', 'classifications')} by ${names(wrong)} ${one(wrong, 'is a false positive', 'are false positives')}`}. False positives on real faces were the most consistent failure in this study: on ${realWhere}, ${rateOf('false-positive rate', wrong, F.fpr)}.`);
      if (right.length) out.push(`${names(right)}, which had lower false-positive ${one(right, 'rate', 'rates')} (${andList(right.map(k => pct(F.fpr[k - 1])))}), classified it correctly.`);
      if (remark) out.push(`Chapter IV identifies this ${noun} as one of the ten inputs misclassified by all four models and describes it as ${remark}. ${MECH}`);
      else if (wrong.length === 4) out.push('No model met the false-positive threshold on either test set.');
    } else {
      out.push(`Because this face is fake, ${wrong.length === 4 ? 'all four real classifications are missed detections'
        : `the real ${one(wrong, 'classification', 'classifications')} by ${names(wrong)} ${one(wrong, 'is a missed detection', 'are missed detections')}`}.`);
      if (s.method === 'NeuralTextures') out.push('NeuralTextures modifies only the texture of the mouth region, which made it the most difficult manipulation for the models: it accounted for approximately half of each model\'s missed fake videos.');
      else if (s.method === 'ADM') out.push(`This image was produced by a diffusion model, a type of forgery that none of the models encountered during training. On ${fakeWhere}, ${rateOf('detection rate', wrong, F.tpr)}.`);
      else out.push(`Such misses were uncommon on FaceForensics++: on ${fakeWhere}, ${rateOf('detection rate', wrong, F.tpr)}.`);
      if (remark) out.push(`Chapter IV identifies this ${noun} as one of the ten inputs misclassified by all four models and describes it as ${remark}.` +
        (s.method === 'ADM' ? ' The generated faces that the models missed were generally sharp and natural in appearance, which suggests that the models relied partly on skin smoothness and image quality as indicators of manipulation.' : ''));
    }
  } else {                                         // an upload: the ground truth is unknown
    out.push('Because the ground truth of this upload is unknown, it cannot be determined which classifications are correct; however, the Chapter IV results indicate how often each type of error occurred.');
    if (!fake.length) {
      out.push(`If the face is genuine, this outcome is expected. If it is fake, all four models missed it, which the results suggest is uncommon, since Model 1 alone detected ${pct(F.tpr[0])} of ${fakeWhere}.`);
    } else if (!real.length) {
      out.push(`If the face is genuine, this outcome reflects the most consistent failure in the study: on ${realWhere}, Model 1 classified ${pct(F.fpr[0])} as fake, and no model met the false-positive threshold on either test set. The real faces misclassified as fake in the error analysis were mostly blurry, compressed or retouched. If the face is fake, all four models detected it.`);
    } else {
      out.push(`If the face is genuine, the fake ${one(fake, 'classification', 'classifications')} by ${names(fake)} ${one(fake, 'is a false positive', 'are false positives')}, which is consistent with ${one(fake, 'its false-positive rate', 'their false-positive rates')} on ${realWhere} (${andList(fake.map(k => pct(F.fpr[k - 1])))}${one(fake, '', ', respectively')}) compared with ${andList(real.map(k => pct(F.fpr[k - 1])))} for ${names(real)}.`);
      out.push(`If the face is fake, the real ${one(real, 'classification', 'classifications')} by ${names(real)} ${one(real, 'is a missed detection', 'are missed detections')}; ${one(real, `Model ${real[0]} detected ${pct(F.tpr[real[0] - 1])}`, `these models detected ${andList(real.map(k => pct(F.tpr[k - 1])))}, respectively,`)} of ${fakeWhere}.`);
    }
    out.push(s.kind === 'video'
      ? 'Because a new video also originates outside FaceForensics++, more errors may occur than these in-domain figures indicate.'
      : 'Because a new photo lies outside the FaceForensics++ training data, it corresponds to the out-of-domain condition of Test Set B, in which every model was less accurate than on FaceForensics++.');
  }
  const near = ks.filter(k => Math.abs(R[k].p - .5) < .05);
  if (near.length) out.push((near.length === 1
    ? `The score of Model ${near[0]} (${f4(R[near[0]].p)}) lies within 0.05 of the 0.5 cut-off, so a small change in the score would reverse its classification.`
    : `The scores of ${andList(near.map(k => `Model ${k} (${f4(R[k].p)})`))} lie within 0.05 of the 0.5 cut-off, so small changes in these scores would reverse their classifications.`)
    + ' Chapter V recommends selecting the cut-off from the validation data rather than fixing it at 0.5.');
  return out;
}
function renderWhy(s) {
  const el = $('#why'), R = S.runs[s.id] || {};
  if (!el) return;
  if (![1, 2, 3, 4].every(k => R[k])) { el.innerHTML = ''; return; }
  const lines = explain(s, R);
  el.innerHTML = `<div class="why"><h3>Why these results</h3><p class="lead1">${esc(lines[0])}</p>${lines.slice(1).map(l => `<p>${esc(l)}</p>`).join('')}</div>`;
}

/* ---------------------------------------------------------------- detect */
function select(id) {
  if (S.busy) return;
  S.sel = S.samples.find(s => s.id === id);
  $$('.item').forEach(b => b.classList.toggle('on', b.dataset.id === id));
  renderDetect();
}
function film(s) {
  const r = Object.values(S.runs[s.id] || {})[0];
  if (r && r.inputs) return r.inputs.crops.map((c, i) => `<figure><img src="/media/${s.id}/${c}" alt=""><figcaption>frame ${r.frame_index[i]}</figcaption></figure>`).join('');
  return `<figure><img src="/thumb/${s.id}" alt=""><figcaption>${s.kind === 'video' ? 'first frame' : 'photo'}</figcaption></figure>`;
}
function renderDetect() {
  const s = S.sel; if (!s) return;
  const lab = s.label ? `<span class="chip ${s.label === 'Real' ? 'real' : 'fake'}">Actually ${s.label.toLowerCase()}</span>` : '<span class="chip unk">Actual answer unknown</span>';
  const set = s.set ? `<span class="chip info">${SETS[s.set][0]} · ${SETS[s.set][1]} (${SETS[s.set][2]})</span>` : '<span class="chip info">New input, not part of our data</span>';
  $('#detect-main').innerHTML = `
  <div class="case"><div><h2>${s.tag && s.tag !== 'New' ? `<span class="tag">${esc(s.tag)}</span> ` : ''}${esc(s.title)}</h2>
    <div class="meta">${lab}${set}<span class="chip info">${s.kind === 'video' ? (s.frames === 1 ? '1 frame' : `${s.frames} frames`) : 'Photo'}</span></div>
    <div class="film" id="film">${film(s)}</div></div>
    <div><button class="btn" id="run">Run the four models</button></div></div>
  <div class="pipe" id="pipe">${pipeSteps(4).map(x => `<span>${x}</span>`).join('')}</div>
  <div class="cards" id="cards">${[1, 2, 3, 4].map(k => card(k, (S.runs[s.id] || {})[k])).join('')}</div>
  <div id="why"></div>
  <div id="detect-err"></div>
  <p class="note">Fake probability: 0 means real, 1 means fake. A score of 0.50 or higher is called FAKE. A video's score is the average of its frames' scores.</p>`;
  $('#run').addEventListener('click', () => runAll(s));
  renderWhy(s);
}
function card(k, r) {
  const [n, d] = MODELS[k];
  if (!r) return `<div class="card wait" id="card${k}"><h3>${n}</h3><div class="d">${d}</div><div class="score">0.0000</div>
    <div class="meter"><i style="width:0"></i></div><div class="mlab"><span>0 Real</span><span>0.5</span><span>1 Fake</span></div>
    <div class="stamp real">NOT RUN</div><div class="rec">&nbsp;</div></div>`;
  const fake = r.p >= 0.5;
  let right = '';
  if (r.correct === true) right = '<span class="ok">Correct</span>';
  if (r.correct === false) right = '<span class="bad">Wrong</span>';
  const m = match(r);
  let rec = 'New input: no recorded test result';
  if (m === true) rec = `<span class="ok">✓</span> Matches the recorded test result (<span class="num">${f4(r.eval_p)}</span>)`;
  if (m === false) rec = `<span class="bad">!</span> Recorded test result was <span class="num">${f4(r.eval_p)}</span>`;
  const n2 = r.frame_p.length;
  return `<div class="card" id="card${k}"><h3>${n} ${right}</h3><div class="d">${d}</div>
    <div class="score">${f4(r.p)}</div>
    <div class="meter"><i style="width:${r.p * 100}%;background:${fake ? 'var(--fake)' : 'var(--real)'}"></i></div>
    <div class="mlab"><span>0 Real</span><span>0.5</span><span>1 Fake</span></div>
    <div class="stamp ${fake ? 'fake' : 'real'}">${fake ? 'FAKE' : 'REAL'}</div>
    <div class="rec">${rec}${r.fallback ? '<br><span class="bad">Saved run shown: the live run failed</span>' : ''}<br>${n2} ${n2 > 1 ? 'frames' : (S.sel && S.sel.kind === 'video' ? 'frame' : 'image')}${r.replayed ? '' : ` · ${r.seconds} s`}</div></div>`;
}
async function runModel(s, k, animate) {
  const pipe = $('#pipe');
  if (pipe) pipe.innerHTML = pipeSteps(k).map(x => `<span>${x}</span>`).join('');
  const req = api('/api/run', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({sample: s.id, model: k})});
  if (animate && pipe) {
    const spans = $$('span', pipe);
    for (let i = 0; i < spans.length; i++) {
      spans.forEach((x, j) => { x.className = j < i ? 'done' : j === i ? 'now' : ''; });
      await sleep(170);
    }
  }
  const r = await req;
  if (pipe) $$('span', pipe).forEach(x => x.className = 'done');
  (S.runs[s.id] = S.runs[s.id] || {})[k] = r;
  return r;
}
async function runAll(s) {
  if (S.busy) return;
  S.busy = true; $('#run').disabled = true; $('#detect-err').innerHTML = ''; $('#why').innerHTML = '';
  try {
    for (const k of [1, 2, 3, 4]) {
      const c = $('#card' + k); c.classList.add('busy');
      const r = await runModel(s, k, true);
      c.outerHTML = card(k, r);
      if (k === 1) $('#film').innerHTML = film(s);
    }
    renderWhy(s);
  } catch (e) { $('#detect-err').innerHTML = `<div class="err">${esc(e.message)}</div>`; }
  S.busy = false; const b = $('#run'); if (b) { b.disabled = false; b.textContent = 'Run again'; }
}

/* upload */
const drop = $('#drop'), file = $('#file');
$('#list').addEventListener('dragenter', e => { if ([...(e.dataTransfer.types || [])].includes('Files')) { $('#upwrap').hidden = false; $('#up-toggle').setAttribute('aria-expanded', true); } });
['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, () => drop.classList.remove('over')));
drop.addEventListener('drop', e => { e.preventDefault(); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
file.addEventListener('change', () => { if (file.files[0]) upload(file.files[0]); });
// Only the text inside the box changes while a file uploads; the file chooser itself stays, so a second upload works.
async function upload(f) {
  const txt = $('#drop-text'), old = txt.innerHTML;
  txt.innerHTML = `<b>Preparing ${esc(f.name)}…</b>`;
  try {
    const fd = new FormData(); fd.append('file', f);
    const m = await api('/api/upload', {method: 'POST', body: fd});
    txt.innerHTML = old; $('#upwrap').hidden = true; $('#up-toggle').setAttribute('aria-expanded', false);
    await loadSamples(m.id);
  } catch (e) { txt.innerHTML = old + `<div class="err">${esc(e.message)}</div>`; }
  file.value = '';   // choosing the same file again still uploads it
}

/* ---------------------------------------------------------------- full-size view of any picture */
// Click a picture on the Detect or Inside screens to see it full size; arrows step through the pictures beside it.
const LB = {imgs: [], i: 0};
function lbShow(i) {
  LB.i = (i + LB.imgs.length) % LB.imgs.length;
  const im = LB.imgs[LB.i], fig = im.closest('figure'), cap = fig && fig.querySelector('figcaption');
  const big = $('#lb-img');
  // for uploads and our own inputs the original frame is kept next to the face crop (raw_fNNNNN.jpg)
  const raw = /\/media\/(own-|upload-)[^/]+\/f\d{5}\.png$/.test(im.getAttribute('src') || '') ? im.src.replace(/f(\d{5})\.png$/, 'raw_f$1.jpg') : null;
  big.onerror = raw ? () => { big.onerror = null; big.src = im.src; $('#lb-cap').textContent = (cap ? cap.textContent : ''); } : null;
  big.src = raw || im.src;
  big.classList.toggle('px', !raw && im.naturalWidth > 0 && im.naturalWidth <= 300);
  $('#lb-cap').textContent = (cap ? cap.textContent : '') + (raw ? ' · original frame' : im.naturalWidth ? ` · ${im.naturalWidth} × ${im.naturalHeight} pixels` : '');
  $$('#lightbox .lb-nav').forEach(b => b.hidden = LB.imgs.length < 2);
  $('#lightbox').hidden = false;
}
document.addEventListener('click', e => {
  const im = e.target.closest('#detect-main img, #inside-body img');
  if (!im || e.target.closest('#lightbox')) return;
  const box = im.closest('.film, .vis, .gallery, .step, #inside-body') || document;
  LB.imgs = $$('img', box).filter(x => x.closest('#detect-main, #inside-body'));
  lbShow(Math.max(0, LB.imgs.indexOf(im)));
});
$('#lightbox').addEventListener('click', e => {
  if (e.target.closest('.lb-prev')) return lbShow(LB.i - 1);
  if (e.target.closest('.lb-next')) return lbShow(LB.i + 1);
  if (!e.target.closest('img')) $('#lightbox').hidden = true;
});
document.addEventListener('keydown', e => {
  if ($('#lightbox').hidden) return;
  if (e.key === 'Escape') $('#lightbox').hidden = true;
  if (e.key === 'ArrowLeft') lbShow(LB.i - 1);
  if (e.key === 'ArrowRight') lbShow(LB.i + 1);
});

/* ---------------------------------------------------------------- inside */
$$('#in-model button').forEach(b => b.addEventListener('click', () => { S.inModel = +b.dataset.k; renderInside(); }));
function vec(v, col) {
  if (!v) return '<span class="mut small">Shown when the model runs live.</span>';
  return `<div class="vec">${v.map(x => `<i style="height:${Math.max(4, Math.abs(x) * 100)}%;background:${x >= 0 ? col : '#9aa7b8'}"></i>`).join('')}</div>`;
}
function histSvg(h, key, W = 300, H = 120) {
  const cols = ['#c0392b', '#18794e', '#1d4f91'], data = h[key], mx = Math.max(...data.flat()) || 1;
  const lines = data.map((ch, c) => `<polyline fill="none" stroke="${cols[c]}" stroke-width="2" points="${ch.map((v, i) => `${8 + i * (W - 16) / (ch.length - 1)},${H - 18 - v / mx * (H - 30)}`).join(' ')}"/>`).join('');
  const [lo, hi] = h[key + '_range'];
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="pixel value histogram">${lines}<line x1="8" x2="${W - 8}" y1="${H - 18}" y2="${H - 18}" stroke="#bbb"/>
    <text x="8" y="${H - 4}" font-size="11" fill="#5b6470">${lo}</text><text x="${W - 8}" y="${H - 4}" font-size="11" fill="#5b6470" text-anchor="end">${hi}</text></svg>`;
}
function framesSvg(r, W = 540, H = 180) {
  const n = r.frame_p.length, bw = (W - 40) / n, y = v => H - 26 - v * (H - 46);
  const bars = r.frame_p.map((p, i) => `<rect x="${32 + i * bw + 3}" y="${y(p)}" width="${bw - 6}" height="${H - 26 - y(p)}" rx="2" fill="${p >= .5 ? '#c0392b' : '#18794e'}"/>
    <text x="${32 + i * bw + bw / 2}" y="${y(p) - 4}" font-size="11" text-anchor="middle" fill="#15181d">${p.toFixed(2)}</text>
    <text x="${32 + i * bw + bw / 2}" y="${H - 10}" font-size="10.5" text-anchor="middle" fill="#5b6470">${r.frame_index[i]}</text>`).join('');
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="fake probability of each frame">
    <line x1="30" x2="${W - 6}" y1="${y(.5)}" y2="${y(.5)}" stroke="#7d8793" stroke-dasharray="5 4"/><text x="26" y="${y(.5) + 4}" font-size="11" text-anchor="end" fill="#5b6470">0.5</text>
    ${n > 1 ? `<line x1="30" x2="${W - 6}" y1="${y(r.p)}" y2="${y(r.p)}" stroke="#1d4f91" stroke-width="2"/><text x="${W - 6}" y="${y(r.p) - 5}" font-size="12" text-anchor="end" fill="#1d4f91" font-weight="700">average ${r.p.toFixed(4)}</text>` : ''}${bars}</svg>`;
}
function kernel(k) {
  const mx = Math.max(...k.flat().map(Math.abs)) || 1;
  return `<div class="kgrid">${k.flat().map(v => `<span style="background:${v > 0 ? `rgba(29,79,145,${Math.abs(v) / mx * .8})` : v < 0 ? `rgba(192,57,43,${Math.abs(v) / mx * .8})` : '#f3f5f7'};color:${Math.abs(v) / mx > .55 ? '#fff' : '#15181d'}">${v === 0 ? '0' : v.toFixed(2)}</span>`).join('')}</div>`;
}
async function renderInside() {
  $$('#in-model button').forEach(b => b.classList.toggle('on', +b.dataset.k === S.inModel));
  const s = S.sel, k = S.inModel, body = $('#inside-body');
  if (!s) { body.innerHTML = '<div class="empty">Choose an input on the Detect screen.</div>'; return; }
  $('#in-title').textContent = `Inside the system · ${s.tag && s.set ? s.tag + ' · ' : ''}${s.title}`;
  const r = (S.runs[s.id] || {})[k];
  if (!r) {
    $('#in-sub').textContent = `${MODELS[k][0]} has not run on this input yet.`;
    body.innerHTML = `<button class="btn" id="in-run">Run ${MODELS[k][0]}</button>`;
    $('#in-run').addEventListener('click', async () => { body.innerHTML = '<div class="empty">Running…</div>'; try { await runModel(s, k, false); } catch (e) { body.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; } renderInside(); });
    return;
  }
  $('#in-sub').textContent = `${MODELS[k][0]} · ${MODELS[k][1]}. The input, step by step, from frames to the final decision.`;
  const I = r.inputs || {}, m = id => `/media/${s.id}/${id}`;
  let n = 0;
  const step = (t, p, vis, on = true) => on ? `<div class="step"><div class="lbl"><b>Step ${++n}</b><h3>${t}</h3><p>${p}</p></div><div class="vis">${vis}</div></div>` : '';
  const fake = r.p >= .5;
  const film2 = `<div class="film" style="margin:0">${(I.crops || []).map((c, i) => `<figure><img src="${m(c)}" alt=""><figcaption>frame ${r.frame_index[i]}</figcaption></figure>`).join('')}</div>`;
  body.innerHTML = [
    step(s.kind === 'video' ? 'Frame extraction' : 'Input photo', s.kind === 'video' ? 'Ten evenly spaced frames are taken from the video (fewer if some frames show no face).' : 'A photo is used as a single frame.', film2),
    step('Face cropping and resizing', 'MTCNN finds the face. The box is widened by 20% on every side, cut out, and resized to 224 × 224 pixels.',
      (I.raw ? `<figure><img class="raw" src="${m(I.raw)}" alt=""><figcaption>${I.raw_drawn ? 'Original frame: face box (dashed) and box + 20% (solid)' : 'Original frame: yellow = face found, blue = box + 20%'}</figcaption></figure>` : '') +
      `<figure><img class="big" src="${m(I.crops[0])}" alt=""><figcaption>224 × 224 face crop${I.face && I.face[0] === false ? ' (no face found: centre crop)' : ''}</figcaption></figure>`),
    step('Facial landmarks', '81 points on the face, found with dlib.',
      I.landmarks ? `<figure><img class="big" src="${m(I.landmarks)}" alt=""><figcaption>81 landmarks</figcaption></figure>` : '<span class="mut small">Shown when the model runs live.</span>'),
    step('Normalization', 'Pixel values are rescaled with the ImageNet mean and standard deviation.' + (k >= 3 ? ' An unscaled copy goes to the SRM stream.' : ''),
      I.hist ? `<figure>${histSvg(I.hist, 'before')}<figcaption>Before: pixel values 0 to 1 (red, green, blue)</figcaption></figure><figure>${histSvg(I.hist, 'after')}<figcaption>After: centred near 0</figcaption></figure>` : ''),
    step('Spatial stream (DFA)', 'CLIP with the Global Feature Adapter, plus a branch guided by the landmarks, turn the face into 1,024 numbers.',
      (r.mask_img ? `<figure><img class="big" src="${m(r.mask_img)}" alt=""><figcaption>Face regions from the landmarks</figcaption></figure>` : '') +
      `<figure>${vec(r.h_spatial, '#1d4f91')}<figcaption>Spatial features (1,024 numbers, shown as 96 averaged bars)</figcaption></figure>`),
    step(k === 4 ? 'Learnable SRM stream' : 'SRM stream', (k === 4 ? 'Three SRM filters, adjusted during training,' : 'Three fixed SRM filters') + ' bring out fine noise patterns that editing leaves behind. A small network turns them into 1,024 numbers.',
      `${(r.srm_imgs || []).map((x, i) => `<figure><img class="big" style="width:170px;height:170px" src="${m(x)}" alt=""><figcaption>Noise pattern, filter ${i + 1}</figcaption></figure>`).join('')}
      ${(r.kernels || []).map((kk, i) => `<figure>${kernel(kk)}<figcaption>Filter ${i + 1} (5 × 5)${k === 4 && r.kernel_change ? `, changed by up to ${r.kernel_change[i].toFixed(4)} in training` : ''}</figcaption></figure>`).join('')}
      <figure>${vec(r.f, '#6b3fa0')}<figcaption>Noise features (1,024 numbers)</figcaption></figure>`, k >= 3),
    step('Feature fusion', 'The spatial and noise features are joined (2,048 numbers) and combined back into 1,024.',
      `<figure>${vec(r.h, '#15181d')}<figcaption>Fused features</figcaption></figure>`, k >= 3),
    step('DAID', 'Before classification, DAID adds a learned term to the features: h + Uᵀ(h / ‖h‖).',
      `<div><div class="eq">h + Uᵀ(h / ‖h‖)</div>${r.daid_ratio != null ? `<p class="small">For this input the added term is <span class="big2">${(r.daid_ratio * 100).toFixed(2)}%</span> of the size of h.</p>` : '<p class="small mut">Its size for this input is shown when the model runs live.</p>'}</div>`, k >= 2),
    step('Classifier', s.kind === 'video' ? 'Each frame gets a fake probability. The video\'s score is their average.' : 'The photo gets a fake probability.', framesSvg(r)),
    step('Decision', '0.50 or higher → FAKE, otherwise REAL.', `<div><span class="stamp ${fake ? 'fake' : 'real'}">${fake ? 'FAKE' : 'REAL'}</span>
      <p class="small">Fake probability <span class="num">${f4(r.p)}</span>${s.label ? ` · actually ${s.label.toLowerCase()} · ${r.correct ? '<span class="ok">correct</span>' : '<span class="bad">wrong</span>'}` : ''}</p>
      ${match(r) === true ? `<p class="small"><span class="ok">✓</span> Matches the recorded test result (<span class="num">${f4(r.eval_p)}</span>)</p>` : ''}</div>`),
  ].join('');
}

/* ---------------------------------------------------------------- batch */
$('#batch-run').addEventListener('click', async () => {
  if (S.busy) return;
  const list = S.samples.filter(s => s.label && s.group !== 'upload');
  const total = list.length * 4; let done = 0;
  S.busy = true; $('#batch-run').disabled = true;
  const body = $('#batch-body');
  try {
    for (const s of list) for (const k of [1, 2, 3, 4]) {
      if (!(S.runs[s.id] || {})[k]) await runModel(s, k, false);
      done++; $('#batch-bar').style.width = (done / total * 100) + '%'; $('#batch-txt').textContent = `${done} / ${total} runs`;
      renderBatch(list, body); await sleep(30);
    }
  } catch (e) { body.insertAdjacentHTML('afterbegin', `<div class="err">${esc(e.message)}</div>`); }
  S.busy = false; $('#batch-run').disabled = false;
});
function renderBatch(list, body) {
  const cm = {1: [0, 0, 0, 0], 2: [0, 0, 0, 0], 3: [0, 0, 0, 0], 4: [0, 0, 0, 0]}, mt = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]};
  const rows = list.map(s => {
    const cells = [1, 2, 3, 4].map(k => {
      const r = (S.runs[s.id] || {})[k];
      if (!r) return '<td class="cell-p mut">…</td>';
      const f = r.p >= .5, y = s.label === 'Fake';
      cm[k][f && y ? 0 : !f && !y ? 1 : f && !y ? 2 : 3]++;
      const m = match(r); if (m !== null) { mt[k][1]++; if (m) mt[k][0]++; }
      return `<td class="cell-p ${f ? 'f' : 'r'}">${f ? 'FAKE' : 'REAL'} ${f4(r.p)} ${f === y ? '<span class="v">✓</span>' : '<span class="x">✗</span>'}</td>`;
    }).join('');
    return `<tr><td><b>${esc(s.tag)}</b> ${esc(s.title)}<br><span class="small mut">${s.set ? SETS[s.set][0] + ' · ' + SETS[s.set][1] : ''}</span></td><td><span class="chip ${s.label === 'Real' ? 'real' : 'fake'}">${s.label}</span></td>${cells}</tr>`;
  }).join('');
  body.innerHTML = `<div class="grid4">${[1, 2, 3, 4].map(k => { const [tp, tn, fp, fn] = cm[k], n = tp + tn + fp + fn;
    return `<div class="cm"><b>${MODELS[k][0]}</b> <span class="small mut">${MODELS[k][1]}</span>
    <div class="cmg"><span></span><span class="h">Called real</span><span class="h">Called fake</span>
    <span class="h">Real</span><span class="c g">${tn}</span><span class="c r">${fp}</span>
    <span class="h">Fake</span><span class="c r">${fn}</span><span class="c g">${tp}</span></div>
    <p class="small" style="margin:8px 0 0">Correct on <b>${tp + tn} of ${n}</b> inputs</p>
    <p class="small" style="margin:2px 0 0">${mt[k][1] ? `${mt[k][0] === mt[k][1] ? '<span class="ok">✓</span>' : '<span class="bad">!</span>'} <b>${mt[k][0]} of ${mt[k][1]}</b> match the recorded test results` : ''}</p></div>`; }).join('')}</div>
  <div class="tbl"><table><thead><tr><th>Input</th><th>Actually</th>${[1, 2, 3, 4].map(k => `<th class="n">${MODELS[k][0]}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

/* ---------------------------------------------------------------- results (chapter IV) */
// Every value on the Results screen is copied from Chapter IV of the manuscript (Tables IV6 to IV13, Figures 4.17 to 4.21).
const CH4 = {
  A: {tpr: [.9888, .9700, .9613, .9700], fpr: [.4050, .1250, .1350, .2400], acc: [.9100, .9510, .9420, .9280],
      auc: [.9784, .9836, .9865, .9685], frame_auc: [.9411, .9526, .9548, .9197]},
  B: {tpr: [.9382, .7884, .8012, .8277], fpr: [.5135, .2704, .2221, .4631], acc: [.7124, .7590, .7895, .6823],
      auc: [.8854, .8361, .8708, .7712]},
  gap: [.0930, .1475, .1157, .1974],
  // confusion counts at the 0.5 cut-off: [TP, FN, FP, TN]
  cm: {A: [[791, 9, 81, 119], [776, 24, 25, 175], [769, 31, 27, 173], [776, 24, 48, 152]],
       B: [[28145, 1855, 15404, 14596], [23652, 6348, 8111, 21889], [24035, 5965, 6663, 23337], [24830, 5170, 13893, 16107]]},
  // threshold tests: [label, set, threshold, key, p-values of Models 1-4, outcome of Models 1-4]
  tests: [['TPR', 'A', '≥ 0.85', 'tpr', ['< 0.001', '< 0.001', '< 0.001', '< 0.001'], [1, 1, 1, 1]],
          ['FPR', 'A', '≤ 0.15', 'fpr', ['= 1.000', '= 0.161', '= 0.276', '= 1.000'], [0, 0, 0, 0]],
          ['Accuracy', 'A', '≥ 0.85', 'acc', ['< 0.001', '< 0.001', '< 0.001', '< 0.001'], [1, 1, 1, 1]],
          ['AUC-ROC', 'A', '≥ 0.90', 'auc', ['< 0.001', '< 0.001', '< 0.001', '< 0.001'], [1, 1, 1, 1]],
          ['TPR', 'B', '≥ 0.85', 'tpr', ['< 0.001', '= 1.000', '= 1.000', '= 1.000'], [1, 0, 0, 0]],
          ['FPR', 'B', '≤ 0.15', 'fpr', ['= 1.000', '= 1.000', '= 1.000', '= 1.000'], [0, 0, 0, 0]],
          ['Accuracy', 'B', '≥ 0.85', 'acc', ['= 1.000', '= 1.000', '= 1.000', '= 1.000'], [0, 0, 0, 0]],
          ['AUC-ROC', 'B', '≥ 0.85', 'auc', ['< 0.001', '= 1.000', '< 0.001', '= 1.000'], [1, 0, 1, 0]],
          ['Cross-domain AUC gap', '', '≤ 0.10', 'gap', ['= 0.045', '= 1.000', '= 1.000', '= 1.000'], [1, 0, 0, 0]]],
  // McNemar: [pair, set, only the earlier model correct, only the later model correct, statistic, p]
  mcnemar: [[[1, 2], 'A', 16, 57, '21.918', '< 0.001'], [[1, 2], 'B', 5889, 8689, '537.413', '< 0.001'],
            [[2, 3], 'A', 18, 9, '2.370', '= 0.124'], [[2, 3], 'B', 3546, 5377, '375.311', '< 0.001'],
            [[3, 4], 'A', 35, 21, '3.018', '= 0.082'], [[3, 4], 'B', 9675, 3240, '3205.293', '< 0.001']],
  // effect of each added component: [step, component, ΔAUC on Test Set B [CI], Δ|gap| [CI]]
  effects: [['1 → 2', 'DAID', [-.0493, -.0518, -.0468], [.0545, .0476, .0615]],
            ['2 → 3', 'Fixed SRM frequency stream', [.0347, .0326, .0368], [-.0318, -.0367, -.0267]],
            ['3 → 4', 'Learnable SRM filters', [-.0997, -.1023, -.0970], [.0817, .0726, .0898]]],
  // inputs misclassified by all four models: [input, truth, scores of Models 1-4, remark]
  errors: [['Real_FFPP/543_bf35b651', 'Real', [.8401, .9067, .7874, .9732], 'Blurry, heavily compressed broadcast frame; skin texture smoothed out'],
           ['Real_FFPP/080_d3b22235', 'Real', [.8844, .9174, .6407, .9697], 'Low-resolution news broadcast frame; soft focus and downward gaze'],
           ['Real_FFPP/256_c5f08095', 'Real', [.9519, .7432, .6070, .9222], 'Bright, even lighting and smooth, made-up skin; mouth open mid-speech'],
           ['NeuralTextures/419_824_83f6d9f9', 'Fake', [.4613, .1117, .1542, .1257], 'Low-quality webcam frame; eyes nearly closed and mouth moving'],
           ['NeuralTextures/522_337_43e97647', 'Fake', [.1982, .1371, .3377, .1939], 'Blurry broadcast frame; mouth open mid-speech, where the manipulation lies'],
           ['Real_DiffFace/18310_91fbb426', 'Real', [.9995, .9992, .9990, .9994], 'Stylized, retouched portrait; high contrast, heavy makeup, hair over the face'],
           ['Real_DiffFace/22314_f67c92ad', 'Real', [.9996, .9990, .9994, .9984], 'Severely blurred, upscaled close-up; face partly outside the crop'],
           ['Real_DiffFace/21633_acb7292f', 'Real', [.9995, .9991, .9985, .9992], 'Studio portrait with soft lighting and very smooth, retouched skin'],
           ['ADM/2755_052cab31', 'Fake', [.0188, .0009, .0003, .0014], 'Sharp image with natural lighting and realistic skin texture and stubble'],
           ['ADM/8822_0abdd520', 'Fake', [.0156, .0041, .0032, .0058], 'Sharp image with natural wrinkles and realistic skin detail']]
};
const KS = [1, 2, 3, 4];
const n0 = v => Number(v).toLocaleString('en-US');
const sgn4 = v => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(4);
const src = t => `<span class="srcTag">${t}</span>`;

function renderResults() {
  const body = $('#results-body');
  // performance table, best of the four highlighted
  const best = (vals, low) => { const b = low ? Math.min(...vals) : Math.max(...vals); return vals.map(v => v === b); };
  const row = (label, vals, low, ref) => { const h = best(vals, low);
    return `<tr><td>${label}${ref ? ' <span class="small mut">(reference)</span>' : ''}</td>${vals.map((v, i) => `<td class="n${h[i] ? ' best' : ''}">${v.toFixed(4)}</td>`).join('')}</tr>`; };
  const setRows = t => [['TPR (fakes detected)', 'tpr'], ['FPR (real faces called fake)', 'fpr', 1], ['Accuracy', 'acc'], ['AUC-ROC', 'auc']]
    .map(([l, key, low]) => row(l, CH4[t][key], low)).join('') + (t === 'A' ? row('Frame-level AUC-ROC', CH4.A.frame_auc, 0, 1) : '');
  const passes = KS.map(k => CH4.tests.filter(x => x[5][k - 1]).length);
  const perf = `<div class="tbl"><table class="res"><thead><tr><th>Metric</th>${KS.map(k => `<th class="n">${MODELS[k][0]}<br><span class="small mut">${MODELS[k][1]}</span></th>`).join('')}</tr></thead><tbody>
    <tr class="grp"><td colspan="5">Test Set A · FaceForensics++ (in-domain) · 1,000 videos, scored per video</td></tr>${setRows('A')}
    <tr class="grp"><td colspan="5">Test Set B · DiffusionFace (out-of-domain) · 60,000 images, scored per image</td></tr>${setRows('B')}
    <tr class="grp"><td colspan="5">Cross-domain</td></tr>${row('Cross-domain AUC gap |AUC A − AUC B|', CH4.gap, 1)}
    <tr><td><b>Threshold tests passed</b></td>${passes.map(p => `<td class="n"><b>${p} of 9</b></td>`).join('')}</tr></tbody></table></div>`;

  // AUC bars (one axis, 0.60 to 1.00)
  const x = v => ((v - .6) / .4 * 100).toFixed(2);
  const bars = `<div class="aucbars">${KS.map(k => `<div class="ab"><b>${MODELS[k][0]}</b><div class="abt">
      ${['A', 'B'].map(t => `<div class="abr ${t}"><i style="width:${x(CH4[t].auc[k - 1])}%"></i><span class="num">${t} ${CH4[t].auc[k - 1].toFixed(4)}</span></div>`).join('')}
      <em class="th" style="left:${x(.90)}%" title="Test Set A threshold 0.90"></em><em class="th b" style="left:${x(.85)}%" title="Test Set B threshold 0.85"></em></div></div>`).join('')}
    <div class="abaxis"><span></span><div>${[.6, .7, .8, .9, 1].map(v => `<span style="left:${x(v)}%">${v.toFixed(2)}</span>`).join('')}</div></div></div>
    <p class="small mut"><span class="key A"></span> Test Set A &nbsp; <span class="key B"></span> Test Set B &nbsp; Dashed lines: thresholds of 0.90 (Test Set A) and 0.85 (Test Set B). The axis starts at 0.60.</p>`;

  // confusion matrices
  const cms = `<div class="grid4">${KS.map(k => ['A', 'B'].map(t => { const [tp, fn, fp, tn] = CH4.cm[t][k - 1];
    return `<div class="cm"><b>${MODELS[k][0]}</b> <span class="small mut">· ${SETS[t][0]}</span>
      <div class="cmg"><span></span><span class="h">Called real</span><span class="h">Called fake</span>
      <span class="h">Real</span><span class="c g">${n0(tn)}</span><span class="c r">${n0(fp)}</span>
      <span class="h">Fake</span><span class="c r">${n0(fn)}</span><span class="c g">${n0(tp)}</span></div></div>`; }).join('')).join('')}</div>`;

  // threshold tests
  const tests = `<div class="tbl"><table class="res"><thead><tr><th>Test</th><th>Threshold</th>${KS.map(k => `<th class="n">${MODELS[k][0]}</th>`).join('')}</tr></thead><tbody>
    ${CH4.tests.map(([l, t, th, , ps, ok]) => `<tr><td>${t ? SETS[t][0] + ' · ' : ''}${l}</td><td class="num">${th}</td>${ps.map((p, i) => `<td class="n ${ok[i] ? 'pass' : 'fail'}">${ok[i] ? 'pass' : 'fail'} · p ${p}</td>`).join('')}</tr>`).join('')}
    <tr><td colspan="2"><b>Decision</b></td>${passes.map(p => `<td class="n"><b>Fail to reject H₀</b><br><span class="small mut">(${p} of 9)</span></td>`).join('')}</tr></tbody></table></div>`;

  // McNemar and component effects
  const mcn = `<div class="tbl"><table class="res"><thead><tr><th>Models</th><th>Test set</th><th class="n">Only the earlier<br>model correct</th><th class="n">Only the later<br>model correct</th><th class="n">χ²</th><th class="n">p</th></tr></thead><tbody>
    ${CH4.mcnemar.map(([[a, b], t, e, l, s, p]) => `<tr><td>${a} vs ${b}</td><td>${t}</td><td class="n">${n0(e)}</td><td class="n">${n0(l)}</td><td class="n">${s.replace(/\B(?=(\d{3})+\.)/g, ',')}</td><td class="n">${p.replace('= ', '')}</td></tr>`).join('')}</tbody></table></div>`;
  const ci = ([v, lo, hi], goodUp) => `<td class="n ${(v > 0) === goodUp ? 'pass' : 'fail'}">${sgn4(v)}<br><span class="small">[${sgn4(lo)}, ${sgn4(hi)}]</span></td>`;
  const eff = `<div class="tbl"><table class="res"><thead><tr><th>Step</th><th>Added component</th><th class="n">Change in Test Set B AUC</th><th class="n">Change in |gap|</th></tr></thead><tbody>
    ${CH4.effects.map(([s, c, a, g]) => `<tr><td>Model ${s}</td><td>${c}</td>${ci(a, true)}${ci(g, false)}</tr>`).join('')}</tbody></table></div>
    <p class="small mut">Green marks a change that improved cross-domain detection and red one that worsened it. No 95% confidence interval contains zero. These are reported as estimates, without an accept-or-reject decision. For reference only, Model 4's Test Set B AUC-ROC (0.7712, 95% CI [0.7674, 0.7749]) was below the 0.816 that Liao et al. (2026) reported on DFDC; the datasets and scoring level differ.</p>`;

  // error analysis
  const errs = `<div class="tbl"><table class="res"><thead><tr><th>Input</th><th>Actually</th>${KS.map(k => `<th class="n">${MODELS[k][0]}</th>`).join('')}<th>Remarks</th></tr></thead><tbody>
    ${CH4.errors.map(([n, y, ps, r]) => `<tr><td class="num small">${esc(n)}</td><td><span class="chip ${y === 'Real' ? 'real' : 'fake'}">${y}</span></td>${ps.map(p => `<td class="cell-p ${p >= .5 ? 'f' : 'r'}">${p.toFixed(4)}</td>`).join('')}<td class="small">${esc(r)}</td></tr>`).join('')}</tbody></table></div>`;

  body.innerHTML = `
  <div class="sec"><h2>Performance of the four models ${src('Table IV6')}</h2><p class="lead">TPR, FPR and accuracy use the 0.5 cut-off. The best value of the four is highlighted in each row.</p>${perf}</div>
  <div class="sec"><h2>AUC-ROC on both test sets ${src('Figure 4.17')}</h2><p class="lead">Every model was less accurate on the diffusion-generated faces of Test Set B. Only Models 1 and 3 stayed above 0.85 on Test Set B, and only Model 1 kept the gap within 0.10.</p>${bars}</div>
  <div class="sec"><h2>Confusion matrices ${src('Figures 4.18–4.21')}</h2><p class="lead">At the 0.5 cut-off. Test Set A has 800 fake and 200 real videos; Test Set B has 30,000 fake and 30,000 real images.</p>${cms}</div>
  <div class="sec"><h2>Threshold tests and hypothesis decisions ${src('Tables IV7–IV10')}</h2><p class="lead">A threshold is met only when the one-proportion z-test (TPR, FPR and accuracy) or the DeLong-based z-test (AUC-ROC and the gap) shows at α = 0.05 that the value significantly meets it. H₀ is rejected only if all nine tests pass.</p>${tests}</div>
  <div class="sec"><h2>Comparison of consecutive models ${src('Tables IV11–IV12')}</h2><div class="grid2"><div><h3>McNemar's test on paired predictions</h3>${mcn}</div><div><h3>Effect of each added component (95% CI)</h3>${eff}</div></div></div>
  <div class="sec"><h2>Inputs misclassified by all four models ${src('Table IV13')}</h2><p class="lead">The most confidently misclassified inputs of each kind among those that every model got wrong. Most were real faces called fake: blurry, compressed or heavily retouched photos.</p>${errs}</div>`;
}

loadStatus(); loadSamples();
}
