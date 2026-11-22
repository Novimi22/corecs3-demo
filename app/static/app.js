{
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const f4 = v => (v == null || isNaN(v)) ? '–' : Number(v).toFixed(4);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const TOL = 0.01;   // a live score within 0.01 of the recorded score, with the same verdict, counts as a match
const MODELS = {1: ['Model 1', 'Baseline (DFA)'], 2: ['Model 2', 'DFA + DAID'], 3: ['Model 3', 'DFA + DAID + fixed SRM'], 4: ['Model 4', 'DFA + DAID + learnable SRM']};
const GROUPS = [['main', 'Sample inputs'], ['hard', 'Difficult inputs'], ['extra', 'More test inputs'], ['upload', 'Your uploads']];
const SETS = {A: ['Test Set A', 'FaceForensics++', 'same source as the training data'], B: ['Test Set B', 'DiffusionFace', 'never seen during training']};
const S = {samples: [], sel: null, runs: {}, inModel: 4, busy: false};

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
  if (!s.set) return 'New input';
  return `${s.tag} · ${SETS[s.set][1]}`;
}
async function loadSamples(selectId) {
  S.samples = await api('/api/samples');
  const g = $('#groups');
  g.innerHTML = GROUPS.map(([key, name]) => {
    const items = S.samples.filter(s => s.group === key);
    if (!items.length) return '';
    return `<div class="gh">${name}</div>` + items.map(s => `<button class="item" data-id="${s.id}">
      <img src="/thumb/${s.id}" alt="" loading="lazy"><span><span class="t">${esc(s.title)}</span><span class="s">${esc(subline(s))}</span></span>
      ${s.label ? `<span class="chip ${s.label === 'Real' ? 'real' : 'fake'}">${s.label}</span>` : '<span class="chip unk">?</span>'}</button>`).join('');
  }).join('') || '<p class="mut small" style="padding:14px">No inputs yet. Copy the bundle folder (README step 3).</p>';
  $$('.item', g).forEach(b => b.addEventListener('click', () => select(b.dataset.id)));
  if (selectId) select(selectId); else if (!S.sel && S.samples.length) select((S.samples.find(s => s.tag === 'S4') || S.samples[0]).id);
}

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
  <div class="case"><div><h2>${s.tag && s.set ? `<span class="tag">${esc(s.tag)}</span> ` : ''}${esc(s.title)}</h2>
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
['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, () => drop.classList.remove('over')));
drop.addEventListener('drop', e => { e.preventDefault(); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
file.addEventListener('change', () => { if (file.files[0]) upload(file.files[0]); });
async function upload(f) {
  const old = drop.innerHTML;
  drop.innerHTML = `<b>Preparing ${esc(f.name)}…</b><br>frames, face crop, 224 × 224, 81 landmarks`;
  try {
    const fd = new FormData(); fd.append('file', f);
    const m = await api('/api/upload', {method: 'POST', body: fd});
    drop.innerHTML = old; await loadSamples(m.id);
  } catch (e) { drop.innerHTML = old + `<div class="err">${esc(e.message)}</div>`; }
}

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

loadStatus(); loadSamples();
}
