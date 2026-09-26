// Robustness Lab – degrade one recording with sliders and watch both models react (see src/services/lab.py).
(() => {
  const $ = id => document.getElementById(id);
  const S = { token: null, meta: null, clean: null, reqId: 0, gtmReady: null, chart: null };
  const OFF = 41;
  const sliders = {
    noise_snr: { el: $('s-noise'), val: $('v-noise'), fmt: v => v >= OFF ? 'Off' : `${v} dB SNR`, get: v => v >= OFF ? null : v, set: v => v == null ? OFF : v },
    bg_snr:    { el: $('s-bg'), val: $('v-bg'), fmt: v => v >= OFF ? 'Off' : `${v} dB SNR`, get: v => v >= OFF ? null : v, set: v => v == null ? OFF : v },
    echo_rt60: { el: $('s-echo'), val: $('v-echo'), fmt: v => v > 0 ? `${v.toFixed(2)} s` : 'Off', get: v => v, set: v => v || 0 },
    distance_m:{ el: $('s-dist'), val: $('v-dist'), fmt: v => v > 0 ? `${v} m` : 'Off', get: v => v, set: v => v || 0 },
    gain_db:   { el: $('s-gain'), val: $('v-gain'), fmt: v => `${v > 0 ? '+' : ''}${v} dB`, get: v => v, set: v => v || 0 },
    cut_start_pct: { el: $('s-cut'), val: $('v-cut'), fmt: v => v > 0 ? `first ${v}% removed` : 'Off', get: v => v, set: v => v || 0 },
  };

  function params() {
    const p = {};
    for (const [k, s] of Object.entries(sliders)) p[k] = s.get(parseFloat(s.el.value));
    p.device = $('s-device').checked;
    return p;
  }
  function labels() { for (const s of Object.values(sliders)) s.val.textContent = s.fmt(parseFloat(s.el.value)); }
  function applyPreset(p) {
    for (const [k, s] of Object.entries(sliders)) s.el.value = s.set(p[k] ?? null);
    $('s-device').checked = !!p.device;
    labels(); schedule(0);
  }

  // ---------------------------------------------------------------- rendering
  const top = sc => Object.entries(sc || {}).sort((a, b) => b[1] - a[1]);
  function bars(el, scores, highlight, color) {
    el.innerHTML = top(scores).slice(0, 5).map(([c, v]) => `
      <div><div class="flex justify-between text-xs"><span class="${c === highlight ? 'font-semibold text-white' : 'text-slate-300'}">${SS.esc(c)}</span>
        <span class="tabular-nums text-slate-400">${(v * 100).toFixed(1)}%</span></div>
        <div class="xai-track"><div class="xai-bar" style="width:${Math.max(1, v * 100)}%;background:${color}"></div></div></div>`).join('');
  }
  function delta(now, base, cls) {
    if (!base || !cls) return '';
    const d = (now[cls] || 0) - (base[cls] || 0);
    if (Math.abs(d) < 0.005) return '<span class="text-slate-500">± 0 vs clean</span>';
    return `<span class="${d > 0 ? 'lab-delta-up' : 'lab-delta-down'}">${d > 0 ? '▲' : '▼'} ${(Math.abs(d) * 100).toFixed(1)} pts vs clean</span>`;
  }
  const refClass = () => (S.meta && S.meta.actual_class) || (S.clean && S.clean.py.prediction);

  function showPython(r) {
    $('py-pred').textContent = r.prediction || '—';
    $('py-conf').innerHTML = `${SS.pct(r.confidence)} ${r.uncertain ? '<span class="badge q-Poor">uncertain</span>' : ''} · ${delta(r.scores, S.clean && S.clean.py.scores, refClass())}`;
    bars($('py-bars'), r.scores, r.prediction, 'linear-gradient(90deg,#8E76B8,#D4C4E8)');
  }
  function showGtm(g) {
    if (!g) { $('gtm-pred').textContent = LAB.gtm ? '…' : 'not installed'; $('gtm-conf').textContent = ''; $('gtm-bars').innerHTML = ''; return; }
    const [c, v] = top(g)[0] || ['—', 0];
    $('gtm-pred').textContent = c;
    $('gtm-conf').innerHTML = `${SS.pct(v)} · ${delta(g, S.clean && S.clean.gtm, refClass())}`;
    bars($('gtm-bars'), g, c, 'linear-gradient(90deg,#B06FA0,#E8A6D0)');
  }
  function verdict(r, g) {
    const ref = refClass(), gp = g ? top(g)[0][0] : null;
    const parts = [];
    if (S.meta.actual_class) {
      parts.push(`True class: <b>${SS.esc(S.meta.actual_class)}</b>.`);
      parts.push(`Python is <b class="${r.prediction === ref ? 'lab-delta-up' : 'lab-delta-down'}">${r.prediction === ref ? 'correct' : 'wrong'}</b>` +
        (g ? `, GTM is <b class="${gp === ref ? 'lab-delta-up' : 'lab-delta-down'}">${gp === ref ? 'correct' : 'wrong'}</b>.` : '.'));
    } else if (S.clean) {
      parts.push(`Clean prediction was <b>${SS.esc(ref)}</b>; Python now says <b>${SS.esc(r.prediction)}</b>` + (g ? `, GTM says <b>${SS.esc(gp)}</b>.` : '.'));
    }
    if (g) parts.push(gp === r.prediction ? '✅ Both models agree.' : '⚠ The models disagree – in the app this goes to manual review.');
    if (['Poor', 'Unusable'].includes(r.quality)) parts.push(`Audio quality is <b>${r.quality}</b> – the app would flag it and ask for manual review instead of trusting an automatic alert.`);
    else if (r.uncertain) parts.push('Confidence is below the threshold – the app would report an Uncertain Result.');
    $('lab-verdict').innerHTML = parts.join(' ');
  }
  function badges(r) {
    const snr = r.snr_db == null ? '' : ` · SNR ≈ ${r.snr_db} dB`;
    $('lab-badges').innerHTML = `${SS.qBadge(r.quality)} <span class="text-xs text-slate-400">${r.duration}s · ${r.segments} segment(s)${snr} · Python ${r.python_ms} ms</span>` +
      (r.quality_issues && r.quality_issues.length ? `<span class="text-xs text-amber-300">${r.quality_issues.map(SS.esc).join('; ')}</span>` : '');
  }

  // ---------------------------------------------------------------- GTM (browser)
  async function gtmScores(url) {
    if (!LAB.gtm) return null;
    if (!S.gtmReady) S.gtmReady = GTM.load(GTM_BASE);
    await S.gtmReady;
    return GTM.classify(await GTM.decodeUrl(url));
  }

  // ---------------------------------------------------------------- run
  let timer = null;
  function schedule(ms = 350) { clearTimeout(timer); timer = setTimeout(run, ms); }
  async function run() {
    if (!S.token) return;
    const id = ++S.reqId;
    $('lab-result').classList.add('lab-busy');
    $('lab-status').textContent = 'Python model analysing…';
    try {
      const r = await SS.api(LAB.urls.run, { method: 'POST', json: { token: S.token, params: params() } });
      if (id !== S.reqId) return;
      $('lab-audio').src = r.audio_url;
      badges(r); showPython(r); showGtm(null);
      $('lab-status').textContent = LAB.gtm ? 'Teachable Machine analysing in your browser…' : '';
      let g = null;
      try { g = await gtmScores(r.audio_url); } catch (e) { console.error(e); $('lab-status').textContent = 'GTM failed: ' + e.message; }
      if (id !== S.reqId) return;
      if (!S.clean) S.clean = { py: r, gtm: g };
      showPython(r); showGtm(g); verdict(r, g);
      $('lab-status').textContent = 'Updated ' + new Date().toLocaleTimeString();
    } catch (e) {
      if (id === S.reqId) { $('lab-status').textContent = ''; SS.toast(e.message, 'error'); }
    } finally {
      if (id === S.reqId) $('lab-result').classList.remove('lab-busy');
    }
  }

  // ---------------------------------------------------------------- load clip
  async function loaded(res) {
    S.token = res.token; S.meta = res.meta; S.clean = null;
    const m = res.meta;
    $('lab-clipinfo').innerHTML = `Loaded <b>${SS.esc(m.filename)}</b> (${m.source}, ${m.duration}s${m.truncated ? ', first 20 s' : ''})` +
      (m.actual_class ? ` · true class <span class="badge">${SS.esc(m.actual_class)}</span>` : '');
    $('lab-clipinfo').classList.remove('hidden');
    $('lab-main').classList.remove('hidden');
    $('sweep-status').textContent = 'Not run yet.'; $('sweep-table').innerHTML = '';
    if (S.chart) { S.chart.destroy(); S.chart = null; }
    applyPreset({});
  }
  $('lab-upload').addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData(e.target);
    try { loaded(await SS.api(LAB.urls.load, { method: 'POST', body: fd })); } catch (err) { SS.toast(err.message, 'error'); }
  });
  const rnd = $('lab-random');
  if (rnd) rnd.addEventListener('click', async () => {
    rnd.disabled = true;
    try { loaded(await SS.api(LAB.urls.load, { method: 'POST', json: { class_label: $('lab-class').value } })); }
    catch (err) { SS.toast(err.message, 'error'); } finally { rnd.disabled = false; }
  });

  for (const s of Object.values(sliders)) s.el.addEventListener('input', () => { labels(); schedule(); });
  $('s-device').addEventListener('change', () => schedule(0));
  document.querySelectorAll('.lab-preset').forEach(b => b.addEventListener('click', () => applyPreset(JSON.parse(b.dataset.p))));
  labels();

  // ---------------------------------------------------------------- sweep
  document.querySelectorAll('[data-sweep]').forEach(btn => btn.addEventListener('click', async () => {
    if (!S.token) return;
    const kind = btn.dataset.sweep;
    $('sweep-status').textContent = 'Python model: classifying 7 versions…';
    try {
      const sw = await SS.api(LAB.urls.sweep, { method: 'POST', json: { token: S.token, params: params(), kind } });
      const ref = sw.reference_class;
      const gtmRef = [], gtmTop = [];
      for (const [i, p] of sw.points.entries()) {
        $('sweep-status').textContent = LAB.gtm ? `Teachable Machine: version ${i + 1} of ${sw.points.length}…` : '';
        let g = null;
        try { g = await gtmScores(p.audio_url); } catch (e) { console.error(e); }
        gtmRef.push(g ? (g[ref] || 0) : null); gtmTop.push(g ? top(g)[0][0] : '—');
      }
      const labelsX = sw.points.map(p => p.label);
      if (S.chart) S.chart.destroy();
      Chart.defaults.color = '#B3A6C9'; Chart.defaults.borderColor = 'rgba(212,196,232,.1)';
      S.chart = new Chart($('sweep-chart'), {
        type: 'line',
        data: { labels: labelsX, datasets: [
          { label: `Python – ${ref}`, data: sw.points.map(p => +(p.ref_confidence * 100).toFixed(1)), borderColor: '#D4C4E8', backgroundColor: '#D4C4E8', tension: .3 },
          ...(LAB.gtm ? [{ label: `GTM – ${ref}`, data: gtmRef.map(v => v == null ? null : +(v * 100).toFixed(1)), borderColor: '#E8A6D0', backgroundColor: '#E8A6D0', tension: .3 }] : []),
        ] },
        options: { scales: { y: { min: 0, max: 100, title: { display: true, text: `Confidence for ${ref} (%)` } },
                             x: { title: { display: true, text: kind === 'bg_snr' ? 'Real background noise (SNR)' : 'White noise (SNR)' } } },
                   plugins: { legend: { position: 'bottom' } } },
      });
      $('sweep-table').innerHTML = '<thead><tr><th>SNR</th><th>Python</th><th>GTM</th><th>Quality</th></tr></thead><tbody>' +
        sw.points.map((p, i) => `<tr><td class="whitespace-nowrap">${p.label}</td><td class="${p.prediction === ref ? '' : 'text-red-400'}">${SS.esc(p.prediction)} ${SS.pct(p.confidence)}</td>
          <td class="${gtmTop[i] === ref || gtmTop[i] === '—' ? '' : 'text-red-400'}">${SS.esc(gtmTop[i])}</td><td>${SS.qBadge(p.quality)}</td></tr>`).join('') + '</tbody>';
      const keep = sw.points.filter(p => p.prediction === ref).length;
      $('sweep-status').innerHTML = `Python kept <b>${SS.esc(ref)}</b> in ${keep} of ${sw.points.length} noise levels` +
        (LAB.gtm ? `, GTM in ${gtmTop.filter(c => c === ref).length} of ${sw.points.length}.` : '.');
    } catch (e) { $('sweep-status').textContent = e.message; }
  }));
})();
