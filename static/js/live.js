// Live microphone monitoring: capture -> GTM (browser) -> POST window -> Python + rules (server) -> dashboard.
(() => {
  const SR = 44100;
  const $ = id => document.getElementById(id);
  let ctx = null, stream = null, node = null, source = null, session = null;
  let buffer = [], bufferLen = 0, paused = false, running = false, busy = false, count = 0;
  const windowLen = Math.round(LIVE.windowSec * SR);

  function setMic(state) {
    const colors = { 'Available': 'bg-emerald-500', 'Active': 'bg-red-500 animate-pulse', 'Paused': 'bg-amber-500',
                     'Disconnected': 'bg-slate-500', 'Permission denied': 'bg-red-800', 'Not available': 'bg-slate-600' };
    $('mic-status').textContent = state;
    $('mic-dot').className = 'w-3 h-3 rounded-full ' + (colors[state] || 'bg-slate-500');
    $('mic-banner').classList.toggle('hidden', state !== 'Active');
    document.body.style.paddingTop = state === 'Active' ? '28px' : '';
  }

  async function checkMic() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return setMic('Not available');
    try {
      if (navigator.permissions) {
        const p = await navigator.permissions.query({ name: 'microphone' });
        if (p.state === 'denied') return setMic('Permission denied');
        p.onchange = () => { if (p.state === 'denied') { stop(); setMic('Permission denied'); } };
      }
      const devs = await navigator.mediaDevices.enumerateDevices();
      setMic(devs.some(d => d.kind === 'audioinput') ? 'Available' : 'Disconnected');
    } catch (_) { setMic('Available'); }
  }

  async function start() {
    $('btn-start').disabled = true;
    try {
      if (LIVE.gtmAvailable && !GTM.loaded) {
        $('gtm-state').textContent = 'loading…';
        try {
          await GTM.load(GTM_BASE);
          $('gtm-state').textContent = `ready (${GTM.labels.length} classes)`;
        } catch (err) {   // keep monitoring with the Python model; results will be "Uncertain Result"
          $('gtm-state').textContent = 'failed – ' + err.message;
          SS.toast('GTM model could not be loaded: ' + err.message, 'warning');
        }
      }
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    } catch (err) {
      $('btn-start').disabled = false;
      if (err.name === 'NotAllowedError') { setMic('Permission denied'); SS.toast('Microphone permission was denied.', 'error'); }
      else { SS.toast('Could not start: ' + err.message, 'error'); }
      return;
    }
    stream.getAudioTracks()[0].onended = () => { stop(); setMic('Disconnected'); SS.toast('Microphone disconnected.', 'warning'); };
    ctx = new AudioContext({ sampleRate: SR });
    source = ctx.createMediaStreamSource(stream);
    node = ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = e => {
      const d = e.inputBuffer.getChannelData(0);
      let peak = 0; for (let i = 0; i < d.length; i++) peak = Math.max(peak, Math.abs(d[i]));
      $('meter').style.width = Math.min(100, peak * 140) + '%';
      if (paused) return;
      buffer.push(new Float32Array(d)); bufferLen += d.length;
      if (bufferLen >= windowLen) flushWindow();
    };
    const mute = ctx.createGain(); mute.gain.value = 0;
    source.connect(node); node.connect(mute); mute.connect(ctx.destination);
    session = (await SS.api(LIVE.urls.start, { method: 'POST' })).session;
    $('sess').textContent = session;
    running = true; paused = false; setMic('Active');
    $('btn-pause').disabled = false; $('btn-stop').disabled = false;
    $('recent').innerHTML = '';
  }

  function flushWindow() {
    const win = new Float32Array(windowLen);
    let off = 0;
    while (off < windowLen && buffer.length) {
      const b = buffer[0], take = Math.min(b.length, windowLen - off);
      win.set(b.subarray(0, take), off); off += take;
      if (take < b.length) buffer[0] = b.subarray(take); else buffer.shift();
    }
    bufferLen -= windowLen;
    if (busy) return;              // drop a window rather than fall behind real time
    analyse(win);
  }

  async function analyse(win) {
    busy = true;
    const t0 = performance.now();
    const fd = new FormData();
    fd.append('session', session);
    fd.append('audio', encodeWav(win, SR), 'window.wav');
    try {
      if (LIVE.gtmAvailable && GTM.loaded) {
        // GTM classifies the window BEFORE the server sees it – fully independent of the Python model.
        fd.append('gtm', JSON.stringify(await GTM.classify(win)));
      }
    } catch (err) { fd.append('gtm_error', String(err.message || err)); }
    try {
      const r = await SS.api(LIVE.urls.window, { method: 'POST', body: fd });
      $('latency').textContent = Math.round(performance.now() - t0) + ' ms';
      $('count').textContent = ++count;
      render(r);
    } catch (err) { SS.toast(err.message, 'error'); }
    busy = false;
  }

  function render(r) {
    if (r.silent) { $('cur-cat').textContent = 'Silence'; $('cur-conf').textContent = '—'; $('cur-q').innerHTML = SS.qBadge(r.quality); return; }
    $('cur-cat').textContent = r.final_category;
    $('cur-conf').textContent = SS.pct(r.final_confidence);
    $('cur-sev').innerHTML = SS.sevBadge(r.severity);
    $('cur-cs').innerHTML = SS.csBadge(r.consistency_status);
    $('cur-q').innerHTML = SS.qBadge(r.quality);
    $('cur-status').textContent = r.status;
    $('cur-py').textContent = `${r.python_prediction} · ${SS.pct(r.python_confidence)}`;
    $('cur-gtm').textContent = r.gtm_prediction ? `${r.gtm_prediction} · ${SS.pct(r.gtm_confidence)}` : `unavailable (${r.gtm_status})`;
    if (r.alert) showAlert(r.alert, r.url);
    if (r.final_category !== 'Background Noise') {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td class="text-xs whitespace-nowrap">${new Date().toLocaleTimeString()}</td>
        <td><a class="text-brand-400 hover:underline" href="${r.url}" target="_blank">${SS.esc(r.final_category)}</a></td>
        <td>${SS.pct(r.final_confidence)}</td><td class="text-xs">${SS.esc(r.python_prediction)}</td>
        <td class="text-xs">${SS.esc(r.gtm_prediction || '—')}</td><td>${SS.csBadge(r.consistency_status)}</td>
        <td>${SS.sevBadge(r.severity)}</td><td class="text-xs">${SS.esc(r.alert_status)}${r.manual_review ? ' · review' : ''}</td>`;
      $('recent').prepend(tr);
      while ($('recent').children.length > 30) $('recent').lastChild.remove();
    }
  }

  function showAlert(a, url) {
    $('active-alert').classList.remove('hidden');
    $('alert-msg').textContent = '🔔 ' + a.message;
    $('alert-action').textContent = a.action;
    SS.toast('🔔 ' + a.message, a.severity, url, 12000);
    const box = $('alert-buttons'); box.innerHTML = '';
    if (!LIVE.canHandleAlerts) return;
    [['acknowledge', 'btn-primary', 'Acknowledge'], ['escalate', 'btn-danger', 'Escalate'], ['dismiss', 'btn-ghost', 'Dismiss']]
      .forEach(([act, cls, label]) => {
        const b = document.createElement('button'); b.className = `btn btn-sm ${cls}`; b.textContent = label;
        b.onclick = async () => {
          await SS.api(LIVE.urls.alertAct.replace('/0/', `/${a.id}/`).replace('ACTION', act), { method: 'POST' });
          $('active-alert').classList.add('hidden'); SS.toast(`Alert ${act}d`, 'success');
        };
        box.appendChild(b);
      });
  }

  async function stop() {
    if (!running) return;
    running = false;
    try { node && node.disconnect(); source && source.disconnect(); } catch (_) {}
    stream && stream.getTracks().forEach(t => t.stop());
    ctx && ctx.close();
    buffer = []; bufferLen = 0;
    await SS.api(LIVE.urls.stop, { method: 'POST', json: { session } }).catch(() => {});
    setMic('Available'); $('meter').style.width = '0';
    $('btn-start').disabled = false; $('btn-pause').disabled = true; $('btn-stop').disabled = true;
  }

  $('btn-start').onclick = start;
  $('btn-stop').onclick = stop;
  $('btn-pause').onclick = () => {
    paused = !paused; buffer = []; bufferLen = 0;
    $('btn-pause').textContent = paused ? '▶ Resume' : '⏸ Pause';
    setMic(paused ? 'Paused' : 'Active');
  };
  window.addEventListener('beforeunload', () => { if (running) navigator.sendBeacon && stop(); });
  checkMic();
})();
