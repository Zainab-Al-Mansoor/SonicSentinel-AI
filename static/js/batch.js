// Batch runner shared by "Batch upload" and the admin "Test-set evaluation".
// For each item: server analysis (validation, Python model) -> GTM on the same segments in the browser -> final decision.
const Batch = (() => {
  async function runGtm(ev, gtmAvailable) {
    if (ev.gtm_status !== 'pending' || !gtmAvailable) return ev;
    const post = body => SS.api(`/api/events/${encodeURIComponent(ev.audio_id)}/gtm`, { method: 'POST', json: body });
    try {
      if (!GTM.loaded) await GTM.load(GTM_BASE);
      const results = [];
      for (const seg of ev.segments) results.push(await GTM.classify(await GTM.decodeUrl(seg.url)));
      return await post({ segments: results });
    } catch (err) {
      return await post({ error: String(err.message || err) });
    }
  }

  function row(label, r) {
    if (r.error) return `<tr><td>${SS.esc(label)}</td><td colspan="8" class="text-red-400">${SS.esc(r.error)}</td></tr>`;
    const e = r.event || r;
    const truth = e.actual_class ? `<td>${SS.esc(e.actual_class)}</td>` : '';
    const ok = e.actual_class ? (e.final_category === e.actual_class ? '✔' : '✘') : '';
    return `<tr>
      <td><a class="text-brand-400 hover:underline font-mono text-xs" href="${e.url}" target="_blank">${e.audio_id}</a>
        <div class="text-xs text-slate-500">${SS.esc(label)}${r.duplicate ? ' · duplicate' : ''}</div></td>${truth}
      <td>${SS.esc(e.python_prediction || '—')} <span class="text-xs text-slate-400">${SS.pct(e.python_confidence)}</span></td>
      <td>${SS.esc(e.gtm_prediction || '—')} <span class="text-xs text-slate-400">${SS.pct(e.gtm_confidence)}</span></td>
      <td>${SS.csBadge(e.consistency_status)}</td>
      <td class="text-white">${SS.esc(e.final_category || '—')} ${ok}</td>
      <td>${SS.sevBadge(e.severity)}</td><td>${SS.qBadge(e.quality)}</td>
      <td class="text-xs">${SS.esc(e.alert_status || '')}${e.manual_review ? ' · review' : ''}</td></tr>`;
  }

  async function run(items, { tbody, progress, gtmAvailable }) {
    let done = 0;
    for (const it of items) {
      progress.textContent = `Processing ${done + 1} of ${items.length}: ${it.label}`;
      let r;
      try {
        r = await it.analyse();
        if (r.event) r.event = await runGtm(r.event, gtmAvailable) || r.event;
      } catch (err) { r = { error: err.message }; }
      tbody.insertAdjacentHTML('beforeend', row(it.label, r));
      done++;
    }
    progress.textContent = `Finished ${done} item(s).`;
  }
  return { run };
})();
