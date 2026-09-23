// Shared helpers for every page.
const SS = (() => {
  const csrf = () => document.querySelector('meta[name="csrf-token"]').content;

  async function api(url, opts = {}) {
    opts.headers = Object.assign({ 'X-CSRFToken': csrf(), 'X-Requested-With': 'fetch' }, opts.headers || {});
    if (opts.json !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(opts.json);
      delete opts.json;
    }
    const r = await fetch(url, opts);
    let data = null;
    try { data = await r.json(); } catch (_) { /* non-JSON */ }
    if (!r.ok) throw new Error((data && data.error) || `Request failed (${r.status})`);
    return data;
  }

  function toast(msg, kind = 'info', href = null, ms = 6000) {
    const colors = { info: 'border-sky-600 bg-slate-900', error: 'border-red-600 bg-red-950',
                     Critical: 'border-red-500 bg-red-950 alert-pulse', High: 'border-orange-500 bg-orange-950',
                     success: 'border-emerald-600 bg-emerald-950', warning: 'border-amber-600 bg-amber-950' };
    const el = document.createElement(href ? 'a' : 'div');
    if (href) el.href = href;
    el.className = `block border-l-4 rounded-lg p-3 text-sm shadow-xl ${colors[kind] || colors.info}`;
    el.textContent = msg;
    document.getElementById('toast-root').appendChild(el);
    setTimeout(() => el.remove(), ms);
  }

  const pct = v => (v === null || v === undefined) ? '—' : (v * 100).toFixed(1) + '%';
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const cls = s => String(s || '').replace(/ /g, '-');
  const sevBadge = s => s ? `<span class="badge sev-${cls(s)}">${esc(s)}</span>` : '';
  const qBadge = q => q ? `<span class="badge q-${cls(q)}">${esc(q)}</span>` : '';
  const csBadge = c => c ? `<span class="badge cs-${cls(c)}">${esc(c)}</span>` : '';

  let lastAlertId = null;
  function startAlertPolling(url) {
    const tick = async () => {
      try {
        const d = await api(url);
        if (d.latest && lastAlertId !== null && d.latest.id !== lastAlertId) {
          toast('🔔 ' + d.latest.message, d.latest.severity, d.latest.url, 10000);
        }
        if (d.latest) lastAlertId = d.latest.id; else lastAlertId = 0;
      } catch (_) { /* offline – ignore */ }
    };
    tick();
    setInterval(tick, 8000);
  }

  // mobile nav
  document.addEventListener('DOMContentLoaded', () => {
    const t = document.getElementById('nav-toggle');
    if (t) t.onclick = () => document.getElementById('nav-links').classList.toggle('hidden');
  });

  return { api, toast, pct, esc, sevBadge, qBadge, csBadge, startAlertPolling, csrf };
})();
