/* SonicSentinel AI – light, professional motion layer.
   - cards, tiles and tables fade up when they scroll into view (staggered)
   - KPI numbers count up from 0
   - quick-action tiles get a soft glow that follows the mouse
   Everything is skipped when the user prefers reduced motion. */
(function () {
  'use strict';
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduce || !('IntersectionObserver' in window)) return;

  // ---- scroll reveal ---------------------------------------------------------
  var sel = '.page-sheet .card, .page-sheet .stat, .page-sheet .class-tile, .page-sheet .flash, .page-sheet .page-head';
  var items = Array.prototype.slice.call(document.querySelectorAll(sel));
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      e.target.classList.add('in');
      io.unobserve(e.target);
      // drop the helper class after the transition so hover transforms work normally
      setTimeout(function () { e.target.classList.remove('reveal', 'in'); e.target.style.removeProperty('--rd'); }, 1400);
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -30px 0px' });

  var groups = new Map();
  items.forEach(function (el) {
    if (el.closest('.hero')) return;
    var parent = el.parentElement, i = groups.get(parent) || 0;
    groups.set(parent, i + 1);
    el.style.setProperty('--rd', Math.min(i, 8) * 0.06 + 's');
    el.classList.add('reveal');
    io.observe(el);
  });

  // ---- count-up numbers ------------------------------------------------------
  function countUp(el) {
    var txt = el.textContent.trim();
    if (!/^\d[\d,]*$/.test(txt)) return;            // only plain integers
    var target = parseInt(txt.replace(/,/g, ''), 10);
    if (!target) return;
    var start = null, dur = Math.min(1600, 600 + target * 2), comma = txt.indexOf(',') > -1;
    function fmt(v) { return comma ? v.toLocaleString() : String(v); }
    function step(ts) {
      if (start === null) start = ts;
      var p = Math.min(1, (ts - start) / dur), eased = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(Math.round(target * eased));
      if (p < 1) requestAnimationFrame(step); else el.textContent = txt;
    }
    el.textContent = fmt(0);
    requestAnimationFrame(step);
  }
  var nums = document.querySelectorAll('.kpi-value, .stat-value, .class-tile-count');
  var nio = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { if (e.isIntersecting) { countUp(e.target); nio.unobserve(e.target); } });
  }, { threshold: 0.4 });
  Array.prototype.forEach.call(nums, function (n) { nio.observe(n); });

  // ---- mouse-follow glow on quick-action tiles --------------------------------
  Array.prototype.forEach.call(document.querySelectorAll('.qtile'), function (t) {
    t.addEventListener('mousemove', function (ev) {
      var r = t.getBoundingClientRect();
      t.style.setProperty('--mx', (ev.clientX - r.left) + 'px');
      t.style.setProperty('--my', (ev.clientY - r.top) + 'px');
    });
  });
})();
