const GTM = (() => {
  const SR = 44100, FFT = 2048, HOP = 1024;
  let recognizer = null, labels = [], frames = 43, cols = 232;

  async function load(baseUrl) {
    if (recognizer) return { labels };
    if (typeof speechCommands === 'undefined') throw new Error('speech-commands library not loaded');
    baseUrl = new URL(baseUrl, window.location.href).href;
    recognizer = speechCommands.create('BROWSER_FFT', undefined, baseUrl + 'model.json', baseUrl + 'metadata.json');
    await recognizer.ensureModelLoaded();
    labels = recognizer.wordLabels();
    const shape = recognizer.modelInputShape();
    frames = shape[1]; cols = shape[2];
    return { labels, frames, cols };
  }


  const win = new Float32Array(FFT);
  for (let n = 0; n < FFT; n++)
    win[n] = 0.42 - 0.5 * Math.cos(2 * Math.PI * n / FFT) + 0.08 * Math.cos(4 * Math.PI * n / FFT);
  const rev = new Uint32Array(FFT);
  for (let i = 0, bits = Math.log2(FFT); i < FFT; i++) {
    let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r;
  }
  const cosT = new Float32Array(FFT / 2), sinT = new Float32Array(FFT / 2);
  for (let i = 0; i < FFT / 2; i++) { cosT[i] = Math.cos(2 * Math.PI * i / FFT); sinT[i] = -Math.sin(2 * Math.PI * i / FFT); }

  function magnitudeDb(frame) {
    const re = new Float32Array(FFT), im = new Float32Array(FFT);
    for (let i = 0; i < FFT; i++) re[rev[i]] = frame[i] * win[i];
    for (let size = 2; size <= FFT; size <<= 1) {
      const half = size >> 1, step = FFT / size;
      for (let s = 0; s < FFT; s += size) {
        for (let j = 0; j < half; j++) {
          const k = j * step, a = s + j, b = a + half;
          const tr = re[b] * cosT[k] - im[b] * sinT[k];
          const ti = re[b] * sinT[k] + im[b] * cosT[k];
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        }
      }
    }
    const out = new Float32Array(cols);
    for (let k = 0; k < cols; k++) {
      const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]) / FFT;
      out[k] = 20 * Math.log10(Math.max(mag, 1e-10));
    }
    return out;
  }

  function spectrogram(samples, start) {
    const x = new Float32Array(frames * cols);
    const buf = new Float32Array(FFT);
    for (let f = 0; f < frames; f++) {
      const off = start + f * HOP;
      for (let i = 0; i < FFT; i++) { const idx = off + i; buf[i] = idx < samples.length ? samples[idx] : 0; }
      x.set(magnitudeDb(buf), f * cols);
    }

    let mean = 0; for (const v of x) mean += v; mean /= x.length;
    let v2 = 0; for (const v of x) v2 += (v - mean) ** 2; const std = Math.sqrt(v2 / x.length) + 1e-7;
    for (let i = 0; i < x.length; i++) x[i] = (x[i] - mean) / std;
    return x;
  }

  async function classifyAt(samples, start) {
    const out = await recognizer.recognize(spectrogram(samples, start));
    const scores = {};
    labels.forEach((l, i) => scores[l] = out.scores[i]);
    return scores;
  }


  async function classify(samples) {
    const span = frames * HOP;
    const starts = [];
    if (samples.length <= span) starts.push(0);
    else for (let s = 0; s + span <= samples.length; s += Math.floor(span / 2)) starts.push(s);
    const acc = {};
    for (const s of starts) {
      const sc = await classifyAt(samples, s);
      for (const k in sc) acc[k] = (acc[k] || 0) + sc[k] / starts.length;
    }
    return acc;
  }


  async function decodeUrl(url) {
    const buf = await (await fetch(url)).arrayBuffer();
    const ctx = new OfflineAudioContext(1, SR, SR);
    const audio = await ctx.decodeAudioData(buf);
    return audio.getChannelData(0);
  }

  return { load, classify, decodeUrl, get labels() { return labels; }, get loaded() { return !!recognizer; } };
})();
