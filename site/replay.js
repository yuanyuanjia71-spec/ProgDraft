/* Shared-clock replay of real round timestamps. No token generation or timing normalization.
   Website player: CC BY-SA 4.0; see SITE_LICENSE.md. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const escape = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let data, duration = 0, elapsed = 0, playing = false, lastTick = 0, frame = null;
  let rate = Number($('rate').value);
  const previousRound = {anchor: -1, ours: -1};
  if (new URLSearchParams(location.search).has('record')) document.body.classList.add('recording');

  function waveform(event) {
    const values = data.audio.envelope;
    const audioDuration = data.audio.duration;
    const maximum = Math.max(...values, 1e-8);
    const x = i => 16 + i * 468 / Math.max(1, values.length - 1);
    const top = values.map((v, i) => `${x(i)},${48 - 17*v/maximum}`);
    const bottom = values.map((v, i) => `${x(i)},${48 + 17*v/maximum}`).reverse();
    let content = `<polygon points="${top.concat(bottom).join(' ')}" fill="#33465b"/>`;
    for (let i=0; i<=4; i++) content += `<text x="${16+468*i/4}" y="87" text-anchor="middle" fill="#7589a1" font-size="9">${(audioDuration*i/4).toFixed(1)}s</text>`;
    if (event?.positions.length) {
      const points = event.centers.map((a,i) => `${16+468*Math.max(0,Math.min(audioDuration,a))/audioDuration},${8+i*3}`);
      content += `<polyline points="${points.join(' ')}" fill="none" stroke="#5eead4" stroke-width="1"/>`;
      event.centers.forEach((a,i) => {
        const px = 16 + 468*Math.max(0,Math.min(audioDuration,a))/audioDuration;
        const py = 8+i*3;
        const color = i<event.accepted ? '#4ade80' : i===event.first_rejected ? '#fb7185' : '#8193ac';
        content += `<line x1="${px}" x2="${px}" y1="${py}" y2="71" stroke="${color}" opacity=".65"/><circle cx="${px}" cy="${py}" r="2.8" fill="${color}"><title>d${i+1}: raw ${event.positions[i].toFixed(4)}s, center ${a.toFixed(4)}s</title></circle><text x="${px+4}" y="${py-1}" fill="${color}" font-size="8">${i+1}</text>`;
      });
    }
    return `<svg viewBox="0 0 500 94" role="img" aria-label="Audio waveform${event ? ' and current acoustic progress' : ''}">${content}</svg>`;
  }

  function updateMethod(name) {
    const method = data.methods[name];
    const finish = method.observed.decode_s;
    const complete = elapsed >= finish;
    const rounds = method.rounds;
    let index = -1;
    for (let i=0; i<rounds.length && rounds[i].observed_decode_s <= elapsed; i++) index=i;
    const event = rounds[index];
    $(name+'-time').innerHTML = `${Math.min(elapsed,finish).toFixed(3)}<span>s</span>`;
    $(name+'-status').textContent = complete ? 'FINISHED' : elapsed ? 'DECODING' : 'READY';
    $(name+'-status').classList.toggle('done',complete);
    $(name+'-round').textContent = complete ? `${rounds.length} rounds · finished` : event ? `Round ${event.round} committed` : elapsed ? 'Target prompt prefill…' : 'Waiting to start';
    $(name+'-exact').textContent = complete ? '✓ Final token IDs exactly match Target-only AR' : 'Final token-ID check pending';
    $(name+'-exact').classList.toggle('verified',complete);
    if (previousRound[name] === index) return;
    previousRound[name] = index;
    $(name+'-transcript').textContent = event?.text || 'Both methods start together.';
    $(name+'-transcript').scrollTop = $(name+'-transcript').scrollHeight;
    $(name+'-accepted').textContent = event ? `${event.accepted} / 8 accepted` : '— / 8 accepted';
    $(name+'-tokens').innerHTML = Array.from({length:8},(_,i) => {
      const state = event && i<event.accepted ? 'accepted' : event && i===event.first_rejected ? 'rejected' : 'unused';
      const piece = event ? event.pieces[i] || '∅' : '—';
      return `<div class="token ${state}" title="${escape(piece)}"><small>d${i+1}</small><code>${escape(piece)}</code></div>`;
    }).join('');
    const extra = event ? event.emitted.length - event.accepted : 0;
    $(name+'-extra').innerHTML = extra ? `${event.accepted===8?'Bonus':'Correction'} from target: <code>${escape(event.emitted_pieces.at(-1))}</code>` : event?.terminal ? 'Accepted EOS · no additional bonus token' : 'Target correction / bonus appears here.';
    $(name+'-wave').innerHTML = waveform(name==='ours' ? event : null);
    $(name+'-count').textContent = event?.accepted_sum || 0;
    $(name+'-rounds').textContent = index+1;
    $(name+'-tau').textContent = event ? (event.tokens.length/event.round).toFixed(3) : '—';
    $(name+'-bar').style.width = `${event ? 100*event.tokens.length/data.reference.tokens.length : 0}%`;
    if (name==='ours') $('progress-label').textContent = event ? `round ${event.round} · a₁=${event.positions[0].toFixed(2)}s → a₈=${event.positions.at(-1).toFixed(2)}s` : 'a₁ → a₂ → … → a₈';
  }

  function render() {
    $('clock').innerHTML = `${elapsed.toFixed(3)} <small>s</small>`;
    $('seek').value = elapsed;
    updateMethod('anchor'); updateMethod('ours');
    $('playback-note').textContent = `Both methods: ${rate}× playback${rate<1 ? ` (${1/rate}× slower)` : ' (real time)'}`;
    $('play').textContent = playing ? 'Ⅱ Pause both' : elapsed >= duration ? '↺ Replay together' : '▶ Play together';
  }
  function tick(now) {
    if (!playing) return;
    elapsed = Math.min(duration, elapsed + (now-lastTick)/1000*rate);
    lastTick=now;
    if (elapsed >= duration) playing=false;
    render();
    if (playing) frame=requestAnimationFrame(tick);
  }
  function play() {
    if (!data) return;
    if (frame!==null) cancelAnimationFrame(frame);
    if (elapsed >= duration) elapsed=0;
    playing=true; lastTick=performance.now(); render(); frame=requestAnimationFrame(tick);
  }
  function pause() {playing=false;if(frame!==null)cancelAnimationFrame(frame);frame=null;render();}
  $('play').onclick=()=> playing ? pause() : play();
  $('restart').onclick=()=> {pause();elapsed=0;render();};
  $('seek').oninput=()=> {const next=Number($('seek').value);pause();elapsed=next;render();};
  $('rate').onchange=()=> {rate=Number($('rate').value);lastTick=performance.now();render();};
  // A hidden tab should not silently skip the comparison when the viewer returns.
  document.addEventListener('visibilitychange',()=> {if(document.hidden && playing) pause();});

  async function init() {
    const response = await fetch('assets/demo-trace.json');
    if (!response.ok) throw new Error(`Trace unavailable (${response.status})`);
    data = await response.json();
    if (data.k!==8 || !data.methods.anchor.exact || !data.methods.ours.exact) throw new Error('Invalid comparison contract');
    for (const name of ['anchor','ours']) {
      const method = data.methods[name];
      if (JSON.stringify(method.observed.tokens)!==JSON.stringify(data.reference.tokens) || JSON.stringify(method.measured.tokens)!==JSON.stringify(data.reference.tokens)) throw new Error('Token-ID consistency check failed');
      let previous=0;
      for (const event of method.rounds) {
        if (event.observed_decode_s<previous || event.observed_decode_s>method.observed.decode_s) throw new Error('Invalid original timeline');
        if (JSON.stringify(event.tokens)!==JSON.stringify(data.reference.tokens.slice(0,event.tokens.length))) throw new Error('Committed prefix mismatch');
        previous=event.observed_decode_s;
      }
    }
    duration=Math.max(data.methods.anchor.observed.decode_s,data.methods.ours.observed.decode_s);
    $('seek').max=duration;
    $('run-context').textContent=`${data.provenance.target} · ${data.audio.duration.toFixed(2)}s LibriSpeech audio · Greedy · ${data.gpu} · Same GPU, independent runs`;
    let rows=`<tr><td>Target-only AR</td><td>${data.reference.decode_s.toFixed(3)} s</td><td>1.000×</td><td>—</td><td>Reference</td></tr>`;
    for (const name of ['anchor','ours']) {
      const m=data.methods[name];
      rows+=`<tr><td>${name==='anchor'?'AnchorDraft':'Ours · ProgDraft'}</td><td>${m.measured.decode_s.toFixed(3)} s</td><td>${m.speedup.toFixed(3)}×</td><td>${m.measured.mean_accepted.toFixed(3)}</td><td>Exact ✓</td></tr>`;
    }
    $('measurements').innerHTML=rows;
    $('provenance').textContent=`Sample ${data.provenance.sample_id}. Recorded ${data.provenance.created_utc.slice(0,10)}. ${data.provenance.anchor}. Ours: ${data.provenance.ours}.`;
    for (const id of ['play','restart','seek','rate']) $(id).disabled=false;
    previousRound.anchor=-2; previousRound.ours=-2;render();
    // Deterministic controls for browser checks / video capture; read-only model data.
    window.progdraftReplay={play,pause,seek:(t)=>{pause();elapsed=Math.max(0,Math.min(duration,t));render();},setRate:(r)=>{if(![1,.25,.1,.05].includes(r))throw new Error('Unsupported playback rate');rate=r;$('rate').value=r;lastTick=performance.now();render();},get data(){return data;},get time(){return elapsed;},get playing(){return playing;}};
  }
  init().catch(error=>{$('run-context').textContent=`Replay could not load: ${error.message}. The video above remains available.`;console.error(error);});
})();
