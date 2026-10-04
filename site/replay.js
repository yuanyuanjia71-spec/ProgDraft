/* Shared-clock replay of real round timestamps. No token generation or timing normalization.
   Website player: CC BY-SA 4.0; see SITE_LICENSE.md. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const escape = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let data, duration = 0, elapsed = 0, playing = false, lastTick = 0, frame = null;
  let rate = Number($('rate').value);
  const previousRound = {ar: -1, anchor: -1, ours: -1};
  if (new URLSearchParams(location.search).has('record')) document.body.classList.add('recording');

  function waveform(event) {
    const values = data.audio.envelope;
    const audioDuration = data.audio.duration;
    const maximum = Math.max(...values, 1e-8);
    const x = i => 16 + i * 468 / Math.max(1, values.length - 1);
    const top = values.map((v, i) => `${x(i)},${48 - 17*v/maximum}`);
    const bottom = values.map((v, i) => `${x(i)},${48 + 17*v/maximum}`).reverse();
    let content = `<polygon points="${top.concat(bottom).join(' ')}" fill="#c9d8d0"/>`;
    for (let i=0; i<=4; i++) content += `<text x="${16+468*i/4}" y="87" text-anchor="middle" fill="#66796e" font-size="11">${(audioDuration*i/4).toFixed(1)}s</text>`;
    if (event?.positions.length) {
      const points = event.centers.map((a,i) => `${16+468*Math.max(0,Math.min(audioDuration,a))/audioDuration},${8+i*3}`);
      content += `<polyline points="${points.join(' ')}" fill="none" stroke="#147568" stroke-width="1"/>`;
      event.centers.forEach((a,i) => {
        const px = 16 + 468*Math.max(0,Math.min(audioDuration,a))/audioDuration;
        const py = 8+i*3;
        const color = i<event.accepted ? '#22704a' : i===event.first_rejected ? '#ad4452' : '#697c71';
        content += `<line x1="${px}" x2="${px}" y1="${py}" y2="71" stroke="${color}" opacity=".65"/><circle cx="${px}" cy="${py}" r="2.8" fill="${color}"><title>d${i+1}: raw ${event.positions[i].toFixed(4)}s, center ${a.toFixed(4)}s</title></circle><text x="${px+4}" y="${py-1}" fill="${color}" font-size="10">${i+1}</text>`;
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
    $(name+'-status').textContent = complete ? 'Finished' : playing ? 'Decoding' : elapsed ? 'Paused' : 'Ready';
    $(name+'-status').classList.toggle('done',complete);
    $(name+'-round').textContent = complete ? `${rounds.length} rounds` : event ? `Round ${event.round}` : elapsed ? 'Preparing…' : 'Ready to start';
    $(name+'-exact').textContent = complete ? '✓ Exact token match' : '';
    $(name+'-exact').classList.toggle('verified',complete);
    if (!event) $(name+'-transcript').textContent = elapsed > 0 ? 'Preparing the transcript…' : 'Press Start demo to begin.';
    if (previousRound[name] === index) return;
    previousRound[name] = index;
    $(name+'-transcript').textContent = event?.text || (elapsed > 0 ? 'Preparing the transcript…' : 'Press Start demo to begin.');
    $(name+'-transcript').classList.toggle('is-empty', !event?.text);
    $(name+'-transcript').scrollTop = $(name+'-transcript').scrollHeight;
    $(name+'-accepted').textContent = event ? `${event.accepted} / 8 accepted` : '— / 8 accepted';
    $(name+'-tokens').innerHTML = Array.from({length:8},(_,i) => {
      const state = event && i<event.accepted ? 'accepted' : event && i===event.first_rejected ? 'rejected' : 'unused';
      const piece = event ? event.pieces[i] || '∅' : '—';
      const label = state==='accepted' ? 'accepted' : state==='rejected' ? 'rejected' : 'unverified';
      const mark = state==='accepted' ? '✓' : state==='rejected' ? '×' : '·';
      return `<div class="token ${state}" title="Draft ${i+1}: ${escape(piece)} (${label})" aria-label="Draft ${i+1}: ${escape(piece)}, ${label}"><code>${escape(piece)}</code><span class="token-mark" aria-hidden="true">${event ? mark : ''}</span></div>`;
    }).join('');
    const extra = event ? event.emitted.length - event.accepted : 0;
    $(name+'-extra').innerHTML = extra ? `${event.accepted===8?'Bonus':'Correction'}: <code>${escape(event.emitted_pieces.at(-1))}</code>` : event?.terminal ? 'End of transcript' : '';
    $(name+'-wave').innerHTML = waveform(name==='ours' ? event : null);
    $(name+'-count').textContent = event?.accepted_sum || 0;
    $(name+'-bar').style.width = `${event ? 100*event.tokens.length/data.reference.tokens.length : 0}%`;
    if (name==='ours') $('progress-label').textContent = event ? `${event.positions[0].toFixed(1)} → ${event.positions.at(-1).toFixed(1)} s` : 'Across 8 draft steps';
  }

  function updateTarget() {
    const method=data.methods.ar;
    if (!method) return;
    const finish=method.observed.decode_s;
    const complete=elapsed>=finish;
    let index=-1;
    for (let i=0; i<method.rounds.length && method.rounds[i].observed_decode_s<=elapsed; i++) index=i;
    const event=method.rounds[index];
    $('target-latency').innerHTML=`${Math.min(elapsed,finish).toFixed(3)}<span>s</span>`;
    $('target-status').textContent=complete ? 'Finished' : playing ? 'Decoding' : elapsed ? 'Paused' : 'Ready';
    $('target-status').classList.toggle('done',complete);
    $('target-bar').style.width=`${event ? 100*event.tokens.length/data.reference.tokens.length : 0}%`;
    if (previousRound.ar===index) return;
    previousRound.ar=index;
    $('target-token-count').textContent=event?.tokens.length || 0;
    $('target-transcript').textContent=event?.text || (elapsed ? 'Preparing the transcript…' : 'Press Start demo to begin.');
    $('target-transcript').classList.toggle('is-empty',!event?.text);
    $('target-transcript').scrollTop=$('target-transcript').scrollHeight;
  }

  function render() {
    $('clock').textContent = elapsed.toFixed(3);
    $('seek').setAttribute('aria-valuetext', `${elapsed.toFixed(3)} of ${duration.toFixed(3)} seconds`);
    $('seek').value = elapsed;
    updateTarget(); updateMethod('anchor'); updateMethod('ours');
    $('playback-note').textContent = `${rate}× speed${rate<1 ? ` · ${1/rate}× slower` : ' · real time'}`;
    $('play').textContent = playing ? 'Ⅱ Pause' : elapsed >= duration ? '↺ Replay demo' : elapsed > 0 ? '▶ Resume' : '▶ Start demo';
    $('play').setAttribute('aria-label', playing ? 'Pause demo' : elapsed >= duration ? 'Replay demo' : elapsed > 0 ? 'Resume demo' : 'Start demo');
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
  // Range inputs may round the final decimal of max; still reach the true finish.
  $('seek').oninput=()=> {const next=Number($('seek').value);pause();elapsed=duration-next<1e-9 ? duration : next;render();};
  $('rate').onchange=()=> {rate=Number($('rate').value);lastTick=performance.now();render();};
  // A hidden tab should not silently skip the comparison when the viewer returns.
  document.addEventListener('visibilitychange',()=> {if(document.hidden && playing) pause();});

  async function init() {
    const response = await fetch('assets/demo-trace.json');
    if (!response.ok) throw new Error(`Trace unavailable (${response.status})`);
    data = await response.json();
    if (data.k!==8 || !data.methods.anchor.exact || !data.methods.ours.exact) throw new Error('Invalid comparison contract');
    if (data.methods.ar) {
      const ar=data.methods.ar;
      if (!ar.exact || JSON.stringify(ar.observed.tokens)!==JSON.stringify(data.reference.tokens) || JSON.stringify(ar.measured.tokens)!==JSON.stringify(data.reference.tokens)) throw new Error('Target-only token-ID consistency check failed');
      let previous=0;
      for (const event of ar.rounds) {
        if (event.observed_decode_s<previous || event.observed_decode_s>ar.observed.decode_s) throw new Error('Invalid Target-only timeline');
        if (JSON.stringify(event.tokens)!==JSON.stringify(data.reference.tokens.slice(0,event.tokens.length))) throw new Error('Target-only committed prefix mismatch');
        previous=event.observed_decode_s;
      }
      if (!ar.rounds.length || JSON.stringify(ar.rounds.at(-1).tokens)!==JSON.stringify(data.reference.tokens)) throw new Error('Incomplete Target-only trace');
    }
    for (const name of ['anchor','ours']) {
      const method = data.methods[name];
      if (JSON.stringify(method.observed.tokens)!==JSON.stringify(data.reference.tokens) || JSON.stringify(method.measured.tokens)!==JSON.stringify(data.reference.tokens)) throw new Error('Token-ID consistency check failed');
      let previous=0;
      for (const event of method.rounds) {
        if (event.observed_decode_s<previous || event.observed_decode_s>method.observed.decode_s) throw new Error('Invalid original timeline');
        if (JSON.stringify(event.tokens)!==JSON.stringify(data.reference.tokens.slice(0,event.tokens.length))) throw new Error('Committed prefix mismatch');
        previous=event.observed_decode_s;
      }
      if (!method.rounds.length || JSON.stringify(method.rounds.at(-1).tokens)!==JSON.stringify(data.reference.tokens)) throw new Error('Incomplete speculative trace');
    }
    duration=Math.max(data.methods.anchor.observed.decode_s,data.methods.ours.observed.decode_s,data.methods.ar?.observed.decode_s || 0);
    $('target-latency').innerHTML=`${data.reference.decode_s.toFixed(3)}<span>s</span>`;
    $('target-transcript').textContent=data.reference.text;
    $('target-token-count').textContent=data.reference.tokens.length;
    $('target-bar').style.width='100%';
    $('target-wave').innerHTML=waveform(null);
    if (data.methods.ar) {
      $('target-time-label').textContent='Observed decode clock';
      $('target-note').textContent=`Per-token timestamps from an independent recorded run. Callback-free baseline: ${data.reference.decode_s.toFixed(3)} s.`;
      $('target-status').textContent='Ready';
      $('target-status').classList.remove('done');
      $('demo-caption').textContent='Three independent runs on the same GPU, replayed from one zero time using their recorded event timestamps.';
      $('timeline-detail').textContent='All three replay traces use their original recorded timestamps. The latency table uses separate callback-free runs and does not rescale the replay.';
    }
    $('seek').max=duration;
    $('duration').textContent=duration.toFixed(3);
    $('run-context').textContent=`${data.provenance.target} · ${data.audio.duration.toFixed(2)}s LibriSpeech audio · Greedy · ${data.gpu} · Same GPU, independent runs`;
    let rows=`<tr><td>Target-only AR</td><td>${data.reference.decode_s.toFixed(3)} s</td><td>1.000×</td><td>—</td><td>Reference</td></tr>`;
    for (const name of ['anchor','ours']) {
      const m=data.methods[name];
      rows+=`<tr><td>${name==='anchor'?'AnchorDraft':'Ours · ProgDraft'}</td><td>${m.measured.decode_s.toFixed(3)} s</td><td>${m.speedup.toFixed(3)}×</td><td>${m.measured.mean_accepted.toFixed(3)}</td><td>Exact ✓</td></tr>`;
    }
    $('measurements').innerHTML=rows;
    $('provenance').textContent=`Sample ${data.provenance.sample_id}. Recorded ${data.provenance.created_utc.slice(0,10)}. ${data.provenance.anchor}. Ours: ${data.provenance.ours}.`;
    for (const id of ['play','restart','seek','rate']) $(id).disabled=false;
    $('load-status').textContent='';
    $('race').setAttribute('aria-busy','false');
    previousRound.ar=-2; previousRound.anchor=-2; previousRound.ours=-2;render();
    // Deterministic controls for browser checks / video capture; read-only model data.
    window.progdraftReplay={play,pause,seek:(t)=>{pause();elapsed=Math.max(0,Math.min(duration,t));render();},setRate:(r)=>{if(![1,.25,.1,.05].includes(r))throw new Error('Unsupported playback rate');rate=r;$('rate').value=r;lastTick=performance.now();render();},get data(){return data;},get time(){return elapsed;},get playing(){return playing;}};
  }
  init().catch(error=>{
    $('race').setAttribute('aria-busy','false');
    $('load-status').classList.add('error');
    $('load-status').innerHTML='The demo could not load. Reload to try again, or <a href="assets/progdraft-synchronized.mp4">watch the recording</a>.';
    console.error(error);
  });
})();
