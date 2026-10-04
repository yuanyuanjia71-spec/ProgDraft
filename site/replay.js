/* Measured three-way finish clock; recorded token events retain their original timestamps.
   Website player: CC BY-SA 4.0; see SITE_LICENSE.md. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const escape = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let data, duration = 0, elapsed = 0, playing = false, lastTick = 0, frame = null;
  let referencePieces = [];
  let archivePrefixesMatch = false;
  let rate = Number($('rate').value);
  const previousRound = {ar: -1, anchor: -1, ours: -1};
  let previousRaceText = '';
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
    const finish = method.measured.decode_s;
    const complete = elapsed >= finish;
    const rounds = method.rounds;
    let index = -1;
    for (let i=0; i<rounds.length && rounds[i].observed_decode_s <= elapsed; i++) index=i;
    const event = rounds[index];
    $(name+'-time').innerHTML = `${Math.min(elapsed,finish).toFixed(3)}<span>s</span>`;
    $(name+'-status').textContent = complete ? 'Finished' : playing ? 'Decoding' : elapsed ? 'Paused' : 'Ready';
    $(name+'-status').classList.toggle('done',complete);
    const traceComplete=event && event.tokens.length===data.reference.tokens.length;
    $(name+'-round').textContent = complete ? `${rounds.length} rounds` : traceComplete ? 'Recorded trace complete' : event ? `Round ${event.round}` : elapsed ? 'Preparing…' : 'Ready to start';
    $(name+'-exact').textContent = complete && traceComplete ? '✓ Exact token match' : '';
    $(name+'-exact').classList.toggle('verified',complete && traceComplete);
    $(name+'-bar').style.width = `${100*Math.min(elapsed/finish,1)}%`;
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
    if (name==='ours') $('progress-label').textContent = event ? `${event.positions[0].toFixed(1)} → ${event.positions.at(-1).toFixed(1)} s` : 'Across 8 draft steps';
  }

  function updateTarget() {
    const method=data.methods.ar;
    const finish=data.reference.decode_s;
    const complete=elapsed>=finish;
    let index=-1;
    if (method) for (let i=0; i<method.rounds.length && method.rounds[i].observed_decode_s<=elapsed; i++) index=i;
    const event=method?.rounds[index];
    const traceComplete=event && event.tokens.length===data.reference.tokens.length;
    const tokenTotal=data.reference.tokens.length;
    const prefill=Math.max(0,Math.min(finish,Number(data.reference.prefill_s)||0));
    const generation=Math.max(finish-prefill,Number.EPSILON);
    const illustratedCount=complete ? tokenTotal : Math.max(0,Math.min(tokenTotal-1,
      Math.floor((elapsed-prefill)*tokenTotal/generation)));
    const count=method ? event?.tokens.length || 0 : illustratedCount;
    $('target-latency').innerHTML=`${Math.min(elapsed,finish).toFixed(3)}<span>s</span>`;
    $('target-status').textContent=complete ? 'Finished' : playing ? 'Decoding' : elapsed ? 'Paused' : 'Ready';
    $('target-status').classList.toggle('done',complete);
    $('target-time-label').textContent=complete ? 'Measured finish' : event ? `Token ${event.token_index}` : elapsed ? 'Greedy decoding' : 'Ready to start';
    $('target-bar').style.width=`${100*Math.min(elapsed/finish,1)}%`;
    $('target-exact').textContent=complete && (!method || traceComplete) ? 'Reference token IDs' : '';
    if (previousRound.ar===index && method) return;
    previousRound.ar=index;
    const text=method ? event?.text || (elapsed ? 'Decoding…' : 'Press Start demo to begin.') :
      complete ? data.reference.text : count && archivePrefixesMatch ? referencePieces.slice(0,count).join('') :
      elapsed ? 'Decoding…' : 'Press Start demo to begin.';
    $('target-token-count').textContent=count;
    $('target-transcript').textContent=text;
    $('target-transcript').classList.toggle('is-empty',method ? !event?.text : !count);
    $('target-transcript').scrollTop=$('target-transcript').scrollHeight;
    $('target-step').textContent=count ? `Token ${count} / ${tokenTotal}` : method ? 'Waiting for first token' : 'Preparing target';
    const recent=method ? method.rounds.slice(Math.max(0,index-7),index+1).map(step=>({id:step.emitted[0],piece:step.piece})) :
      data.reference.tokens.slice(Math.max(0,count-8),count).map((id,i)=>({id,piece:referencePieces[Math.max(0,count-8)+i] ?? String(id)}));
    $('target-tokens').innerHTML=Array.from({length:8},(_,i)=>{
      const token=recent[i];
      return `<div class="token ${token ? 'target-token' : 'unused'}" title="${token ? `Target token ID ${escape(token.id)}` : 'No token event'}"><code>${token ? escape(token.piece || '∅') : '—'}</code><span class="token-mark" aria-hidden="true">${token ? '✓' : ''}</span></div>`;
    }).join('');
  }

  function updateRaceResult() {
    const finishes=[
      ['ProgDraft',data.methods.ours.measured.decode_s],
      ['AnchorDraft',data.methods.anchor.measured.decode_s],
      ['Target-only',data.reference.decode_s],
    ].sort((a,b)=>a[1]-b[1]);
    const completed=finishes.filter(([,seconds])=>elapsed>=seconds);
    const result=completed.length ? `Finish order: ${completed.map(([name,seconds],i)=>`${i+1}. ${name} ${seconds.toFixed(3)} s`).join('  ·  ')}${completed.length<3 ? '  ·  others still running' : ''}` : 'All three start together. Watch their measured finish times.';
    if (result!==previousRaceText) $('race-result').textContent=previousRaceText=result;
  }

  function render() {
    $('clock').textContent = elapsed.toFixed(3);
    $('seek').setAttribute('aria-valuetext', `${elapsed.toFixed(3)} of ${duration.toFixed(3)} seconds`);
    $('seek').value = elapsed;
    updateTarget(); updateMethod('anchor'); updateMethod('ours');
    updateRaceResult();
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
    referencePieces=data.methods.anchor.rounds.flatMap(event=>event.emitted_pieces);
    if (referencePieces.length!==data.reference.tokens.length) throw new Error('Incomplete target token pieces');
    archivePrefixesMatch=referencePieces.slice(0,-1).join('')===data.reference.text;
    duration=Math.max(data.reference.decode_s,data.methods.anchor.measured.decode_s,data.methods.ours.measured.decode_s,
      data.methods.anchor.observed.decode_s,data.methods.ours.observed.decode_s,data.methods.ar?.observed.decode_s || 0);
    $('target-wave').innerHTML=waveform(null);
    if (data.methods.ar) {
      $('demo-caption').textContent='All three timers start together and finish at independently measured decode latencies. Token events retain their recorded timestamps.';
      $('timeline-detail').textContent='All three finish times use independent callback-free measurements. Token events retain their original recorded timestamps, so a transcript may complete before or after its measured finish clock.';
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
