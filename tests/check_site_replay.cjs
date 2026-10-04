// Exercise archived and newly exported Target-only traces without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const script = fs.readFileSync(path.join(root, 'site/replay.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'site/index.html'), 'utf8');
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]));
const archived = JSON.parse(fs.readFileSync(path.join(root, 'site/assets/demo-trace.json'), 'utf8'));

async function load(trace) {
  const elements = new Map();
  const element = id => {
    assert.ok(ids.has(id), `missing HTML element #${id}`);
    if (!elements.has(id)) elements.set(id, {
      value: id === 'rate' ? '0.1' : '0', style: {}, scrollHeight: 0,
      classList: {add() {}, remove() {}, toggle() {}}, setAttribute() {},
    });
    return elements.get(id);
  };
  const errors = [];
  const context = {
    document: {getElementById: element, body: {classList: {add() {}}}, addEventListener() {}},
    location: {search: ''}, URLSearchParams, performance: {now: () => 0},
    requestAnimationFrame: () => 1, cancelAnimationFrame() {},
    fetch: async () => ({ok: true, json: async () => trace}),
    console: {error: error => errors.push(error)}, window: {},
  };
  vm.runInNewContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(errors, []);
  assert.ok(context.window.progdraftReplay, 'player initialized');
  return {elements, replay: context.window.progdraftReplay};
}

(async () => {
  const old = await load(archived);
  assert.equal(old.elements.get('target-transcript').textContent, 'Press Start demo to begin.');
  assert.equal(old.elements.get('target-status').textContent, 'Ready');
  old.elements.get('play').onclick();
  for (const name of ['target', 'anchor', 'ours']) {
    assert.equal(old.elements.get(`${name}-status`).textContent, 'Decoding');
  }
  old.replay.seek(archived.reference.prefill_s/2);
  assert.equal(old.elements.get('target-token-count').textContent, 0);
  const generationHalf=archived.reference.prefill_s+
    (archived.reference.decode_s-archived.reference.prefill_s)/2;
  old.replay.seek(generationHalf);
  const halfwayCount=old.elements.get('target-token-count').textContent;
  assert.ok(halfwayCount > 0 && halfwayCount < archived.reference.tokens.length);
  const pieces=archived.methods.anchor.rounds.flatMap(event=>event.emitted_pieces);
  assert.equal(old.elements.get('target-transcript').textContent, pieces.slice(0,halfwayCount).join(''));
  const firstAnchorEvent = archived.methods.anchor.rounds[0].observed_decode_s;
  old.replay.seek(firstAnchorEvent - .001);
  assert.equal(old.elements.get('anchor-round').textContent, 'Preparing…');
  old.replay.seek(firstAnchorEvent);
  assert.equal(old.elements.get('anchor-round').textContent, 'Round 1');
  const oursFinish = archived.methods.ours.measured.decode_s;
  const anchorFinish = archived.methods.anchor.measured.decode_s;
  const targetFinish = archived.reference.decode_s;
  old.replay.seek((oursFinish+anchorFinish)/2);
  assert.equal(old.elements.get('ours-status').textContent, 'Finished');
  assert.match(old.elements.get('ours-time').innerHTML,
    new RegExp(oursFinish.toFixed(3)));
  assert.equal(old.elements.get('ours-bar').style.width, '100%');
  assert.equal(old.elements.get('anchor-status').textContent, 'Paused');
  assert.equal(old.elements.get('target-status').textContent, 'Paused');
  assert.ok(parseFloat(old.elements.get('target-bar').style.width) < 100);
  assert.match(old.elements.get('race-result').textContent, /1\. ProgDraft/);
  old.replay.seek((anchorFinish+targetFinish)/2);
  assert.equal(old.elements.get('anchor-status').textContent, 'Finished');
  assert.equal(old.elements.get('target-status').textContent, 'Paused');
  assert.ok(old.elements.get('target-token-count').textContent > halfwayCount);
  old.replay.seek(targetFinish);
  assert.equal(old.elements.get('target-status').textContent, 'Finished');
  assert.match(old.elements.get('target-latency').innerHTML,
    new RegExp(targetFinish.toFixed(3)));
  assert.equal(old.elements.get('target-bar').style.width, '100%');
  assert.equal(old.elements.get('target-transcript').textContent, archived.reference.text);
  assert.equal(old.elements.get('target-token-count').textContent, archived.reference.tokens.length);
  assert.match(old.elements.get('target-tokens').innerHTML, /Target token ID/);
  assert.match(old.elements.get('race-result').textContent, /3\. Target-only/);
  old.elements.get('restart').onclick();
  assert.equal(old.elements.get('target-transcript').textContent, 'Press Start demo to begin.');

  const current = structuredClone(archived);
  current.methods.ar = {
    exact: true, observed: {...current.reference, decode_s: .96},
    measured: current.reference,
    rounds: [
      {token_index: 1, emitted: [current.reference.tokens[0]], piece: 'From',
        tokens: current.reference.tokens.slice(0, 1), text: 'From', observed_decode_s: .1},
      {token_index: current.reference.tokens.length, tokens: current.reference.tokens,
        emitted: [current.reference.tokens.at(-1)], piece: '<end>',
        text: current.reference.text, observed_decode_s: .94},
    ],
  };
  const fresh = await load(current);
  assert.match(fresh.elements.get('demo-caption').textContent, /All three timers/);
  fresh.replay.seek(.2);
  assert.equal(fresh.elements.get('target-transcript').textContent, 'From');
  assert.equal(fresh.elements.get('target-token-count').textContent, 1);
  fresh.replay.seek(.96);
  assert.equal(fresh.elements.get('target-transcript').textContent, current.reference.text);
  assert.equal(fresh.elements.get('target-token-count').textContent, current.reference.tokens.length);
  assert.match(fresh.elements.get('target-tokens').innerHTML, /&lt;end&gt;/);
  process.stdout.write('site replay: archived baseline and timestamped AR trace passed\n');
})().catch(error => {console.error(error); process.exitCode = 1;});
