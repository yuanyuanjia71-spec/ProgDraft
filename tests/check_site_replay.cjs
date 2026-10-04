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
  assert.equal(old.elements.get('target-transcript').textContent, archived.reference.text);
  assert.match(old.elements.get('target-latency').innerHTML,
    new RegExp(archived.reference.decode_s.toFixed(3)));
  old.replay.seek(.4);
  assert.equal(old.elements.get('target-transcript').textContent, archived.reference.text);

  const current = structuredClone(archived);
  current.methods.ar = {
    exact: true, observed: {...current.reference, decode_s: .96},
    measured: current.reference,
    rounds: [
      {token_index: 1, tokens: current.reference.tokens.slice(0, 1), text: 'From', observed_decode_s: .1},
      {token_index: current.reference.tokens.length, tokens: current.reference.tokens,
        text: current.reference.text, observed_decode_s: .94},
    ],
  };
  const fresh = await load(current);
  assert.match(fresh.elements.get('demo-caption').textContent, /Three independent runs/);
  fresh.replay.seek(.2);
  assert.equal(fresh.elements.get('target-transcript').textContent, 'From');
  assert.equal(fresh.elements.get('target-token-count').textContent, 1);
  fresh.replay.seek(.96);
  assert.equal(fresh.elements.get('target-transcript').textContent, current.reference.text);
  assert.equal(fresh.elements.get('target-token-count').textContent, current.reference.tokens.length);
  process.stdout.write('site replay: archived baseline and timestamped AR trace passed\n');
})().catch(error => {console.error(error); process.exitCode = 1;});
