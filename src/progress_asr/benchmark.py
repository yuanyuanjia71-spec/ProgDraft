"""Paired AR/Ours timing. A divergence fails the run before aggregate speedup."""
import argparse
import csv
import json
from pathlib import Path
from .io import read_json, write_json, sha256
from .models import load_target, make_models, load_weights
from .runtime import execute
from .metrics import normalize, distance


def summarize(rows):
    output = []
    for dataset in ['ALL', *dict.fromkeys(r['dataset'] for r in rows)]:
        subset = [r for r in rows if dataset == 'ALL' or r['dataset'] == dataset]
        ar_time = sum(r['e2e_s'] for r in subset if r['method'] == 'Target-only AR')
        for method in ['Target-only AR', 'Ours']:
            values = [r for r in subset if r['method'] == method]
            rounds = sum(r['rounds'] for r in values)
            word_err = char_err = word_n = char_n = 0
            for r in values:
                ref, hyp = normalize(r['reference']), normalize(r['text'])
                word_err += distance(ref.split(), hyp.split())
                char_err += distance(ref.replace(' ', ''), hyp.replace(' ', ''))
                word_n += len(ref.split()); char_n += len(ref.replace(' ', ''))
            output.append(dict(dataset=dataset, method=method,
                               Mean_Accepted=sum(len(r['tokens']) for r in values)/rounds if rounds else None,
                               E2E_speedup=ar_time/sum(r['e2e_s'] for r in values),
                               WER=word_err/word_n if word_n else None, CER=char_err/char_n if char_n else None))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--manifest', required=True, help='JSONL: id, dataset, audio, reference')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--target-path')
    parser.add_argument('--k', type=int, default=8)
    a = parser.parse_args()
    config = read_json(a.config)
    manifest = Path(a.manifest)
    samples = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    assert samples and len({s['id'] for s in samples}) == len(samples)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'per_utterance.jsonl').exists():
        raise FileExistsError('use a new output directory; existing measurements are immutable')
    target, processor = load_target(config, a.device, target_path=a.target_path)
    draft, predictor = make_models(config, a.device)
    load_weights(draft, predictor, a.weights, a.device)
    draft.eval().requires_grad_(False); predictor.eval().requires_grad_(False)
    write_json(out/'config.json', dict(config=config, K=a.k, weights_sha256=sha256(a.weights),
               manifest_sha256=sha256(manifest), device=a.device, protocol='fp32_eager_native_batched_independent_correction_v2',
               timing='waveform load through final text; boundary synchronizations; warm-up excluded'))
    def run(sample, name):
        # Reference text and FA never reach the runtime.
        return execute(target, processor, manifest.parent/sample['audio'], a.device,
                       draft if name == 'Ours' else None, predictor if name == 'Ours' else None, a.k)
    for name in ['Target-only AR', 'Ours']:
        run(samples[0], name)
    rows = []
    for index, sample in enumerate(samples):
        names = ['Target-only AR', 'Ours'] if index % 2 == 0 else ['Ours', 'Target-only AR']
        group = [dict(id=sample['id'], dataset=sample['dataset'], method=name, reference=sample['reference'], **run(sample, name)) for name in names]
        if group[0]['tokens'] != group[1]['tokens']:
            write_json(out/'divergence.json', dict(id=sample['id'], outputs=group))
            raise RuntimeError('target-only token mismatch; no aggregate speedup emitted')
        rows.extend(group)
        with (out/'per_utterance.jsonl').open('a') as handle:
            for row in group:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        write_json(out/'status.json', dict(completed=index+1, total=len(samples)))
    metrics = summarize(rows)
    with (out/'metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    write_json(out/'summary.json', dict(metrics=metrics, exact_match=f'{len(samples)}/{len(samples)}'))


if __name__ == '__main__':
    main()
