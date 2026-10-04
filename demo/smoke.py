"""Real GPU test of all local demo presets, observed and unobserved paths."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from demo.demo_backend import DemoBackend, resolve_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'demo/local_config.json'))
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    config = resolve_config(args.config)
    backend = DemoBackend(config)
    for preset in config.get('presets', [])[:args.limit]:
        results, rounds = {}, {'ar': [], 'anchor': [], 'ours': []}
        for event in backend.events(preset['path']):
            if event['kind'] == 'error':
                raise RuntimeError(event['text'])
            if event['kind'] == 'round':
                r = event['event']
                if event['method'] == 'ar':
                    assert r['token_index'] == len(r['tokens'])
                    assert len(r['emitted']) == 1
                else:
                    assert len(r['candidates']) == 8
                    assert len(r['positions']) == (8 if event['method'] == 'ours' else 0)
                    if r['positions']:
                        assert all(b >= a for a, b in zip(r['positions'], r['positions'][1:]))
                rounds[event['method']].append(r)
            if event['kind'] == 'measured':
                r = event['result']
                expected_events = r['emitted'] if event['method'] == 'ar' else r['rounds']
                assert len(rounds[event['method']]) == expected_events
                results[event['method']] = dict(rounds=r['rounds'], accepted_sum=r['accepted_sum'],
                                                tokens=len(r['tokens']), exact=True)
        assert set(results) == {'ar', 'anchor', 'ours'}
        print(json.dumps(dict(preset=preset['label'], results=results), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
