"""Export an exact-checked, timestamped comparison for the static project page.

No decoder is implemented here. Uses the same runtime as the Gradio demo.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from demo.demo_backend import DemoBackend, resolve_config, replay_timeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'demo/local_config.json'))
    parser.add_argument('--preset', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = resolve_config(args.config)
    preset = config['presets'][args.preset]
    backend = DemoBackend(config)
    comparison = backend.capture(preset['path'], lambda **e: print(e.get('text', e['kind']), flush=True))
    replay_timeline(comparison)  # Validate original event order and finish boundaries.
    model_config = json.loads(Path(config['model_config']).read_text())
    comparison['provenance'] = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        sample_id=preset.get('sample_id', ''), sample_label=preset['label'],
        selection='First sample of this dataset in fixed Final Test manifest order; not outcome-selected.',
        target=model_config['target']['model_id'],
        anchor='Paper-table AnchorDraft; fixed K=3 training; step 14760; no Runtime Correction',
        ours='Joint Progress-Aware + Random-K[3,8]; step 14760',
        audio_sha256=hashlib.sha256(Path(preset['path']).read_bytes()).hexdigest(),
        weights_sha256={name: hashlib.sha256(Path(config[name+'_weights']).read_bytes()).hexdigest()
                        for name in ('anchor', 'ours')},
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(comparison, ensure_ascii=False, separators=(',', ':')) + '\n')
    for name, state in comparison['methods'].items():
        print(name, 'recorded_s=', state['observed']['decode_s'],
              'callback_free_s=', state['measured']['decode_s'],
              'rounds=', len(state['rounds']), 'exact=', state['exact'], flush=True)


if __name__ == '__main__':
    main()
