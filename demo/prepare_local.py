"""Export a trusted local AnchorDraft checkpoint and register local test presets.

No model training, downloading, public audio redistribution or research imports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor-checkpoint', required=True)
    parser.add_argument('--anchor-label', required=True)
    parser.add_argument('--ours-weights', default=str(ROOT / 'artifacts/0.6b/ours.safetensors'))
    parser.add_argument('--model-config', default=str(ROOT / 'configs/qwen3_asr_0.6b.json'))
    parser.add_argument('--target-path')
    parser.add_argument('--manifest', help='optional local Final Test JSONL: dataset, audio_npy, global_index')
    parser.add_argument('--audio-root', help='resolve manifest audio_npy relative to this directory')
    parser.add_argument('--trusted-local-pickle', action='store_true', required=True,
                        help='source .pt must be a trusted local checkpoint')
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import save_file
    checkpoint = torch.load(args.anchor_checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint.get('draft', checkpoint.get('drafter'))
    if not isinstance(state, dict):
        raise ValueError('Expected draft/drafter state_dict')
    out = ROOT / 'artifacts/demo'
    out.mkdir(parents=True, exist_ok=True)
    destination = out / 'anchordraft.safetensors'
    save_file({f'drafter.{k}': v.detach().cpu().contiguous() for k, v in state.items()}, str(destination),
              metadata={'source_sha256': sha(args.anchor_checkpoint), 'label': args.anchor_label,
                        'step': str(checkpoint.get('step', 'unknown'))})
    presets = []
    if args.manifest:
        if not args.audio_root:
            parser.error('--manifest requires --audio-root')
        rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
        seen = set()
        for row in rows:
            if row['dataset'] in seen:
                continue
            seen.add(row['dataset'])
            source = Path(args.audio_root) / row['audio_npy']
            wave = np.load(source, allow_pickle=False)
            path = out / f'{row["dataset"]}.wav'
            # FLOAT WAV preserves the exact input samples; no PCM quantization.
            sf.write(path, wave, 16000, subtype='FLOAT')
            presets.append(dict(label=f'{row["dataset"]} · {row["duration_s"]:.1f}s', path=str(path),
                                sample_id=row.get('utterance_id'), global_index=row.get('global_index')))
    config = dict(model_config=str(Path(args.model_config).resolve()), ours_weights=str(Path(args.ours_weights).resolve()),
                  anchor_weights=str(destination), anchor_label=args.anchor_label,
                  ours_label='Joint + Random-K[3,8] · step 14760', device='cuda:0', max_audio_seconds=60,
                  ours_weights_sha256=sha(args.ours_weights), anchor_weights_sha256=sha(destination),
                  presets=presets)
    if args.target_path:
        config['target_path'] = str(Path(args.target_path).resolve())
    (ROOT / 'demo/local_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    (out / 'provenance.json').write_text(json.dumps(dict(
        anchor_source_sha256=sha(args.anchor_checkpoint), anchor_export_sha256=sha(destination),
        ours_sha256=sha(args.ours_weights), source_manifest_sha256=sha(args.manifest) if args.manifest else None,
        preset_selection='first sample per dataset in manifest order; no outcome-based selection'), indent=2)+'\n')
    print('Local assets ready. Start: python demo/app.py')


if __name__ == '__main__':
    main()
