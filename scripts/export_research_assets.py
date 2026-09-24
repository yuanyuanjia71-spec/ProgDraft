"""Convert trusted local research assets into portable records/tensor weights.

Requires an explicit source root; no source module imports. Exported assets
stay outside Git. This tool does not upload or redistribute dataset audio.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from progress_asr.alignment import positions_from_alignment
from progress_asr.data import validate_record
from progress_asr.io import sha256, write_json


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def export_weights(source, destination, *, initial=False, cuda_index=0):
    payload = torch.load(source, map_location='cpu', weights_only=False)
    tensors = {}
    for component in ['drafter', 'progress_predictor']:
        for name, value in payload[component].items():
            tensors[f'{component}.{name}'] = value.detach().cpu().contiguous()
    if initial:
        tensors['rng_torch'] = payload['rng_torch'].cpu().contiguous()
        rng = payload['rng_cuda']
        tensors['rng_cuda'] = (rng[cuda_index] if isinstance(rng, list) else rng).cpu().contiguous()
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(destination), metadata={'source_sha256': sha256(source), 'step': str(payload.get('step', 0))})
    return dict(source_sha256=sha256(source), safetensors_sha256=sha256(destination), file=destination.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--scale', choices=['0.6b', '1.7b'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights-only', action='store_true')
    parser.add_argument('--limit', type=int, help='record export smoke only; omit for full training cache')
    parser.add_argument('--trusted-local-pickle', action='store_true', required=True,
                        help='acknowledge that source .pt checkpoints must be trusted')
    args = parser.parse_args()
    root, out = Path(args.source_root), Path(args.output)
    if args.scale == '0.6b':
        final = root/'random_k_progress_aware_drafter/checkpoints/step_14760.pt'
        initial = root/'random_k_progress_aware_drafter/checkpoints/step_00000.pt'
        manifest = root/'baseline_rebuild_sanity/data_manifest.json'
        features = root/'cache/features_strict_same_position_v1'
        model_id = 'Qwen/Qwen3-ASR-0.6B'; revision = '5eb144179a02acc5e5ba31e748d22b0cf3e303b0'
        hidden = 1024
    else:
        base = root/'target_scale_1p7b'
        final = base/'joint/formal/checkpoints/step_14760.pt'; initial = base/'step0.pt'
        manifest, features = base/'data_manifest.json', base/'features'
        model_id = 'Qwen/Qwen3-ASR-1.7B'; revision = '7278e1e70fe206f11671096ffdd38061171dd6e5'
        hidden = 2048
    out.mkdir(parents=True, exist_ok=True)
    weights = dict(final=export_weights(final, out/'ours.safetensors'),
                   initial=export_weights(initial, out/'step0.safetensors', initial=True))
    write_json(out/'weights.json', weights)
    if args.weights_only: return
    from huggingface_hub import snapshot_download
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
    path = snapshot_download(model_id, revision=revision, cache_dir=str(root/'cache/hf'), local_files_only=True)
    tokenizer = Qwen3ASRProcessor.from_pretrained(path).tokenizer
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    samples = json.loads(manifest.read_text())['samples']
    if args.limit: samples = samples[:args.limit]
    rows = []
    for sample in samples:
        stem = key(sample['id'])
        cache = torch.load(features/f'{stem}.pt', map_location='cpu', weights_only=True)
        if args.scale == '0.6b':
            align_dir = root/'cache'/('mms_fa_target_alignment_train_v1' if sample['split']=='train' else 'mms_fa_target_alignment_v1')
            l21_dir = root/'cache'/('verification_attention_l21_displacement_ablation_train_v1' if sample['split']=='train' else 'verification_attention_mean_peak_v1')
            with np.load(l21_dir/f'{stem}.npz') as values:
                layer_index = list(values['layers']).index(21)
                a1 = torch.tensor(values['peak_time'][:, layer_index], dtype=torch.float32)
        else:
            align_dir = root/'target_scale_1p7b/alignments'
            a1 = torch.as_tensor(cache['a1_verification_s'], dtype=torch.float32)
        alignment = json.loads((align_dir/f'{stem}.json').read_text())
        fa, valid = positions_from_alignment(tokenizer, sample['gold_token_ids'], alignment)
        c = {name: cache[name] for name in ['audio_memory', 'features', 'target_feature', 'current_tokens']}
        c.update(id=sample['id'], token_ids=sample['gold_token_ids'], duration_s=sample['duration_s'],
                 a1_verification_s=a1, fa_positions=fa, fa_valid=valid,
                 contract='same_position_target_greedy_l21_span_midpoint_v1')
        validate_record(c, hidden, eos)
        dest = out/'records'/f'{stem}.pt'; dest.parent.mkdir(exist_ok=True)
        torch.save(c, dest)
        rows.append(dict(id=sample['id'], split=sample['split'], cache=f'records/{stem}.pt', sha256=sha256(dest)))
    (out/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write_json(out/'cache_provenance.json', dict(source_manifest_sha256=sha256(manifest),
               records=len(rows), target=model_id, target_revision=revision, limited_export=bool(args.limit)))


if __name__ == '__main__':
    main()
