"""One immutable tensor record per utterance; FA labels never enter inference."""
from pathlib import Path
import json
import numpy as np
import torch
from .io import sha256


def audio_memory_times(duration_s, memory_length):
    # Historical Ours convention. This is a uniform duration-based midpoint
    # grid, not an encoder receptive-field timestamp reconstruction.
    if duration_s <= 0 or memory_length < 1:
        raise ValueError('invalid duration or audio memory length')
    return (np.arange(memory_length, dtype=np.float32) + .5) * (float(duration_s) / memory_length)


def validate_record(c, hidden_size, eos):
    tokens = c['token_ids']
    n = len(tokens) - 2
    assert n > 0 and tokens[-1] == eos and eos not in tokens[:-1]
    assert c['features'].shape == (n, 4, hidden_size)
    assert c['target_feature'].shape == (n, hidden_size)
    assert c['audio_memory'].ndim == 2 and c['audio_memory'].shape[1] == hidden_size
    assert c['current_tokens'].shape == (n,)
    assert torch.equal(c['current_tokens'][1:], torch.tensor(tokens[:n-1]))
    assert c['a1_verification_s'].shape == (n,)
    assert torch.isfinite(c['a1_verification_s']).all()
    assert c['fa_positions'].shape == c['fa_valid'].shape == (len(tokens),)
    assert c['fa_valid'].dtype == torch.bool and not c['fa_valid'][-1]
    assert c['duration_s'] > 0
    assert torch.isfinite(c['fa_positions'][c['fa_valid']]).all()
    assert c['contract'] == 'same_position_target_greedy_l21_span_midpoint_v1'


def load_records(manifest, split, hidden_size, eos):
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [r for r in rows if r['split'] == split]
    result = []
    for row in selected:
        path = manifest.parent / row['cache']
        if row.get('sha256') and sha256(path) != row['sha256']:
            raise ValueError(f"cache checksum mismatch for {row['id']}")
        c = torch.load(path, map_location='cpu', weights_only=True)
        validate_record(c, hidden_size, eos)
        assert c['id'] == row['id']
        result.append(c)
    if len({r['id'] for r in result}) != len(result):
        raise ValueError('duplicate utterance IDs')
    return result


def epoch_plan(records, order, rng):
    sampled, actual = [], []
    for index in order:
        record = records[index]
        n = len(record['current_tokens'])
        s = rng.integers(3, 9, size=n)
        sampled.append(s)
        actual.append(np.minimum(s, len(record['token_ids']) - np.arange(n)))
    flat = np.concatenate(actual)
    probabilities = [float((flat >= k).mean()) for k in range(1, 9)]
    if min(probabilities) <= 0:
        raise ValueError('no source anchor can reach one of the configured depths')
    return sampled, actual, probabilities


def collate(items, device):
    n = len(items)
    length = max(len(c['audio_memory']) for c, _ in items)
    hidden = items[0][0]['audio_memory'].shape[-1]
    memory = torch.zeros(n, length, hidden, dtype=torch.bfloat16)
    mask = torch.zeros(n, length, dtype=torch.bool)
    times = torch.zeros(n, length)
    gold = torch.full((n, 8), -1, dtype=torch.long)
    fa = torch.zeros(n, 8)
    valid = torch.zeros(n, 8, dtype=torch.bool)
    for row, (c, anchor) in enumerate(items):
        m = len(c['audio_memory'])
        memory[row, :m] = c['audio_memory']
        mask[row, :m] = True
        times[row, :m] = torch.from_numpy(audio_memory_times(c['duration_s'], m))
        tokens = c['token_ids'][anchor:anchor+8]
        gold[row, :len(tokens)] = torch.tensor(tokens)
        fa[row, :len(tokens)] = c['fa_positions'][anchor:anchor+8]
        valid[row, :len(tokens)] = c['fa_valid'][anchor:anchor+8]
    batch = dict(audio_memory=memory, audio_mask=mask, audio_time_positions=times,
                 gold_extended=gold, fa_extended=fa, fa_valid_extended=valid,
                 current_tokens=torch.stack([c['current_tokens'][a] for c, a in items]),
                 target_feature=torch.stack([c['target_feature'][a] for c, a in items]),
                 a1_verification_s=torch.stack([c['a1_verification_s'][a] for c, a in items]),
                 audio_duration_s=torch.tensor([c['duration_s'] for c, _ in items]))
    batch = {k: v.to(device) for k, v in batch.items()}
    batch['features'] = [torch.stack([c['features'][a, d] for c, a in items])[:, None].to(device) for d in range(4)]
    return batch


def progress_denominators(items):
    return [sum(int(c['fa_valid'][a+d]) for c, a in items if a+d < len(c['token_ids'])) for d in range(8)]
