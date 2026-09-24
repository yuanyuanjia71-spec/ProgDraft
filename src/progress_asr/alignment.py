"""MMS character-span midpoint labels for Ours; distinct from AnchorDraft."""
import torch
from functools import lru_cache

ALPHABET = set("abcdefghijklmnopqrstuvwxyz'")


def token_spans(tokenizer, ids):
    previous, spans = '', []
    for i in range(len(ids)):
        decoded = tokenizer.decode(ids[:i+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        current = ''.join(c for c in decoded.lower() if c in ALPHABET)
        spans.append((len(previous), len(current)) if current.startswith(previous) and len(current)>len(previous) else None)
        previous = current
    return previous, spans


def positions_from_alignment(tokenizer, ids, alignment):
    text, spans = token_spans(tokenizer, ids)
    if text != alignment['normalized_target_text']:
        raise ValueError('FA text is not the target-generated token sequence')
    intervals = alignment['character_intervals']
    positions = torch.zeros(len(ids)); valid = torch.zeros(len(ids), dtype=torch.bool)
    for i, span in enumerate(spans):
        if span is not None and 0 <= span[0] < span[1] <= len(intervals):
            start, end = span
            positions[i] = (intervals[start]['start_s'] + intervals[end-1]['end_s']) / 2
            valid[i] = True
    return positions, valid


@lru_cache(maxsize=1)
def _mms_model(device):
    import torchaudio
    bundle = torchaudio.pipelines.MMS_FA
    vocabulary = bundle.get_dict(star=None)
    labels = bundle.get_labels(star=None)
    model = bundle.get_model(with_star=False).to(device).eval()
    return model, vocabulary, labels


@torch.inference_mode()
def align_target_text(waveform, token_ids, tokenizer, device):
    """Optional preprocessing, never called by training forward or inference."""
    import torchaudio
    model, vocabulary, labels = _mms_model(str(device))
    expected, _ = token_spans(tokenizer, token_ids)
    if not expected: raise ValueError('no MMS characters in target transcript')
    emission, _ = model(torch.as_tensor(waveform, device=device)[None])
    targets = torch.tensor([[vocabulary[c] for c in expected]], device=device)
    path, _ = torchaudio.functional.forced_align(emission, targets, blank=0)
    values = path[0].cpu().tolist()
    duration, start, position, intervals = len(waveform)/16000, 0, 0, []
    for end in range(1, len(values)+1):
        if end != len(values) and values[end] == values[start]: continue
        label = values[start]
        if label != 0:
            if position >= len(expected) or labels[label] != expected[position]:
                raise ValueError('MMS CTC path disagrees with target text')
            intervals.append(dict(char=labels[label], start_s=duration*start/len(values), end_s=duration*end/len(values)))
            position += 1
        start = end
    assert position == len(expected)
    return dict(normalized_target_text=expected, character_intervals=intervals)
