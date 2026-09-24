"""Batch-one waveform-to-text runtime used for both paired benchmark arms."""
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .data import audio_memory_times
from .target_cache import CachedTargetRunner
from .rollout import progress_rollout


def read_audio(path):
    if Path(path).suffix == '.npy':
        wave = np.load(path, allow_pickle=False)
        if wave.ndim != 1:
            raise ValueError('npy input must be mono 16 kHz')
        return wave.astype(np.float32, copy=False)
    import soundfile as sf
    wave, sr = sf.read(path, dtype='float32')
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != 16000:
        import librosa
        wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(wave, dtype=np.float32)


class WaveformRunner(CachedTargetRunner):
    def __init__(self, target, processor, waveform, device, capture=True, language='English'):
        self.target, self.processor, self.device = target, processor, device
        self.capture_runtime_attention = capture
        self.component_timing = None
        duration = len(waveform) / 16000
        self.cache_record = {'sample': {'duration_s': duration}}
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': [{'type': 'audio', 'audio': ''}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt += f'language {language}<asr_text>'
        x = processor(text=[prompt], audio=[waveform], sampling_rate=16000, return_tensors='pt', padding=True)
        ids = x['input_ids'].to(device)
        memory = target.thinker.get_audio_features(
            x['input_features'].to(device=device, dtype=next(target.parameters()).dtype),
            feature_attention_mask=x['feature_attention_mask'].to(device))
        emb = target.thinker.get_input_embeddings()(ids)
        mask = target.thinker.get_placeholder_mask(ids, inputs_embeds=emb)
        slots = mask[0, :, 0].nonzero().flatten()
        if len(slots) != len(memory):
            assert len(slots) > 0 and torch.equal(slots, torch.arange(slots[0], slots[-1]+1, device=device))
            ids = torch.cat([ids[0, :slots[0]], ids.new_full((len(memory),), ids[0, slots[0]]), ids[0, slots[-1]+1:]])[None]
            emb = target.thinker.get_input_embeddings()(ids)
            mask = target.thinker.get_placeholder_mask(ids, inputs_embeds=emb)
        assert int(mask[0, :, 0].sum()) == len(memory)
        emb = emb.masked_scatter(mask, memory.to(emb.dtype))
        attn = torch.ones_like(ids)
        pos, _ = target.thinker.get_rope_index(attn)
        self.prepared = dict(inputs_embeds=emb, attention_mask=attn, position_ids=pos)
        self.prompt_ids, self.prompt_length = ids, ids.shape[1]
        self.current_token = int(ids[0, -1])
        self.audio_positions = mask[0, :, 0].nonzero().flatten()
        self.audio_memory = memory[None].to(torch.bfloat16)
        self.audio_mask = torch.ones((1, len(memory)), device=device, dtype=torch.bool)
        self.audio_times = torch.from_numpy(audio_memory_times(duration, len(memory))).to(device)
        self.forward_counts = dict(prefill=0, candidate_verification=0, context_update=0)

    def prefill(self, *, capture_features=True):
        self.forward_counts['prefill'] += 1
        context = self._forward(**self.prepared, capture_runtime_attention=capture_features)
        context.current_token = self.current_token
        torch.cuda.synchronize(self.device)
        self.generation_start = time.perf_counter()
        return context


@torch.inference_mode()
def decode(runner, draft=None, predictor=None, k=8, max_new_tokens=512):
    if k < 1:
        raise ValueError('K must be positive')
    context = runner.prefill(capture_features=draft is not None)
    eos = int(runner.processor.tokenizer.convert_tokens_to_ids('<|im_end|>'))
    tokens, rounds, accepted_sum, distribution = [], 0, 0, Counter()
    while len(tokens) < max_new_tokens:
        if draft is None:
            token = int(context.next_logits.argmax(-1)[0])
            tokens.append(token)
            if token == eos:
                break
            context = runner.append_target_token(context, token, capture_features=False)
        else:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                candidates = progress_rollout(draft, predictor, runner.target, runner, context, k)
            accepted, _, emitted, context = runner.verify_block(context, candidates)
            tokens.extend(emitted)
            rounds += 1
            accepted_sum += accepted
            distribution[accepted] += 1
            if context is None:
                assert tokens[-1] == eos
                break
    else:
        raise RuntimeError('generation reached token cap; exclude from completed benchmark')
    return tokens, dict(rounds=rounds, accepted_sum=accepted_sum, accept_distribution=dict(distribution),
                        emitted=len(tokens), mean_accepted=len(tokens)/rounds if rounds else None,
                        target_forwards=dict(runner.forward_counts), drafter_forwards=rounds*k)


@torch.inference_mode()
def execute(target, processor, path, device, draft=None, predictor=None, k=8):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    runner = WaveformRunner(target, processor, read_audio(path), device, capture=draft is not None)
    torch.cuda.synchronize(device)
    decode_start = time.perf_counter()
    tokens, stats = decode(runner, draft, predictor, k)
    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    return dict(tokens=tokens, text=text, e2e_s=end-start, decode_s=end-decode_start,
                prefill_s=runner.generation_start-decode_start, generation_s=end-runner.generation_start,
                audio_duration_s=runner.cache_record['sample']['duration_s'], **stats)
