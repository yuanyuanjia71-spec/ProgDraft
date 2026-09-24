"""Random per-anchor horizons for the original Joint Progress architecture.

No new model parameters. Active rows alone enter each continuation block.
The caller supplies per-depth global denominators before microbatching.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from .progress import SIGMA_SECONDS, _center

MAX_K = 8
WEIGHTS = [0.7 ** depth for depth in range(MAX_K)]
P_SEEN = [1.0 if depth < 3 else (8 - depth) / 6 for depth in range(MAX_K)]
FACTORS = [w / p for w, p in zip(WEIGHTS, P_SEEN)]


def random_k_rollout_loss(drafter, predictor, embedding, lm_head, batch,
                          horizons, total_anchors, progress_denominators,
                          *, weights=WEIGHTS, probabilities=P_SEEN):
    """horizons are actual per-anchor lengths, bounded by available tokens.

CE means use all source anchors, including zero contributions from unsampled
depths. Progress means use all source anchors with an available FA label at
that depth, not just sampled/active rows. Thus 1/P_seen is applied exactly once.
    """
    device = batch['gold_extended'].device
    active = torch.arange(len(horizons), device=device)
    current_a = batch['a1_verification_s']
    state = kv = None
    total = torch.zeros((), device=device)
    observations, graph = [], []
    for depth in range(int(horizons.max())):
        selected = (horizons.index_select(0, active) > depth).nonzero().flatten()
        active = active.index_select(0, selected)
        if depth:
            state = state.index_select(0, selected)
            kv = tuple(value.index_select(0, selected) for value in kv)
            current_a = current_a.index_select(0, selected)
        audio = batch['audio_memory'].index_select(0, active)
        mask = batch['audio_mask'].index_select(0, active)
        adapter_values = {}
        if depth == 0:
            state, kv = drafter(
                embedding(batch['current_tokens'].index_select(0, active))[:, None],
                audio, mask,
                target_features=[value.index_select(0, active) for value in batch['features']],
            )
        else:
            duration = batch['audio_duration_s'].index_select(0, active)
            def adapter(hidden_pre):
                delta = predictor(hidden_pre, current_a, duration)
                raw = current_a + delta
                adapter_values.update(delta=delta, raw=raw, previous=current_a)
                return _center(raw, duration)
            state, kv = drafter(
                embedding(batch['gold_extended'][active, depth - 1])[:, None],
                audio, mask, recurrent_state=state, past_kv=kv,
                audio_time_positions=batch['audio_time_positions'].index_select(0, active),
                acoustic_sigma_seconds=SIGMA_SECONDS, acoustic_anchor_adapter=adapter,
            )
            current_a = adapter_values['raw']
        if kv[0].shape[-2] != depth + 1:
            raise AssertionError('incorrect cumulative self-KV length')
        logits = lm_head(state[:, 0]).float()
        labels = batch['gold_extended'][active, depth]
        if bool(labels.lt(0).any()):
            raise AssertionError('rollout entered unavailable future target token')
        ce = F.cross_entropy(logits, labels, reduction='sum')
        feature = torch.zeros((), device=device)
        progress = torch.zeros((), device=device)
        error = torch.zeros((), device=device)
        valid_count = 0
        if depth == 0:
            feature = F.smooth_l1_loss(
                state[:, 0].float(), batch['target_feature'][active].float(), reduction='sum'
            ) / state.shape[-1]
            total = total + 0.5 * feature / total_anchors
        else:
            valid = batch['fa_valid_extended'][active, depth]
            valid_count = int(valid.sum())
            if valid_count:
                fa = batch['fa_extended'][active, depth][valid]
                progress = F.smooth_l1_loss(current_a[valid].float(), fa, reduction='sum')
                error = (current_a[valid].float() - fa).abs().sum()
        factor = weights[depth] / probabilities[depth]
        total = total + factor * ce / total_anchors
        if depth:
            total = total + 0.1 * factor * progress / max(progress_denominators[depth], 1)
        observations.append(dict(
            depth=depth + 1, count=len(active), CE_sum=float(ce.detach()),
            progress_sum=float(progress.detach()), position_abs_sum=float(error.detach()),
            progress_valid_count=valid_count, feature_sum=float(feature.detach()),
            correct=int(logits.argmax(-1).eq(labels).sum()),
        ))
        graph.append(dict(depth=depth + 1, active=active, state=state,
                          position=current_a, CE=ce, progress=progress,
                          kv_length=kv[0].shape[-2], **adapter_values))
    return total, observations, graph
