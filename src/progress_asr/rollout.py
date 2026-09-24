"""Free-running proposals on the live progress-corrected path."""
import torch
from .progress import SIGMA_SECONDS, _center

def progress_rollout(draft, predictor, target, runner, context, depth: int) -> torch.Tensor:
    if context.verification_anchor_time_s is None or not context.q_from_current_verification:
        raise AssertionError("Joint rollout requires the current L21 verification anchor")
    embedding, head = target.thinker.get_input_embeddings(), target.thinker.lm_head
    duration = torch.tensor(
        [float(runner.cache_record["sample"]["duration_s"])],
        device=runner.device,
        dtype=torch.float32,
    )
    audio_times = runner.audio_times[None].float()
    current_position = torch.tensor(
        [float(context.verification_anchor_time_s)], device=runner.device, dtype=torch.float32
    )
    token = torch.tensor([context.current_token], device=runner.device)
    z, kv = draft(
        embedding(token)[:, None], runner.audio_memory, runner.audio_mask,
        target_features=context.features,
    )
    candidates = [head(z[:, 0]).argmax(dim=-1)]
    for _ in range(2, depth + 1):
        holder: dict[str, torch.Tensor] = {}

        def adapter(hidden_pre, previous=current_position):
            delta = predictor(hidden_pre, previous, duration)
            raw = previous + delta
            holder.update(raw=raw, center=_center(raw, duration))
            return holder["center"]

        z, kv = draft(
            embedding(candidates[-1])[:, None], runner.audio_memory, runner.audio_mask,
            recurrent_state=z, past_kv=kv,
            audio_time_positions=audio_times,
            acoustic_sigma_seconds=SIGMA_SECONDS,
            acoustic_anchor_adapter=adapter,
        )
        if not holder:
            raise AssertionError("progress adapter was not invoked")
        current_position = holder["raw"]
        candidates.append(head(z[:, 0]).argmax(dim=-1))
    if int(kv[0].shape[-2]) != depth:
        raise AssertionError(f"self-KV length {kv[0].shape[-2]} != K={depth}")
    return torch.stack([token[0] for token in candidates])
