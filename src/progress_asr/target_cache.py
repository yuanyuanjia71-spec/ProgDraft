"""Native cached target verification; no auxiliary target-only guard."""
from dataclasses import dataclass
from typing import Callable
import torch
from torch.nn import functional as F
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import apply_rotary_pos_emb, repeat_kv
FEATURE_LAYERS = (0, 9, 18, 27)
RUNTIME_LAYER = 21

@dataclass
class TargetContext:
    """Target state at the current committed text token."""
    cache: object
    cache_length: int
    current_token: int
    features: list[torch.Tensor]
    last_hidden_state: torch.Tensor
    next_logits: torch.Tensor
    verification_anchor_time_s: float | None
    verification_anchor_audio_index: int | None
    verification_peak_weight: float | None
    q_from_current_verification: bool


class _TargetForwardCapture:
    """Capture strict restart features and L21 query during one target forward."""

    def __init__(self, target, audio_positions: torch.Tensor, audio_times: torch.Tensor,
                 *, capture_runtime_attention: bool = True):
        self.target = target
        self.audio_positions = audio_positions
        self.audio_times = audio_times
        self.features: dict[int, torch.Tensor] = {}
        self.query: torch.Tensor | None = None
        self.capture_runtime_attention = capture_runtime_attention
        self._handles = []

    def _feature_hook(self, layer: int) -> Callable:
        def callback(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            self.features[layer] = value[:, -1:].detach()
        return callback

    def _query_hook(self, module, _args, kwargs) -> None:
        hidden = kwargs['hidden_states']
        cos, sin = kwargs['position_embeddings']
        shape = (*hidden.shape[:-1], -1, module.head_dim)
        query = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
        # Apply exactly the same RoPE path used by Qwen self-attention.  The
        # companion key is only needed by the helper and is deliberately
        # discarded: runtime scores below use the key cache written by the
        # actual target forward, not a second K projection.
        placeholder_key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
        query, _ = apply_rotary_pos_emb(query, placeholder_key, cos, sin)
        self.query = query[:, :, -1:, :].detach()

    def install(self) -> None:
        for layer in FEATURE_LAYERS:
            self._handles.append(self.target.thinker.model.layers[layer].register_forward_hook(self._feature_hook(layer)))
        if self.capture_runtime_attention:
            self._handles.append(self.target.thinker.model.layers[RUNTIME_LAYER].self_attn.register_forward_pre_hook(
                self._query_hook, with_kwargs=True))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def runtime_anchor_from_existing_cache(self, cache) -> tuple[int, float, float]:
        if self.query is None:
            raise AssertionError('L21 query was not captured during target verification')
        attention = self.target.thinker.model.layers[RUNTIME_LAYER].self_attn
        # ``keys`` is the DynamicCache content written by the very same target
        # forward.  Repeat GQA KV heads before the QK product, just as Qwen's
        # self-attention implementation does.
        keys = cache.layers[RUNTIME_LAYER].keys
        if keys is None:
            raise AssertionError('target L21 KV cache is empty')
        keys = repeat_kv(keys, attention.num_key_value_groups)
        scores = torch.matmul(self.query.float(), keys.float().transpose(-2, -1)) * attention.scaling
        all_key_weights = torch.softmax(scores, dim=-1)
        audio_weights = all_key_weights[:, :, :, self.audio_positions]
        mean_audio = audio_weights.mean(dim=1)[0, 0]
        index = int(mean_audio.argmax())
        return index, float(self.audio_times[index]), float(mean_audio[index])



class CachedTargetRunner:
    def target_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Float32 logit readout; released runtime also uses an FP32 target."""
        head = self.target.thinker.lm_head
        bias = head.bias.float() if head.bias is not None else None
        return F.linear(hidden.float(), head.weight.float(), bias)


    def _forward(self, *, inputs_embeds: torch.Tensor, past_key_values=None,
                 attention_mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
                 capture_runtime_attention: bool) -> TargetContext:
        collector = (_TargetForwardCapture(self.target, self.audio_positions, self.audio_times,
                                           capture_runtime_attention=self.capture_runtime_attention)
                     if capture_runtime_attention else None)
        if collector is not None:
            collector.install()
        try:
            output = self.target.thinker.model(
                inputs_embeds=inputs_embeds, past_key_values=past_key_values,
                attention_mask=attention_mask, position_ids=position_ids, use_cache=True)
        finally:
            if collector is not None:
                collector.remove()
        cache = output.past_key_values
        if collector is None:
            return TargetContext(
                cache=cache, cache_length=int(cache.get_seq_length()), current_token=-1,
                features=[], last_hidden_state=output.last_hidden_state.detach(),
                next_logits=self.target_logits(output.last_hidden_state[:, -1]),
                verification_anchor_time_s=None, verification_anchor_audio_index=None,
                verification_peak_weight=None, q_from_current_verification=False)
        if set(collector.features) != set(FEATURE_LAYERS):
            raise AssertionError('a strict restart feature was not captured from the target forward')
        if self.capture_runtime_attention:
            anchor_start = anchor_end = None
            if self.component_timing is not None:
                anchor_start = torch.cuda.Event(enable_timing=True); anchor_start.record()
            index, time_s, weight = collector.runtime_anchor_from_existing_cache(cache)
            if self.component_timing is not None:
                anchor_end = torch.cuda.Event(enable_timing=True); anchor_end.record()
                self.component_timing.setdefault('anchor_extraction_events', []).append((anchor_start, anchor_end))
        else:
            index = time_s = weight = None
        return TargetContext(
            cache=cache, cache_length=int(cache.get_seq_length()), current_token=-1,
            features=[collector.features[layer] for layer in FEATURE_LAYERS],
            last_hidden_state=output.last_hidden_state.detach(),
            next_logits=self.target_logits(output.last_hidden_state[:, -1]),
            verification_anchor_time_s=time_s, verification_anchor_audio_index=index,
            verification_peak_weight=weight, q_from_current_verification=self.capture_runtime_attention)


    @torch.inference_mode()
    def append_target_token(self, context: TargetContext, token: int, *,
                            capture_features: bool = True) -> TargetContext:
        self.forward_counts['context_update'] += 1
        token_tensor = torch.tensor([[int(token)]], device=self.device)
        embeds = self.target.thinker.get_input_embeddings()(token_tensor)
        next_context = self._forward(inputs_embeds=embeds, past_key_values=context.cache,
                                     capture_runtime_attention=capture_features)
        next_context.current_token = int(token)
        return next_context


    @torch.inference_mode()
    def verify_block(
            self, context: TargetContext, candidates: torch.Tensor
    ) -> tuple[int, list[int], list[int], TargetContext | None]:
        """Verify one variable-length draft block and commit canonical output.

        The target candidate block is the single target verification forward.
        The subsequent one-token context append is part of ordinary greedy
        sequence progression in *every* matched condition; the runtime arm
        only reads its already-produced L21 query/KV cache and introduces no
        additional forward.
        """
        if candidates.ndim != 1 or candidates.numel() < 1:
            raise ValueError('cached loop requires a non-empty 1-D draft proposal')
        proposal_length = int(candidates.numel())
        self.forward_counts['candidate_verification'] += 1
        candidate_embeds = self.target.thinker.get_input_embeddings()(candidates[None])
        block = self._forward(inputs_embeds=candidate_embeds, past_key_values=context.cache,
                              capture_runtime_attention=False)
        # Logits at the existing current state predict d1; each candidate state
        # predicts the following token.  ``expected[3]`` is the target token
        # committed after an all-accepted proposal block.
        expected = [int(context.next_logits.argmax(dim=-1)[0])]
        candidate_next = self.target_logits(block.last_hidden_state[0]).argmax(dim=-1)
        expected.extend(int(value) for value in candidate_next.tolist())
        eos = int(self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>'))
        accepted = 0
        committed: list[int] = []
        for candidate, target_token in zip(candidates.tolist(), expected[:proposal_length]):
            if int(candidate) != int(target_token):
                break
            accepted += 1
            committed.append(int(candidate))
            # EOS is itself a committed, accepted draft token.  A speculative
            # round ends here; there is no post-block correction token.  The
            # old path fell through and appended ``expected[accepted]``, which
            # made lossless outputs continue beyond the target's EOS.
            if int(candidate) == eos:
                context.cache.crop(context.cache_length + accepted)
                return accepted, expected, committed, None
        correction_token = int(expected[accepted])
        # Candidate states beyond the accepted prefix must never enter the
        # next target or drafter state.  DynamicCache.crop performs this on all
        # decoder layers.
        context.cache.crop(context.cache_length + accepted)
        committed.append(correction_token)
        if correction_token == eos:
            return accepted, expected, committed, None
        next_context = self.append_target_token(context, correction_token)
        return accepted, expected, committed, next_context
