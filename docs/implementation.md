# Exact implementation contract

## Models

| Setting | Qwen3-ASR-0.6B | Qwen3-ASR-1.7B |
|---|---:|---:|
| Decoder blocks / initialization block | 28 / 21 (zero based) | 28 / 21 (zero based) |
| Hidden size | 1024 | 2048 |
| Drafter attention heads / FFN size | 16 / 2048 | 16 / 4096 |
| Drafter parameters | 17,846,272 | 71,344,128 |
| Shared progress predictor parameters | 591,105 | 1,115,393 |
| Progress predictor widths | 1025→512→128→1 | 2049→512→128→1 |
| Last utterance batch of each epoch | keep 9 | repeat first 3 shuffled utterances, giving 12 |

The drafter block is LayerNorm/self-attention with residual, LayerNorm/cross-attention with residual, LayerNorm/FFN with residual, then output LayerNorm. All affine LayerNorms and linear biases are retained. Restart concatenates the current embedding and four target block outputs, then projects 5H→H. Continuation adds the previous drafter state and current token embedding. Self-KV grows one entry per step; every step recomputes cross-K/V from the full frozen audio memory.

`features=[F_0(t), F_9(t), F_18(t), F_27(t)]` are raw zero-based decoder block forward outputs, before the target's final RMSNorm, at the same position as the current token. This selection is an implementation choice. It is not attributed as a setting specified by another paper. `target_feature` for the d1 SmoothL1 loss is the decoder's **final normalized hidden state at t**. The d1 token label is y(t+1).

## Acoustic state and timing coordinates

The target decoder has no separate cross-attention module. It reads audio placeholder keys through causal self-attention. L21 initialization captures the current committed token's normalized, RoPE-transformed query during the existing target forward. Attention is recomputed against the same layer's live KV cache. Softmax normalization includes **all visible keys**, then only valid audio placeholder probabilities are selected and averaged over heads. There is no per-head audio-only renormalization. The mean distribution's peak initializes a1.

At the first round, the query is the prompt's last token. After acceptance/rejection it is the actually committed correction/bonus token. The next round therefore advances by its actual committed length, never by a fixed multiple of K. No extra anchor-only target forward is performed.

The historical Ours time grid is exactly `(j+0.5)*audio_duration/audio_memory_length`. It is a **uniform duration-based midpoint approximation**, not an encoder receptive-field timestamp reconstruction. This release preserves it. Changing this grid would be a new experimental setting.

For k≥2, `delta_k=Softplus(MLP([h_pre, a_(k-1)/duration]))`, `a_k=a_(k-1)+delta_k`. The MLP uses GELU after its first two linear layers. Raw acoustic positions are never detached or reset with FA. Only the center used in the Gaussian bias is clipped to `[0,duration]`; recursion and supervision use raw positions. The full-memory bias is `-(time_j-center)^2/(2*0.2^2)` with no additional coefficient or hard window. d1 has no Gaussian bias.

## Random horizon and exact loss

Each source anchor independently samples K uniformly from {3,4,5,6,7,8} using NumPy PCG64 seed 0. Actual depth is clipped by the remaining target tokens, including terminal EOS. The original cache creates `len(target_tokens)-2` anchors: anchors having fewer than three future labels are not source anchors, but deeper steps near EOS are clipped. Each epoch's actual horizons are planned before updates. `p_k` is the empirical fraction of all epoch anchors that reach depth k.

For a batch of utterances, let N denote all source anchors before horizon sampling, N_FA,k the count of source anchors with an available valid FA label at depth k before horizon sampling, and S_k the active anchors at depth k. `w_k=0.7^(k-1)`. The objective is:

```text
L_token = sum_k (w_k/p_k) * sum_{i in S_k} CE_i,k / N
L_feat  = sum_i SmoothL1(z_i,1, final_target_hidden_i,t) / (N*H)
L_prog  = sum_{k>=2} (w_k/p_k)
          * sum_{i in S_k, FA-valid} SmoothL1(raw_a_i,k, FA_i,k)
          / max(N_FA,k, 1)
L = L_token + 0.5*L_feat + 0.1*L_prog
```

The reduction is not an unweighted mean over sampled depths and is not renormalized by sum(w). The source-batch denominators are fixed before anchor microbatching (24 anchors). The third token depth weight is 0.49 in this final Random-K recipe, not the older K=3 recipe's 0.5. d1 feature matching is computed once per anchor; d1 has no progress supervision.

Token inputs at step k≥2 are target-greedy gold token y(t+k−1), not argmax from the previous draft. Acoustic positions, drafter hidden states and self-KV remain connected across depths. CE backpropagates through the cross-attention Gaussian center into the predictor. A gradient-capable math SDPA backend is used when the Gaussian center requires gradients; frozen inference follows the original SDPA path.

## Forced alignment

MMS-FA aligns the target-generated transcript. The English alignment alphabet is lowercase `a-z` plus apostrophe. Cumulative token prefixes are decoded with `skip_special_tokens=True` and no tokenization-space cleanup, then restricted to that alphabet. A token maps to its newly appended character span; a prefix rewrite or no new character gives an invalid mapping. Whitespace, ordinary punctuation, and special tokens without surviving characters are invalid. An apostrophe can survive this alphabet. A span's position is `(first_character_start + last_character_end)/2`.

Invalid mappings contribute zero progress loss but retain token CE. This is the existing Ours span-midpoint mapping, not AnchorDraft's last-new-character/punctuation-carry mapping. No FA data enters inference.

## Optimization

Both configurations jointly optimize the full drafter and the progress predictor. Target, embeddings, LM head, and audio path are frozen. Batch size is 12 utterances, 45 epochs / 14,760 updates, seed 0, AdamW lr 1e−3, weight decay 0.01, gradient clipping 1.0, cosine T_max=14,760. Restart feature noise is independent Gaussian noise with standard deviation 0.6 on the four frozen target features only. See the final-batch difference in the table above.
