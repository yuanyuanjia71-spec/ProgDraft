# Runtime and numerical protocol

The released runtime uses actual free-running drafter tokens, a live target KV cache, one batched verification forward, accepted-prefix cache cropping, and an independent correction/bonus-token target forward when generation continues. This last forward is retained deliberately and included in timing. Progress prediction, L21 score reconstruction, Gaussian bias and all cache operations are also included. There is no canonical stepwise guard, tree verifier, FA input or hidden target replay.

At each round, the current state's logits score candidate d1; the target output at candidate k scores candidate k+1. The first mismatch ends acceptance. After cropping rejected candidates, the target correction is appended; if all candidates match, the bonus token is appended. Accepted EOS stops immediately without a bonus. Target-only and Ours use the same prompt, audio preparation, tokenizer and stopping conditions.

The tested exact-output configuration is an FP32/eager target with FP32 logit readout, BF16 drafter autocast, and frozen weights. BF16 target or different kernels may change greedy tokens through numerical rounding. Exact equality must be rechecked on a new environment; the benchmark fails before publishing a speedup if any token list differs.

Timing starts before waveform loading and processor/audio encoding for E2E, after audio preparation for decode, and after prompt prefill for generation. Boundary CUDA synchronizations are present; there is no artificial synchronization inside every token. Both arms warm up first, then alternate order per utterance. Model loading, warm-up and corpus scoring are excluded. Tokenizer decoding is included. The measured ratio is sum(AR E2E)/sum(Ours E2E), not the average of per-utterance ratios.

Mean Accepted (tau) is total actually emitted token IDs / speculative rounds. Actual correction/bonus and EOS are counted; a fictitious post-EOS token is never added. Raw `accepted_sum` retains its narrower accepted-draft-token definition for audits.

Historical results were measured on an otherwise idle NVIDIA H100 80GB, physical GPU0, batch size 1, concurrency 1. Source `results/` data are historical measurements, not speed claims from the release smoke checks. The generic package allows another CUDA device, but comparisons must use the same idle GPU and protocol.

WER/CER are corpus-level edit distances under the exact source normalization: convert GigaSpeech punctuation markers, remove special tags, NFKC, lowercase, normalize curly apostrophe, replace other punctuation by spaces, collapse whitespace. CER removes spaces. Test references are passed only to scoring.
