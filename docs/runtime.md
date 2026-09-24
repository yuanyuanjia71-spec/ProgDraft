# Runtime and numerical protocol

[Documentation](README.md) / Runtime and evaluation

## Run a paired benchmark

Create a JSONL manifest with one record per utterance. Audio paths are relative to the manifest; references are used only for WER/CER scoring:

```json
{"id":"example-001","dataset":"example-test","audio":"audio/example.wav","reference":"the reference transcript"}
```

The checked-in [identity manifests](../manifests/) identify dataset membership and are not runtime manifests. Obtain the corresponding audio and references through the dataset providers, then construct the input records above.

```bash
progdraft-benchmark \
  --config configs/qwen3_asr_0.6b.json \
  --weights artifacts/0.6b/ours.safetensors \
  --manifest data/test.jsonl \
  --output runs/test_k8 \
  --device cuda:0 \
  --k 8
```

Use a new output directory for each run. The command writes:

| File | Contents |
| :--- | :--- |
| `config.json` | Configuration, weight/manifest hashes and timing protocol |
| `per_utterance.jsonl` | Outputs, timings and counters for both methods |
| `status.json` | Completed utterance count |
| `metrics.csv` | WER, CER, E2E speedup and Mean Accepted, overall and per dataset |
| `summary.json` | Aggregate metrics and exact-output consistency |
| `divergence.json` | Written on a token mismatch; the run stops before aggregate speedup |

## Verification and numerical settings

The released runtime uses actual free-running drafter tokens, a live target KV cache, one batched verification forward, accepted-prefix cache cropping, and an independent correction/bonus-token target forward when generation continues. This last forward is retained deliberately and included in timing. Progress prediction, L21 score reconstruction, Gaussian bias and all cache operations are also included. There is no canonical stepwise guard, tree verifier, FA input or hidden target replay.

At each round, the current state's logits score candidate d1; the target output at candidate k scores candidate k+1. The first mismatch ends acceptance. After cropping rejected candidates, the target correction is appended; if all candidates match, the bonus token is appended. Accepted EOS stops immediately without a bonus. Target-only and Ours use the same prompt, audio preparation, tokenizer and stopping conditions.

The tested exact-output configuration is an FP32/eager target with FP32 logit readout, BF16 drafter autocast, and frozen weights. BF16 target or different kernels may change greedy tokens through numerical rounding. Exact equality must be rechecked on a new environment; the benchmark fails before publishing a speedup if any token list differs.

## Timing and metrics

Timing starts before waveform loading and processor/audio encoding for E2E, after audio preparation for decode, and after prompt prefill for generation. Boundary CUDA synchronizations are present; there is no artificial synchronization inside every token. Both arms warm up first, then alternate order per utterance. Model loading, warm-up and corpus scoring are excluded. Tokenizer decoding is included. The measured ratio is sum(AR E2E)/sum(Ours E2E), not the average of per-utterance ratios.

Mean Accepted (tau) is total actually emitted token IDs / speculative rounds. Actual correction/bonus and EOS are counted; a fictitious post-EOS token is never added. Raw `accepted_sum` retains its narrower accepted-draft-token definition for audits.

The research benchmarks used an otherwise idle NVIDIA H100 80GB, physical GPU0, batch size 1, concurrency 1. The release smoke checks verify correctness only. The generic package allows another CUDA device, but comparisons must use the same idle GPU and protocol.

WER/CER are corpus-level edit distances under the exact source normalization: convert GigaSpeech punctuation markers, remove special tags, NFKC, lowercase, normalize curly apostrophe, replace other punctuation by spaces, collapse whitespace. CER removes spaces. Test references are passed only to scoring.
