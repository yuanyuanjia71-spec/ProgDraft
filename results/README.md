# Historical Ours results

Final Test 1000, 200 utterances per dataset, inference K=8. Checkpoints were trained with Random-K[3,8]. These are the existing paper-table measurements, not a new benchmark of the extracted package. E2E includes the independent correction/bonus forward. Mean Accepted includes actual correction/bonus tokens.

| Target | Dataset | Mean Accepted | E2E speedup | WER | CER |
|---|---|---:|---:|---:|---:|
| Qwen3-ASR-0.6B | librispeech_clean | 5.055 | 1.786× | 1.99% | 0.60% |
| Qwen3-ASR-0.6B | librispeech_other | 4.553 | 1.624× | 4.57% | 1.92% |
| Qwen3-ASR-0.6B | tedlium | 4.867 | 1.737× | 10.17% | 2.17% |
| Qwen3-ASR-0.6B | gigaspeech | 4.533 | 1.633× | 8.38% | 4.32% |
| Qwen3-ASR-0.6B | fleurs_en_us | 4.135 | 1.504× | 5.40% | 2.79% |
| Qwen3-ASR-1.7B | librispeech_clean | 4.009 | 1.289× | 1.21% | 0.28% |
| Qwen3-ASR-1.7B | librispeech_other | 3.752 | 1.224× | 3.46% | 1.42% |
| Qwen3-ASR-1.7B | tedlium | 3.967 | 1.291× | 10.15% | 2.32% |
| Qwen3-ASR-1.7B | gigaspeech | 3.751 | 1.216× | 8.66% | 4.58% |
| Qwen3-ASR-1.7B | fleurs_en_us | 3.416 | 1.116× | 4.11% | 1.87% |

Source CSV retains unrounded metrics and source paths. Every historical arm included here passed 1,000/1,000 token-sequence equality against its corresponding frozen target-only run. No cross-scale identity is implied.

Tests use licensed official-split audio identified in `../manifests/`. No test-set tuning or new measurements were performed as part of packaging.
