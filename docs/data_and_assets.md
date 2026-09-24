# Data and assets

[Documentation](README.md) / Data and assets

## Availability

| Asset | Current status | Use |
| :--- | :--- | :--- |
| Source and model configurations | Included | Implementation and commands |
| Dataset IDs, revisions and checksums | [Included](../manifests/) | Fixed split membership |
| Frozen Qwen3-ASR target | Downloaded from the pinned upstream revision | Training and inference |
| ProgDraft final weights, both scales | Public download pending | Inference and evaluation |
| Original step-0 weights and frozen caches | Public download pending | Exact historical training replay |
| Raw audio and transcripts | Obtain from the dataset providers | Data preparation and scoring |

For each scale, the expected portable weight names are `ours.safetensors` and `step0.safetensors`. Their SHA-256 checksums are in the [checkpoint inventory](checkpoint_inventory.json). These files contain our drafter/predictor tensors, not the frozen target weights. Local `artifacts/`, `data/` and `runs/` directories are ignored by Git.

## Fixed datasets

The historical training mixture contains 3,933 utterances: 1,797 LibriSpeech Clean, 598 LibriSpeech Other, 598 TED-LIUM, 300 GigaSpeech and 640 FLEURS. Validation contains 250 fixed utterances. Final Test contains 1,000 fixed utterances, 200 each from LibriSpeech test-clean, LibriSpeech test-other, TED-LIUM 3 test, GigaSpeech test and FLEURS en-us test.

`manifests/` contains source IDs, dataset revisions, source rows/shards, split memberships and waveform hashes. It omits transcripts, audio, machine-local paths and generated features. These are **identity manifests**, not ready-to-run audio manifests. Download datasets through their official access procedures, including authorization where needed, then construct relative-path input manifests. No audio is redistributed here.

## Portable training cache

For exact training replay, each portable utterance cache has:

| Key | Shape / meaning |
|---|---|
| `id`, `duration_s`, `token_ids` | source ID, seconds, target-generated tokens including terminal EOS |
| `audio_memory` | [M,H], frozen final target audio representation |
| `features` | [N,4,H], raw outputs of target blocks 0/9/18/27 at position t |
| `target_feature` | [N,H], final normalized target state at t |
| `current_tokens` | [N], current token at t |
| `a1_verification_s` | [N], current-position L21 mean-head peak time |
| `fa_positions`, `fa_valid` | [len(token_ids)], span-midpoint labels and validity |
| `contract` | `same_position_target_greedy_l21_span_midpoint_v1` |

N is len(token_ids)−2. Cache rows are keyed by source anchor position, not by speculative round number. Online runtime never reads these training caches; it recomputes the initialization from the current committed target context.

Exact step-0/final Ours weight release URLs, cache hosting and dataset reconstruction download automation remain TBD. Their checksums and local conversion tooling are provided. Model tensors exported here contain only our drafter/predictor, plus initialization RNG when applicable; no target weights or optimizer state are included in public-format weight exports.
