# ProgDraft: Acoustic Progress Propagation for Speculative ASR

Code for **Acoustic Progress Propagation for Long-Horizon Speculative Decoding in ASR**.

**Yuanyuan Jia, Qianqian Yang — Zhejiang University**

Planned repository: https://github.com/yuanyuanjia71-spec/ProgDraft

Paper / preprint: **TBD**

License: **TBD** — a formal `LICENSE` will be added after the authors choose it. No open-source license is granted by this placeholder.

## Method

ProgDraft jointly trains a lightweight recurrent ASR drafter and a scalar acoustic progress predictor. The frozen Qwen3-ASR target supplies restart features, token embeddings, the LM head, and full audio memory. A target attention peak initializes the acoustic position once per speculative round. At subsequent draft steps, a shared predictor produces nonnegative displacements; their recursively accumulated positions supply an additive Gaussian cross-attention bias.

The released method is **Joint Progress-Aware + Random-K[3,8]**, with Qwen3-ASR-0.6B and Qwen3-ASR-1.7B configurations. It does not include other experimental adapters, AnchorDraft, or rejected variants.

```text
current committed target state
   ├─ token embedding + block outputs [0,9,18,27] → d1 → z1
   └─ L21 current-query attention → initial acoustic position a1

for k ≥ 2:
   previous z + token embedding → self-attention → h_pre
   [h_pre, a_(k-1)/duration] → shared MLP → Softplus → delta_k
   a_k = a_(k-1) + delta_k
   full audio cross-attention with Gaussian(center=a_k, sigma=0.2 s)
   → FFN → z_k → frozen LM head
```

Training uses teacher-forced tokens from the frozen target's greedy trajectory, while hidden states, self-KV, and acoustic positions recurse through the drafter. Runtime uses predicted tokens. FA positions are training labels only. See [the exact implementation contract](docs/implementation.md).

## Installation

The tested research environment uses Python 3.11, PyTorch 2.11.0 with CUDA 12.8, Transformers 4.57.6 and qwen-asr 0.0.6. Install the appropriate PyTorch wheel for your machine, then:

```bash
python -m pip install -e '.[asr]'
python -m unittest discover -s tests -v
```

MMS-FA preprocessing additionally needs a matching TorchAudio installation (`pip install -e '.[alignment]'`). Target weights are downloaded from the pinned upstream revisions in `configs/`. No upstream model source or weights are bundled.

## Assets and reproduction status

This is a source release prepared from the completed experiments. Download URLs for the exact initialization weights, final weights, frozen feature caches and paper are **TBD**. They must not be inferred from the planned repository address. The code includes a trusted-local-asset conversion tool; tensor-only weight exports are staged locally under ignored `artifacts/`, not in Git.

- [Asset inventory and data access](docs/data_and_assets.md)
- [Training and reproduction](docs/reproduction.md)
- [Runtime and timing protocol](docs/runtime.md)
- [Historical results and their provenance](results/README.md)
- [Release validation](docs/validation.md)

Historical experiment results below were measured with the research implementation. They are not new full-test measurements of this extracted package. Package parity checks are documented separately.

## Inference

After obtaining a tensor-only Ours checkpoint:

```bash
progdraft-decode --config configs/qwen3_asr_0.6b.json \
  --weights artifacts/0.6b/ours.safetensors --audio example.wav --k 8
```

Use `configs/qwen3_asr_1.7b.json` and the matching weights for the larger target. `--target-path` optionally points to an already downloaded, matching upstream snapshot. Never mix scale-specific checkpoints or target revisions.

## Training

```bash
progdraft-train --config configs/qwen3_asr_0.6b.json \
  --initial-weights artifacts/0.6b/step0.safetensors \
  --manifest artifacts/0.6b/manifest.jsonl --output runs/ours_0.6b
```

This requires the complete 3,933-training / 250-validation cache, not the two-record release smoke fixture. The scheduler horizon remains 14,760 even when using `--stop-after` for debugging. [Reproduction instructions](docs/reproduction.md) explain asset export, preparation on new licensed data, resume behavior, and the preserved scale-specific final-batch policy.

## Paired runtime benchmark

```bash
progdraft-benchmark --config configs/qwen3_asr_0.6b.json \
  --weights artifacts/0.6b/ours.safetensors \
  --manifest data/test.jsonl --output runs/test_k8 --k 8
```

Each test JSONL record supplies `id`, `dataset`, `audio` (relative to the manifest), and `reference`. Reference text is used only for scoring. The benchmark warms up both methods, alternates their order, uses native batched target verification and an independent correction/bonus-token forward, and fails on token-ID divergence from target-only AR. A new output directory is required for each run.

Reported columns are **Mean Accepted, E2E speedup, WER, CER**. Mean Accepted includes actual correction/bonus tokens; EOS rounds do not receive an artificial bonus. The paper runs use physical GPU0, batch size one and concurrency one. Run on an otherwise idle GPU.

## Acknowledgments and citation

The frozen target is [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), and offline alignment uses [TorchAudio MMS-FA](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html). Their code, models and datasets retain their respective terms. See [third-party notices](THIRD_PARTY_NOTICES.md).

The paper has not been assigned a public link or publication record in this release. Add a BibTeX entry when those details are available; do not cite a fabricated DOI or conference acceptance.
