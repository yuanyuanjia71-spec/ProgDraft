# Inference

[Documentation](README.md) / Inference

## Required assets

1. Install the [ASR extra](installation.md).
2. Obtain the ProgDraft checkpoint for the chosen target scale. Public download links are pending; filenames and hashes are in the [checkpoint inventory](checkpoint_inventory.json).
3. Supply an audio file. The runtime prepares the waveform, processor inputs and frozen target audio representation.

## Transcribe one file

```bash
progdraft-decode \
  --config configs/qwen3_asr_0.6b.json \
  --weights artifacts/0.6b/ours.safetensors \
  --audio data/example.wav \
  --device cuda:0 \
  --k 8
```

For Qwen3-ASR-1.7B, replace both the config and weight path. The configured target revision is fetched automatically, or can be supplied with `--target-path /path/to/matching/snapshot`.

| Argument | Meaning |
| :--- | :--- |
| `--config` | Scale-specific JSON configuration |
| `--weights` | Matching drafter and progress-predictor safetensors |
| `--audio` | Input audio path |
| `--device` | CUDA device; default `cuda:0` |
| `--k` | Maximum draft length per round; default 8 |
| `--target-path` | Optional local target snapshot |

The command prints JSON containing the transcript, token IDs, timings and decoding counters. Use [the paired benchmark](runtime.md) to measure speedup against target-only AR; one standalone decode is not a speedup measurement.

## Generation behavior

Each round starts from the actual committed prefix. Target L21 attention initializes the acoustic position once; subsequent draft steps recursively predict positive displacements. Draft tokens feed the next draft step, and every step attends to the full audio memory. No forced-alignment labels or reference transcript enter inference.

The target verifies candidates in a batch, commits their accepted prefix and generates a correction or bonus token when needed. The released implementation processes that token with a separate target forward before the next round. All of this work is included in runtime timing.

Training horizons are sampled from 3–8. Other inference horizons reuse the same recurrent parameters; they do not imply training at those depths. See the [implementation reference](implementation.md) for state and position recursion.
