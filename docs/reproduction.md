# Reproduction

## Exact historical assets

The public download location for exact checkpoints/caches is TBD. For the authors' existing research checkout, export without changing any original tensors:

```bash
python scripts/export_research_assets.py --source-root /path/to/research-checkout \
  --scale 0.6b --output artifacts/0.6b --trusted-local-pickle
python scripts/export_research_assets.py --source-root /path/to/research-checkout \
  --scale 1.7b --output artifacts/1.7b --trusted-local-pickle
```

The source-root argument is explicit; this converter does not import the old experiment modules. Only trusted local `.pt` files may be converted. Public weight loading uses safetensors. The converter writes tensor-only initial/final weights, portable utterance caches, a relative-path manifest and source hashes. Raw audio and original target weights are not copied. `--weights-only` omits caches; `--limit` creates an incomplete smoke fixture and cannot be used for the full training run.

Run the README training command with the full manifest. For 1.7B substitute its config and assets. `--resume runs/.../checkpoints/step_XXXXX.pt` restores optimizer, scheduler, CPU/CUDA RNG, utterance shuffle and the preplanned per-anchor K values. Resume checkpoints are local trusted PyTorch files; never load unknown downloaded pickle checkpoints. The portable trainer evaluates teacher-forced depth metrics at saved checkpoints and does not perform runtime tuning.

The objective, model operation order and cache verifier are extracted from the recorded source revision (see `source_provenance.json`). The portable trainer is a packaging refactor; a new 45-epoch training run has **not** been performed to establish end-to-end bitwise equality. Model/loss/gradient and runtime smoke equivalence are checked separately. Future environment changes or newly generated caches can produce different training trajectories.

## Standalone cache preparation on licensed audio

Prepare a JSONL with `id`, `split` (`train` or `validation`), and `audio` relative to the manifest. Reference transcripts do not enter target greedy generation. Run:

```bash
for phase in trajectories features alignment; do
  progdraft-prepare --phase "$phase" --config configs/qwen3_asr_0.6b.json \
    --manifest data/train_val_audio.jsonl --output data/prepared
done
```

The builder runs FP32/eager target-only greedy decoding, BF16 same-position feature extraction with causal L21 attention, and offline MMS-FA span midpoint labeling in separate phases. Only after the alignment phase does it write a training manifest. The source target token IDs are stored directly; text is not decoded and re-encoded as training tokens.

This builder makes new compatible caches. It does not recreate the historical 0.6B batched feature extraction bit for bit, the original sample selection process, or the original target-generated trajectory automatically. Use exported immutable caches for comparisons to the paper's historical checkpoints. The fixed IDs/checksums in `manifests/` identify the intended dataset membership. Users must acquire the corresponding audio under the upstream dataset terms.

## Testing

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

CPU tests check teacher forcing, cumulative KV lengths, recursive positions, the full loss reduction, EOS clipping, FA mapping, and token-CE-only predictor gradients. With ASR extras installed, verifier tests cover first rejection, all-accepted bonus and EOS termination using the actual verifier method. Real-target parity checks used in packaging are described in `validation.md`.
