# Release validation — 2026-09-24

[Documentation](README.md) / Release validation

This records packaging checks, not a new training experiment or a replacement Final Test benchmark.

## Automated contracts

The original six tests cover exact Random-K loss reduction; cumulative-prefix FA span mapping; EOS horizon clipping and empirical depth probabilities; teacher-forced token indexing and self-KV/position recursion; token-CE-only gradient connectivity; real verifier branch/crop behavior covering first rejection, full acceptance, bonus and EOS. The verifier unit test uses mocked target outputs but executes the actual extracted verifier function. Five additional demo tests cover observational parity, token rendering, decode events, exact-match failures and initial UI updates (11 tests with the pinned demo extra).

## Source equivalence

[The source equivalence audit](audits/source_equivalence_audit.json) records two scales, two real exported training anchors, horizons 3 and 8, source checkpoint tensors and the actual frozen target embedding/LM head. The source and released drafter produced exactly equal states, acoustic positions, observation records and total loss. Parameter counts match both research models.

With feature/progress losses excluded, CE-only backward at the original step-0 initialization gives predictor gradient norms approximately **0.09826 (0.6B)** and **0.54441 (1.7B)**. This verifies a live gradient path. On the tiny trained-checkpoint fixture the norms were 0.73565 and 0 respectively; the latter's displacement outputs were numerically zero on that fixture. The audit does not claim a nonzero local derivative for every trained example, or infer dataset-wide behavior from two anchors. No optimizer update was performed.

## Real target/cache replay

[The runtime equivalence audit](audits/runtime_equivalence_audit.json) records one fixed Final Test utterance from each of the five datasets, for each scale, at inference K=8. All **10/10** checks passed:

- Original Ours token IDs equal released Ours token IDs.
- Released Ours token IDs equal the corresponding target-only greedy IDs.
- Speculative round counts and accepted-draft counts equal the source run.

The check used existing frozen final weights and the actual target KV cache. It measured correctness only; no new speedup is reported. It does not replace a full 1,000-utterance benchmark.

## Demo checks — 2026-09-28

The Gradio demo was checked on GPU 0 with the requested paper-table AnchorDraft
(fixed-K3 training, source SHA-256 `1e04449b6ca9f5bcd46abf89a4f40a671ca2c200d6eb274af00f6f39ca1bc32f`)
and the released 0.6B Ours Random-K[3,8] export, both at inference K=8. The first
manifest sample from each of the five Final Test datasets was used, without
outcome-based selection. On all five, both methods' observed and callback-free
outputs equaled target-only AR token IDs; round counts and accepted-draft counts
also matched between observed and unobserved passes. The UI was exercised in
headless Chromium through preset selection, WAV upload, streaming completion and
round review. No training or weight updates were performed. This verifies the
demo integration, not a new full-dataset speed benchmark.

## Limits

The portable 45-epoch trainer has not been rerun end to end. The generic standalone cache builder has not regenerated the full training dataset. Complete training data, upstream target snapshots, exact initialization weights and final weights remain external assets. The local two-record cache exports are smoke fixtures only and are ignored by Git. Tokenizer/backend versions are pinned; the existing tokenizer regex warning was observed and its behavior was not silently changed during extraction.

The source release excludes unrelated experiments, local paths, audio, transcripts, upstream weights and optimizer checkpoints. Public download links, the formal project license and paper URL still need to be supplied by the authors.
