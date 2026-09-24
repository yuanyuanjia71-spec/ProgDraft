# Release validation — 2026-09-24

This records packaging checks, not a new training experiment or a replacement Final Test benchmark.

## Automated contracts

Six tests pass in the tested environment: exact Random-K loss reduction; cumulative-prefix FA span mapping; EOS horizon clipping and empirical depth probabilities; teacher-forced token indexing and self-KV/position recursion; token-CE-only gradient connectivity; real verifier branch/crop behavior covering first rejection, full acceptance, bonus and EOS. The verifier unit test uses mocked target outputs but executes the actual extracted verifier function.

## Source equivalence

`source_equivalence_audit.json` records two scales, two real exported training anchors, horizons 3 and 8, source checkpoint tensors and the actual frozen target embedding/LM head. The source and released drafter produced exactly equal states, acoustic positions, observation records and total loss. Parameter counts match both research models.

With feature/progress losses excluded, CE-only backward at the original step-0 initialization gives predictor gradient norms approximately **0.09826 (0.6B)** and **0.54441 (1.7B)**. This verifies a live gradient path. On the tiny trained-checkpoint fixture the norms were 0.73565 and 0 respectively; the latter's displacement outputs were numerically zero on that fixture. The audit does not claim a nonzero local derivative for every trained example, or infer dataset-wide behavior from two anchors. No optimizer update was performed.

## Real target/cache replay

`runtime_equivalence_audit.json` records one fixed Final Test utterance from each of the five datasets, for each scale, at inference K=8. All **10/10** checks passed:

- Original Ours token IDs equal released Ours token IDs.
- Released Ours token IDs equal the corresponding target-only greedy IDs.
- Speculative round counts and accepted-draft counts equal the source run.

The check used existing frozen final weights and the actual target KV cache. It measured correctness only; no new speedup is reported. It does not replace the historical full 1,000-utterance audit in `results/`.

## Limits

The portable 45-epoch trainer has not been rerun end to end. The generic standalone cache builder has not regenerated the full training dataset. Complete training data, upstream target snapshots, exact initialization weights and final weights remain external assets. The local two-record cache exports are smoke fixtures only and are ignored by Git. Tokenizer/backend versions are pinned; the existing tokenizer regex warning was observed and its behavior was not silently changed during extraction.

The source release excludes unrelated experiments, local paths, audio, transcripts, upstream weights and optimizer checkpoints. Public download links, the formal project license and paper URL still need to be supplied by the authors.
