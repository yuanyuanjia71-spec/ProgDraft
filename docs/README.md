# Documentation

[Project home](../README.md) · [简体中文](../README.zh-CN.md)

ProgDraft implements acoustic progress propagation with Random-K[3,8] training for frozen Qwen3-ASR targets. Choose a guide by task:

| Task | Guide |
| :--- | :--- |
| Set up an environment or run CPU checks | [Installation](installation.md) |
| Locate the required weights, caches and dataset IDs | [Data and assets](data_and_assets.md) |
| Transcribe an audio file | [Inference](inference.md) |
| Prepare caches, train or resume | [Reproduction](reproduction.md) |
| Benchmark against target-only decoding | [Runtime and evaluation](runtime.md) |
| Inspect the model, indexing or exact losses | [Implementation reference](implementation.md) |
| Understand what has been verified | [Release validation](validation.md) |
| Report an issue or contribute | [Development guide](../CONTRIBUTING.md) |

## Reading the release

The JSON files in [configs/](../configs/) define the two scale-specific recipes. The code in [src/progress_asr/](../src/progress_asr/) provides four commands: `progdraft-prepare`, `progdraft-train`, `progdraft-decode` and `progdraft-benchmark`.

The manifests identify fixed dataset membership; they do not contain audio, transcript payloads or generated features. Public weights and cache download links are pending. Installation and CPU tests can be run independently of those assets.

## Reproducibility records

- [Checkpoint inventory](checkpoint_inventory.json): initialization and final-weight checksums.
- [Source provenance](audits/source_provenance.json): research revision and source-file hashes.
- [Model and loss equivalence](audits/source_equivalence_audit.json): packaging checks against the research implementation.
- [Target/cache replay equivalence](audits/runtime_equivalence_audit.json): ten real-audio comparisons across both model scales.

These records document the scope of the source release. They are not a replacement for full training or Final Test evaluation.
