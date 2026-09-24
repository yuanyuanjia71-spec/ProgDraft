# Installation

[Documentation](README.md) / Installation

## Tested environment

| Component | Version / setting |
| :--- | :--- |
| Python | 3.11 |
| PyTorch | 2.11.0 |
| CUDA build | 12.8 |
| Transformers | 4.57.6 |
| qwen-asr | 0.0.6 |
| Research benchmark GPU | NVIDIA H100 80GB |

GPU training and inference require CUDA. CPU contract tests do not load target weights or audio data. Recheck output equality and timing when changing hardware or numerical libraries; see [runtime](runtime.md).

## Install for inference and training

Create and activate a Python 3.11 environment. Install a PyTorch 2.11 build compatible with your CUDA environment, then:

```bash
git clone https://github.com/yuanyuanjia71-spec/ProgDraft.git
cd ProgDraft
python -m pip install -e '.[asr]'
```

The ASR extra installs the pinned target backend, tokenizer and audio dependencies. Check that the entry points are available:

```bash
progdraft-prepare --help
progdraft-train --help
progdraft-decode --help
progdraft-benchmark --help
```

Target snapshots are downloaded on first use from the exact revision in the selected configuration. Use `--target-path /path/to/snapshot` to point to a matching local snapshot. ProgDraft weights are separate assets; see [data and assets](data_and_assets.md).

## Optional forced alignment

MMS-FA is needed only for preparing training supervision. Install TorchAudio matching the installed PyTorch version and CUDA build, then install the alignment extra:

```bash
python -m pip install -e '.[asr,alignment]'
```

Inference and runtime benchmarking do not use MMS-FA.

## CPU checks

For model/loss checks without the target backend:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The verifier test is skipped when the ASR backend is unavailable. Install `.[asr]` to exercise all six tests, including mocked-output checks of the actual verifier. These tests do not download model weights or require a GPU.
