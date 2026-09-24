# Third-party components

- Qwen3-ASR target implementation/model: installed from `qwen-asr`, with pinned Hugging Face model revisions. Upstream source: https://github.com/QwenLM/Qwen3-ASR . No vendored target implementation or weights are included.
- PyTorch and TorchAudio: tensor operations, attention kernels, and optional MMS-FA preprocessing. MMS model weights are obtained through upstream tooling under their applicable terms.
- Hugging Face Transformers / Hub / safetensors, NumPy, SoundFile and librosa are declared dependencies. They retain their upstream licenses.
- LibriSpeech, TED-LIUM, GigaSpeech and FLEURS are external datasets. Identity metadata are provided; audio and transcript payloads are not bundled. Users must obtain the data through the official sources and comply with applicable access conditions.

The ProgDraft code license is TBD. This file does not grant rights to third-party models or datasets and does not substitute for the future project LICENSE.
