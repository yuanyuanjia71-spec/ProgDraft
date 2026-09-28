# Third-party components

- Qwen3-ASR target implementation/model: installed from `qwen-asr`, with pinned Hugging Face model revisions. Upstream source: https://github.com/QwenLM/Qwen3-ASR . No vendored target implementation or weights are included.
- PyTorch and TorchAudio: tensor operations, attention kernels, and optional MMS-FA preprocessing. MMS model weights are obtained through upstream tooling under their applicable terms.
- Hugging Face Transformers / Hub / safetensors, NumPy, SoundFile and librosa are declared dependencies. They retain their upstream licenses.
- Gradio provides the optional local visualization UI. It is installed as a dependency; no upstream UI source is vendored.
- LibriSpeech, TED-LIUM, GigaSpeech and FLEURS are external datasets. No dataset audio or bulk transcripts are bundled. The project-page demonstration includes a single short LibriSpeech test-clean transcript excerpt (sample 6930-75918-0003) derived from its public-domain LibriVox source, with waveform-envelope values and token/position traces. Users must obtain training/test data through the official sources and comply with applicable access conditions.
- The academic project page adapts the hero/teaser structure of [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) by Eliahu Horwitz, partly based on Nerfies. The adapted website layout and player are CC BY-SA 4.0; see [website attribution](site/SITE_LICENSE.md). This does not change the research-code license.

The ProgDraft code license is TBD. This file does not grant rights to third-party models or datasets and does not substitute for the future project LICENSE.
