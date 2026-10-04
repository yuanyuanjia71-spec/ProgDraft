# Local comparison demo

A Gradio view of **Target-only greedy AR**, **AnchorDraft-only (paper-table checkpoint, fixed-K3 training)** and **Ours: Joint + Random-K[3,8]**, with the same frozen target and audio. The two speculative methods use inference **K=8**. The demo loads existing checkpoints and never trains. It is a single-user CUDA application, bound to localhost by default.

## Launch

From the repository root, in the Python 3.11 / PyTorch 2.11 environment:

```bash
pip install -e '.[asr,demo]'
# Configure local checkpoints first (see below).
python demo/app.py
```

Open `http://127.0.0.1:7860`. For a remote GPU machine, forward port 7860 through SSH. `--port` and `--config` are optional. GPU 0 is the default; keep it free of other workloads when reading timing results. Audio is processed locally; the app creates no public Gradio tunnel.

## Local assets

Checkpoint download links remain pending. A fresh clone cannot run inference until matching weights are supplied; the demo does not silently substitute random weights. The repository contains no test audio or checkpoint binaries.

Copy `demo/config.example.json` to `demo/local_config.json` and set the local paths. Paths are resolved relative to the config file. `ours_weights` is the existing export with `drafter.*` and `progress_predictor.*` tensors. `anchor_weights` contains the same drafter architecture with `drafter.*` keys and no predictor. `target_path` is optional; without it, the pinned target snapshot in the model config is downloaded. Set `anchor_label` to the exact checkpoint variant being compared. The demo always uses AnchorDraft-only, without Runtime Correction.

To export a **trusted local** research AnchorDraft `.pt` and add five deterministic Final Test presets:

```bash
python demo/prepare_local.py \
  --anchor-checkpoint /path/to/anchordraft/checkpoints/step_14760.pt \
  --anchor-label 'Paper-table AnchorDraft, fixed-K3 training, step 14760' \
  --ours-weights artifacts/0.6b/ours.safetensors \
  --manifest /path/to/final_test_1000_manifest.jsonl \
  --audio-root /path/to/research/root \
  --trusted-local-pickle
```

The helper chooses the first sample per dataset in manifest order, exports lossless float WAVs and writes local hashes/provenance. It does not select examples based on acceptance. The manifest expects `dataset`, `audio_npy` (relative to `--audio-root`), and `duration_s`. Omit the manifest arguments for upload-only use. Manually supplied presets use `{ "label": "sample", "path": "/path/to/sample.wav" }`.

Assets go to ignored `artifacts/demo/`; machine paths stay in ignored `demo/local_config.json`. Do not commit local weights or dataset audio. The configuration can also select the matching 1.7B config and weights.

## What the UI measures

- Each real speculative round displays eight proposed token IDs/text pieces. Green is the accepted draft prefix; red is the first rejection; gray is the unaccepted/unverified suffix. Accepted EOS ends a round without inventing a rejection or bonus. Target correction/bonus is shown separately.
- Ours' waveform overlay uses the actual live L21 initialization and recursive positions. d1 attention remains unrestricted; Gaussian correction starts at d2. Raw positions and clamped bias centers are distinguished. Time coordinates are the existing runtime's uniform audio-memory duration midpoints, not an added forced-alignment estimate. Round review is available after completion.
- Total accepted draft tokens exclude correction/bonus. Average draft acceptance is that count divided by rounds. The separately labeled paper quantity τ is **actual emitted tokens / rounds**, including correction/bonus only when emitted.
- The shared `runtime.execute` performs warm-up for each path, then a complete callback-free target-only reference. It independently records Target-only per-token events and AnchorDraft/Ours per-round events on the same GPU. Token text rendering and per-prefix checks happen after each recorded run. Separate **callback-free** runs provide latency and speedup; Target-only is the 1.000× baseline.
- Decode timing starts after audio representation preparation and includes target prompt prefill, generation, draft/predictor forwards, L21 extraction, Gaussian bias, batched verification, cache operations, and correction/bonus target forwards. CUDA synchronizes only at existing measurement boundaries. Speedup = target-only decode time / method decode time. Weights loading and warm-up are excluded. This is a single-audio demonstration, not a replacement for the paper's multi-utterance benchmark.
- After recording, all three panels start at the same zero time. `replay_timeline` merges the original `observed_decode_s` timestamps; it never aligns token/round numbers or rescales an arm to its callback-free latency. A common playback rate controls all panels (default 0.1×). A finished panel stops while the others continue. This is synchronized replay of independent runs, not concurrent GPU benchmarking. Every observed committed prefix and all final observed/unobserved token sequences must equal target-only AR. A mismatch fails visibly and suppresses comparison speedups; there is no fallback transcript or extra canonical guard inside the decoder.

## Academic project page and video

The public [project page](https://yuanyuanjia71-spec.github.io/ProgDraft/) provides a playable MP4 and an interactive shared-clock replay without requiring a GPU. The archived trace has real round timestamps for AnchorDraft/Ours and the callback-free total latency plus final token IDs for Target-only. Its Target-only card therefore shows measured latency and final text without inventing a per-token animation. The current backend records Target-only per-token timestamps, and a newly exported trace lets the website player animate them. The single displayed example is the first LibriSpeech test-clean item in the fixed manifest, not selected by outcome. Its trace stores checkpoint hashes, GPU, sample identity, exact token IDs, original speculative round timestamps and separate observed/callback-free measurements. It contains no waveform file or model weights.

To recreate the trace and serve the page locally:

```bash
python demo/export_replay.py --preset 0 --output site/assets/demo-trace.json
python -m http.server 8765 --directory site
```

Then open `http://127.0.0.1:8765`. The export reuses `DemoBackend.capture` and the existing runtime. To record the same player as an MP4 (both sides uniformly slowed to 0.05×), install Playwright/Chromium and ffmpeg in a separate recording environment, then run:

```bash
pip install playwright
python -m playwright install chromium
python demo/record_project_video.py
```

The recording has a visible playback rate and shared decode clock. It does not use video duration as a benchmark. GitHub Pages publishes only `site/`; Gradio inference stays local.

## Code path

`demo/app.py` → `DemoBackend` → `progress_asr.runtime.execute/decode` → `CachedTargetRunner.verify_block`. The verifier, rollback and correction logic are unchanged. `strict_rollout` is the existing research unrestricted rollout extracted for AnchorDraft. Ours uses the existing `progress_rollout`; its optional trace only copies positions after a round. No training code or weights are altered.

`demo/render.py` renders escaped token text and native SVG; no attention tensors or FA labels are fabricated. `demo/smoke.py` can exercise all configured presets on a real GPU without a browser. Run CPU tests with `python -m unittest discover -s tests -v` and site replay checks with `node tests/check_site_replay.cjs`.
