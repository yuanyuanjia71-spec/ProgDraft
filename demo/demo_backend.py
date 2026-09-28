"""Stream observations from the shared runtime; never implement a verifier here."""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from queue import Queue, Empty
from threading import Event, Lock, Thread

import numpy as np
import torch

from progress_asr.drafter import Drafter, DraftConfig
from progress_asr.models import load_target, load_weights, make_models
from progress_asr.runtime import execute, read_audio

ROOT = Path(__file__).resolve().parents[1]
K = 8


class DemoCancelled(Exception):
    pass


def resolve_config(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{path.name} missing. See demo/README.md for local asset setup.')
    config = json.loads(path.read_text())
    for key in ('model_config', 'ours_weights', 'anchor_weights', 'target_path'):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    for preset in config.get('presets', []):
        preset['path'] = str((path.parent / preset['path']).resolve())
    return config


def validate_wave(path, max_seconds=60):
    wave = read_audio(path)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('音频必须非空，且不含 NaN/Inf。')
    if len(wave) / 16000 > max_seconds:
        raise ValueError(f'本地 Demo 单条音频上限为 {max_seconds:g} 秒。')
    return wave


def envelope(wave, bins=400):
    return [float(np.max(np.abs(x))) for x in np.array_split(wave, min(bins, len(wave)))]


def assert_exact(reference, actual, method):
    if reference != actual:
        index = next((i for i, (a, b) in enumerate(zip(reference, actual)) if a != b),
                     min(len(reference), len(actual)))
        raise RuntimeError(f'{method}: target-only token ID 一致性失败，首次差异位置 {index}。'
                           '本次不报告 speedup，不用 AR 文本覆盖模型输出。')


class DemoBackend:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cuda:0')
        self.lock = Lock()
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        if not torch.cuda.is_available():
            raise RuntimeError('本 Demo 复用 CUDA 论文推理路径，需要 NVIDIA GPU。')
        for key in ('model_config', 'ours_weights', 'anchor_weights'):
            if not Path(self.config[key]).is_file():
                raise FileNotFoundError(f'{key} 文件不存在；请检查本地配置。')
            expected = self.config.get(key + '_sha256')
            if expected:
                with Path(self.config[key]).open('rb') as f:
                    if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
                        raise ValueError(f'{key} SHA-256 不匹配；文件与本地配置不同。')
        torch.set_num_threads(1)
        spec = json.loads(Path(self.config['model_config']).read_text())
        self.target, self.processor = load_target(spec, self.device, target_path=self.config.get('target_path'))
        self.ours, self.predictor = make_models(spec, self.device)
        self.anchor = Drafter(DraftConfig(**spec['drafter'])).to(self.device)
        load_weights(self.ours, self.predictor, self.config['ours_weights'])
        load_weights(self.anchor, None, self.config['anchor_weights'])
        for model in (self.target, self.ours, self.predictor, self.anchor):
            model.requires_grad_(False).eval()
        self.loaded = True

    def run(self, path, method, observer=None):
        draft = self.anchor if method == 'anchor' else self.ours if method == 'ours' else None
        predictor = self.predictor if method == 'ours' else None
        return execute(self.target, self.processor, path, self.device,
                       draft=draft, predictor=predictor, k=K, on_round=observer)

    def capture(self, path, notify=lambda **event: None):
        """Independent GPU runs; retain original round timestamps for paired replay."""
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('GPU 正在处理上一条音频，请等待其结束。')
        try:
            wave = validate_wave(path, self.config.get('max_audio_seconds', 60))
            audio = dict(duration=len(wave)/16000, envelope=envelope(wave))
            notify(kind='audio', **audio)
            notify(kind='phase', text='加载冻结模型与现有权重…')
            self.load()
            for method in ('ar', 'anchor', 'ours'):
                notify(kind='phase', text=f'预热 {method}（不计入展示统计）…')
                self.run(path, method)
            notify(kind='phase', text='独立测量 Target-only AR，保存完整 token ID 参考…')
            reference = self.run(path, 'ar')
            methods = {}
            for method in ('anchor', 'ours'):
                notify(kind='phase', text=f'记录 {method} 的真实轮次时间戳；两侧将在测量完成后同步起跑…')
                rounds = []
                # Minimal observer: runtime already supplies independent CPU snapshots.
                # Token decoding / validation happen AFTER the timed observed run.
                observed = self.run(path, method, rounds.append)
                assert_exact(reference['tokens'], observed['tokens'], method)
                tokenizer = self.processor.tokenizer
                for event in rounds:
                    assert_exact(reference['tokens'][:len(event['tokens'])], event['tokens'], method)
                    event['pieces'] = [tokenizer.decode([t], skip_special_tokens=False)
                                       for t in event['candidates']]
                    event['emitted_pieces'] = [tokenizer.decode([t], skip_special_tokens=False)
                                               for t in event['emitted']]
                    event['text'] = tokenizer.decode(event['tokens'], skip_special_tokens=True)
                notify(kind='phase', text=f'{method}：关闭观测，独立测量 decode 延迟…')
                measured = self.run(path, method)
                assert_exact(reference['tokens'], measured['tokens'], method)
                for key in ('rounds', 'accepted_sum', 'accept_distribution'):
                    if measured[key] != observed[key]:
                        raise RuntimeError(f'{method}: 观测/无观测运行的 {key} 不一致。')
                methods[method] = dict(rounds=rounds, observed=observed, measured=measured,
                                       speedup=reference['decode_s']/measured['decode_s'], exact=True)
            return dict(schema_version=1, k=K, audio=audio, reference=reference, methods=methods,
                        gpu=torch.cuda.get_device_name(self.device),
                        timing='Original observed decode timestamps, including prefill; no per-method time rescaling. '
                               'Separate callback-free runs provide benchmark timings.')
        finally:
            self.lock.release()

    def events(self, path, playback_rate=0.1):
        """Capture independently, then start BOTH traces on one wall-clock origin.

        A single playback multiplier applies to both methods. Round timings are
        never stretched to match callback-free latency or to align round numbers.
        """
        if not 0 < playback_rate <= 1:
            raise ValueError('Playback rate must be in (0, 1].')
        queue, cancel = Queue(), Event()

        def send(**event):
            if cancel.is_set():
                raise DemoCancelled()
            queue.put(event)

        def worker():
            try:
                comparison = self.capture(path, send)
                send(kind='reference', result=comparison['reference'])
                send(kind='replay_start', playback_rate=playback_rate)
                start = time.perf_counter()
                for seconds, event in replay_timeline(comparison):
                    delay = seconds/playback_rate - (time.perf_counter()-start)
                    if delay > 0 and cancel.wait(delay):
                        raise DemoCancelled()
                    send(**event, replay_time=seconds)
                send(kind='done')
            except DemoCancelled:
                pass
            except Exception as exc:
                queue.put(dict(kind='error', text=str(exc)))
            finally:
                queue.put(None)

        Thread(target=worker, daemon=True, name='progdraft-demo-gpu').start()
        try:
            while True:
                try:
                    item = queue.get(timeout=0.25)
                except Empty:
                    continue
                if item is None:
                    break
                yield item
        finally:
            cancel.set()


def replay_timeline(comparison):
    """Merge original timestamps, preserving unequal duration and round count."""
    timeline = []
    for method in ('anchor', 'ours'):
        state = comparison['methods'][method]
        finish = state['observed']['decode_s']
        previous = 0.0
        for event in state['rounds']:
            seconds = event['observed_decode_s']
            if not previous <= seconds <= finish:
                raise ValueError('Non-monotonic / out-of-range observed round timestamp')
            previous = seconds
            timeline.append((seconds, dict(kind='round', method=method, event=event)))
        timeline.append((finish, dict(kind='observed_done', method=method, result=state['observed'])))
        timeline.append((finish, dict(kind='measured', method=method, result=state['measured'], speedup=state['speedup'])))
    return sorted(timeline, key=lambda item: item[0])


def initial_view():
    return dict(status='上传 WAV 或选择测试集样例，然后开始对比。', duration=0, envelope=[],
                reference=None, error=False, done=False, replay_time=0.0, playback_rate=None,
                anchor=dict(rounds=[], measured=None, exact=False),
                ours=dict(rounds=[], measured=None, exact=False))


def update_view(view, event):
    kind = event['kind']
    if 'replay_time' in event:
        view['replay_time'] = event['replay_time']
    if kind == 'audio':
        view.update(duration=event['duration'], envelope=event['envelope'])
    elif kind == 'phase':
        view['status'] = event['text']
    elif kind == 'reference':
        view['reference'] = event['result']
    elif kind == 'replay_start':
        view.update(playback_rate=event['playback_rate'], replay_time=0.0,
                    status='双侧同步回放：同一起点、原始时间戳、统一播放倍率。')
    elif kind == 'round':
        view[event['method']]['rounds'].append(event['event'])
    elif kind == 'observed_done':
        view[event['method']].update(exact=True, observed=event['result'])
    elif kind == 'measured':
        view[event['method']].update(measured=event['result'], speedup=event['speedup'])
    elif kind == 'done':
        view.update(done=True, status='完成：两种方法的全部输出 token ID 均与 Target-only greedy AR 一致。')
    elif kind == 'error':
        view.update(error=True, status=event['text'])
        # An incomplete/failed comparison must not advertise a speedup.
        for method in ('anchor', 'ours'):
            view[method]['measured'] = None
            view[method]['exact'] = False
    return view
