"""Run from the repository root: python demo/app.py."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from demo.demo_backend import DemoBackend, initial_view, resolve_config, update_view
from demo.render import method_panel, metrics_html, status_html


def create_app(config):
    import gradio as gr
    backend = DemoBackend(config)
    presets = {x['label']: x['path'] for x in config.get('presets', [])}
    css = (ROOT / 'demo/style.css').read_text()
    view = initial_view()
    model_id = json.loads(Path(config['model_config']).read_text())['target']['model_id']
    with gr.Blocks(title='ProgDraft · Long-horizon ASR', css=css, theme=gr.themes.Soft(primary_hue='teal')) as app:
        gr.Markdown('# ProgDraft\n### Hear the speech. Follow the draft.')
        gr.Markdown('长跨度投机解码 · AnchorDraft 与 Acoustic Progress Propagation 的逐轮对比')
        gr.Markdown(f'**同一 Target：{model_id}　·　Greedy　·　K = 8　·　同一 GPU 顺序运行**')
        gr.Markdown(f'AnchorDraft 权重：{config.get("anchor_label", "local checkpoint")}  \n'
                    f'Ours 权重：{config.get("ours_label", "Joint + Random-K[3,8]")}')
        with gr.Row():
            with gr.Column(scale=2):
                audio = gr.Audio(label='上传 WAV / 播放测试样例', sources=['upload'], type='filepath', format='wav')
            with gr.Column(scale=1):
                preset = gr.Dropdown(label='固定 Final Test 预设', choices=list(presets), value=None,
                                     info='也可直接上传自己的音频。')
                start = gr.Button('开始真实解码对比', variant='primary')
                gr.Markdown('先预热并运行 Target-only AR，然后顺序展示两个方法。模型首次加载需要一些时间。')
        status = gr.HTML(status_html(view))
        gr.HTML('<div class="legend"><span class="accepted">绿色 · 已接受</span>'
                '<span class="rejected">红色 · 首次拒绝</span><span class="unused">灰色 · 未接受/未验证</span></div>')
        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown('## AnchorDraft')
                left = gr.HTML(method_panel(view, 'anchor'))
            with gr.Column():
                gr.Markdown('## Ours · Acoustic Progress Propagation')
                right = gr.HTML(method_panel(view, 'ours'))
                review = gr.Slider(1, 2, value=1, step=1, label='完成后回看 Ours 某一轮的声学进度', interactive=False)
        gr.Markdown('### 实时接受统计与独立推理计时')
        metrics = gr.HTML(metrics_html(view))
        reference = gr.Textbox(label='Target-only greedy AR transcript（参考）', interactive=False)
        state = gr.State(view)

        def select_preset(label):
            return presets.get(label)

        def render(current):
            n = len(current['ours']['rounds'])
            return (status_html(current), method_panel(current, 'anchor'), method_panel(current, 'ours'),
                    metrics_html(current), (current['reference'] or {}).get('text', ''), current,
                    gr.update(minimum=1, maximum=max(2, n), step=1, value=max(1, n), interactive=current['done'] and n > 1),
                    gr.update(interactive=current['done'] or current['error']),
                    gr.update(interactive=current['done'] or current['error']),
                    gr.update(interactive=current['done'] or current['error']))

        def compare(path):
            current = initial_view()
            if not path:
                current.update(error=True, status='请先上传 WAV 或选择预设。')
                yield render(current)
                return
            yield render(current)
            try:
                for event in backend.events(path):
                    update_view(current, event)
                    yield render(current)
            except Exception as exc:
                update_view(current, dict(kind='error', text=str(exc)))
                yield render(current)

        preset.change(select_preset, preset, audio, queue=False, api_name=False)
        start.click(compare, audio, [status, left, right, metrics, reference, state, review, start, audio, preset],
                    concurrency_limit=1, concurrency_id='single-gpu', api_name='compare')
        review.input(lambda n, current: method_panel(current, 'ours', n), [review, state], right,
                     queue=False, api_name=False)
    return app.queue(default_concurrency_limit=1, max_size=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'demo/local_config.json'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    config = resolve_config(args.config)
    app = create_app(config)
    # No public tunnel, file browser, or broad allowed_paths.
    app.launch(server_name=args.host, server_port=args.port, share=False, show_error=True)


if __name__ == '__main__':
    main()
