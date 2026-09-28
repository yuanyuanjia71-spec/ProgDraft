"""Escaped, dependency-free HTML/SVG views of actual decoding events."""
from html import escape


def token_chip(piece, token_id, state, depth):
    label = piece if piece else '∅'
    return (f'<span class="token {state}" title="d{depth} · token ID {int(token_id)}">'
            f'<small>d{depth}</small><span>{escape(label)}</span></span>')


def tokens_html(event):
    chips = []
    for i, (token, piece) in enumerate(zip(event['candidates'], event['pieces'])):
        state = 'accepted' if i < event['accepted'] else 'rejected' if i == event['first_rejected'] else 'unused'
        chips.append(token_chip(piece, token, state, i+1))
    return '<div class="tokens">' + ''.join(chips) + '</div>'


def waveform(view, event=None):
    duration, values = view['duration'], view['envelope']
    if not values:
        return '<div class="empty">音频载入后显示波形与 a₁ → a₈。</div>'
    width, left, span, mid = 760, 40, 680, 100
    maximum = max(max(values), 1e-8)
    top, bottom = [], []
    for i, value in enumerate(values):
        x = left + span*i/max(len(values)-1, 1)
        amplitude = 34*value/maximum
        top.append(f'{x:.2f},{mid-amplitude:.2f}')
        bottom.append(f'{x:.2f},{mid+amplitude:.2f}')
    svg = [f'<svg viewBox="0 0 {width} 190" role="img" aria-label="音频波形及当前轮声学进度">',
           f'<polygon points="{" ".join(top+bottom[::-1])}" fill="#d8e5ec"/>']
    for i in range(5):
        x = left + span*i/4
        svg.append(f'<line x1="{x}" y1="142" x2="{x}" y2="148" stroke="#94a3b8"/>'
                   f'<text x="{x}" y="166" text-anchor="middle" fill="#475569" font-size="12">{duration*i/4:.2f}s</text>')
    if event and event['positions']:
        points = []
        for i, (raw, center) in enumerate(zip(event['positions'], event['centers'])):
            x = left+span*max(0, min(duration, center))/duration
            y = 24+i*5
            color = '#15803d' if i < event['accepted'] else '#dc2626' if i == event['first_rejected'] else '#64748b'
            points.append(f'{x:.2f},{y}')
            svg.append(f'<line x1="{x:.2f}" y1="{y}" x2="{x:.2f}" y2="141" stroke="{color}" opacity=".5"/>'
                       f'<circle cx="{x:.2f}" cy="{y}" r="4" fill="{color}"><title>d{i+1}: raw={raw:.4f}s; display/center={center:.4f}s</title></circle>'
                       f'<text x="{x+5:.2f}" y="{y-5}" fill="{color}" font-size="11">{i+1}</text>')
        svg.insert(2, f'<polyline points="{" ".join(points)}" fill="none" stroke="#0d9488" stroke-width="1.5"/>')
    svg.append('</svg>')
    if event and event['positions']:
        positions = ' → '.join(f'a{i+1}={a:.3f}s' for i, a in enumerate(event['positions']))
        svg.append(f'<div class="position-values">{positions}</div>')
        if any(a > duration or a < 0 for a in event['positions']):
            svg.append('<p class="muted">有预测位置超出音频范围；上方点显示实际截断后的 bias center，数值保留原始递归位置。</p>')
    return '<div class="wave">' + ''.join(svg) + '</div>'


def method_panel(view, method, review_round=None):
    state = view[method]
    events = state['rounds']
    parts = []
    if method == 'ours':
        chosen = events[-1] if events else None
        if review_round is not None and events:
            chosen = events[max(0, min(len(events)-1, int(review_round)-1))]
        if chosen:
            parts.append(f'<div class="round-heading">声学进度 · 第 {chosen["round"]} 轮</div>')
        parts.append(waveform(view, chosen))
        if review_round is not None and chosen:
            parts.append(tokens_html(chosen))
        parts.append('<p class="muted">a₁ 来自当前 Target L21；d1 无 Gaussian bias。d2–d8 使用递归预测位置，σ=0.2s。时间轴沿用论文运行路径的 audio-memory 时间网格。</p>')
    else:
        parts.append('<div class="round-heading">完整音频 · 无显式声学进度轨迹</div>')
        parts.append(waveform(view))
        parts.append('<div class="method-note">AnchorDraft-only · 每一步读取完整音频。<br>使用已训练权重，不加 Runtime Correction 窗口。</div>')
    if not events:
        parts.append('<div class="empty">等待实际解码轮次…</div>')
    else:
        latest = events[-1]
        parts.append(f'<div class="transcript"><label>已提交 transcript</label><p>{escape(latest["text"])}</p></div>')
        parts.append('<div class="round-history">')
        for e in reversed(events):
            extra = len(e['emitted'])-e['accepted']
            parts.append(f'<div class="round-card"><div class="round-heading">第 {e["round"]} 轮'
                         f'<span>接受 draft {e["accepted"]}/8 · 本轮输出 {len(e["emitted"])} token</span></div>')
            parts.append(tokens_html(e))
            if extra:
                label = 'Bonus' if e['accepted'] == 8 else 'Correction'
                parts.append(f'<p class="correction">{label}（Target 产生）：'
                             f'<code>{escape(e["emitted_pieces"][-1])}</code></p>')
            elif e['terminal']:
                parts.append('<p class="muted">已接受 EOS，本轮没有额外 bonus。</p>')
            parts.append('</div>')
        parts.append('</div>')
    if state['exact']:
        parts.append('<div class="exact">✓ 完整 token ID 序列与 Target-only greedy AR 一致</div>')
    return '<div class="method-panel">' + ''.join(parts) + '</div>'


def metrics_html(view):
    headers = ('方法', '接受 draft 总数', '轮数', '平均 draft 接受长度', 'τ（含额外 token）', 'Decode 延迟', '相对 AR')
    rows = []
    reference = view['reference']
    if reference:
        rows.append(f'<tr><td>Target-only AR</td><td>—</td><td>—</td><td>—</td><td>—</td>'
                    f'<td>{reference["decode_s"]:.3f}s</td><td>1.00×</td></tr>')
    for key, title in [('anchor', 'AnchorDraft'), ('ours', 'Ours')]:
        state = view[key]
        events, measured = state['rounds'], state['measured']
        latest = events[-1] if events else None
        rounds = len(events)
        accepted = latest['accepted_sum'] if latest else 0
        draft_mean = f'{accepted/rounds:.3f}' if rounds else '—'
        mean = f'{len(latest["tokens"])/rounds:.3f}' if rounds else '—'
        latency = (f'{measured["decode_s"]:.3f}s' if measured else
                   f'{latest["observed_decode_s"]:.3f}s（观测中）' if latest else '—')
        speed = f'{state["speedup"]:.3f}×' if measured else '待无观测计时'
        rows.append(f'<tr><td>{title}</td><td>{accepted}</td><td>{rounds}</td><td>{draft_mean}</td>'
                    f'<td>{mean}</td><td>{latency}</td><td>{speed}</td></tr>')
    return ('<div class="metrics"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) +
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
            '<p class="muted">绿色只统计 draft 接受数。τ = 实际输出 token 总数 / 轮数，包含实际产生的 correction/bonus；EOS 后不虚加。'
            '正式 Decode = prompt prefill + generation（audio memory 已就绪），含 predictor、L21、Gaussian、verification、KV 及 correction forward。'
            'Speedup 使用独立、预热后、关闭观测的完整运行；实时展示耗时不用于 speedup。单条音频结果不是整套论文 benchmark。</p>')


def status_html(view):
    css = 'status error' if view['error'] else 'status success' if view['done'] else 'status'
    return f'<div class="{css}">{escape(view["status"])}</div>'
