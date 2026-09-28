"""Observation, EOS semantics, progress traces and exact-match failures."""
from contextlib import nullcontext
from types import SimpleNamespace
import importlib.util
import importlib.metadata
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from progress_asr.drafter import Drafter, DraftConfig
from progress_asr.progress import AcousticProgressPredictor
from progress_asr.rollout import progress_rollout, strict_rollout
from demo.render import tokens_html


class TraceTests(unittest.TestCase):
    def test_observer_does_not_change_rollout(self):
        torch.manual_seed(7)
        h = 16
        draft = Drafter(DraftConfig(hidden_size=h, attention_heads=4, ffn_hidden=32)).eval()
        predictor = AcousticProgressPredictor(h).eval()
        embedding, head = torch.nn.Embedding(20, h), torch.nn.Linear(h, 20)
        target = SimpleNamespace(thinker=SimpleNamespace(get_input_embeddings=lambda: embedding, lm_head=head))
        runner = SimpleNamespace(device='cpu', cache_record={'sample': {'duration_s': 2}},
                                 audio_memory=torch.randn(1, 12, h), audio_mask=torch.ones(1, 12, dtype=torch.bool),
                                 audio_times=torch.linspace(0, 2, 12))
        context = SimpleNamespace(current_token=3, features=[torch.randn(1, 1, h) for _ in range(4)],
                                  verification_anchor_time_s=.3, q_from_current_verification=True)
        with torch.inference_mode():
            plain = progress_rollout(draft, predictor, target, runner, context, 8)
            trace = {}
            observed = progress_rollout(draft, predictor, target, runner, context, 8, trace=trace)
            self.assertTrue(torch.equal(plain, observed))
            self.assertEqual(len(trace['positions']), 8)
            positions = torch.cat(trace['positions'])
            self.assertTrue(bool((positions[1:] >= positions[:-1]).all()))
            self.assertTrue(bool((torch.cat(trace['centers']) <= 2).all()))
            self.assertEqual(strict_rollout(draft, target, runner, context, 8).shape, (8,))

    def test_html_escapes_tokens_and_uses_verified_prefix(self):
        event = dict(candidates=[1, 2, 3], pieces=['<script>', '&', 'later'], accepted=1, first_rejected=1)
        html = tokens_html(event)
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertEqual(html.count('token accepted'), 1)
        self.assertEqual(html.count('token rejected'), 1)
        self.assertEqual(html.count('token unused'), 1)
        event.update(accepted=2, first_rejected=None)
        self.assertNotIn('token rejected', tokens_html(event))


class DecodeObservationTests(unittest.TestCase):
    def test_rejection_bonus_and_accepted_eos(self):
        from progress_asr.runtime import decode
        # Actual decode loop; only model/verification computation is simulated.
        class Runner:
            target = object()
            forward_counts = {}
            processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 99))

            def prefill(self, **kwargs):
                self.index = 0
                return object()

            def verify_block(self, context, candidates):
                rows = [(1, [1, 2], object()), (4, [3, 4, 5, 6, 7], object()), (2, [8, 99], None)]
                accepted, emitted, ctx = rows[self.index]
                self.index += 1
                return accepted, [], emitted, ctx

        def rollout(draft, target, runner, context, k):
            return torch.tensor([[1, 9, 9, 9], [3, 4, 5, 6], [8, 99, 1, 1]][runner.index])

        events = []
        with patch('progress_asr.runtime.strict_rollout', rollout), patch('torch.autocast', lambda *a, **kw: nullcontext()):
            observed = decode(Runner(), draft=object(), k=4, on_round=events.append)
            plain = decode(Runner(), draft=object(), k=4)
        self.assertEqual(observed, plain)
        self.assertEqual([e['first_rejected'] for e in events], [1, None, None])
        self.assertEqual([len(e['emitted']) for e in events], [2, 5, 2])
        self.assertEqual(observed[1]['accepted_sum'], 7)
        self.assertEqual(observed[1]['mean_accepted'], 3)
        events[-1]['tokens'].append(-10)
        self.assertNotIn(-10, observed[0])

    def test_bad_reference_fails_closed(self):
        from demo.demo_backend import assert_exact, initial_view, update_view
        with self.assertRaisesRegex(RuntimeError, '位置 1'):
            assert_exact([1, 2], [1, 3], 'test')
        view = initial_view()
        view['anchor']['measured'] = {'decode_s': 1}
        update_view(view, dict(kind='error', text='failed'))
        self.assertIsNone(view['anchor']['measured'])
        self.assertTrue(view['error'])


@unittest.skipUnless(importlib.util.find_spec('gradio') and importlib.metadata.version('gradio') == '5.50.0',
                     'install the pinned demo extra for UI checks')
class DemoUITests(unittest.TestCase):
    def test_empty_input_and_initial_slider_updates(self):
        from demo.app import create_app
        root = Path(__file__).resolve().parents[1]
        app = create_app({'model_config': str(root/'configs/qwen3_asr_0.6b.json')})
        compare = next(f.fn for f in app.fns.values() if f.fn.__name__ == 'compare')
        outputs = list(compare(None))
        self.assertEqual(len(outputs), 1)
        self.assertIn('请先上传', outputs[0][0])
        self.assertGreater(outputs[0][6]['maximum'], outputs[0][6]['minimum'])
        self.assertTrue(outputs[0][7]['interactive'])


if __name__ == '__main__':
    unittest.main()
