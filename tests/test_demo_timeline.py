"""A faster decoding trace must finish first on an unmodified shared timeline."""
import unittest
from demo.demo_backend import replay_timeline, initial_view, update_view


class SharedClockTests(unittest.TestCase):
    def test_target_only_steps_share_clock_with_speculative_rounds(self):
        ar = dict(rounds=[dict(token_index=1, observed_decode_s=.15),
                          dict(token_index=2, observed_decode_s=.5)],
                  observed=dict(decode_s=.55), measured=dict(decode_s=.53), speedup=1)
        other = dict(rounds=[dict(round=1, observed_decode_s=.3)],
                     observed=dict(decode_s=.35), measured=dict(decode_s=.34), speedup=1.5)
        timeline = replay_timeline(dict(methods=dict(ar=ar, anchor=other, ours=other)))
        self.assertEqual([(t, e['method']) for t, e in timeline if e['kind']=='round'],
                         [(.15, 'ar'), (.3, 'anchor'), (.3, 'ours'), (.5, 'ar')])
        self.assertEqual([(t, e['method']) for t, e in timeline if e['kind']=='observed_done'],
                         [(.35, 'anchor'), (.35, 'ours'), (.55, 'ar')])

    def test_merge_uses_elapsed_time_not_round_number(self):
        def method(times, finish):
            return dict(rounds=[dict(round=i+1, observed_decode_s=t) for i, t in enumerate(times)],
                        observed=dict(decode_s=finish), measured=dict(decode_s=finish), speedup=1)
        comparison = dict(methods=dict(anchor=method([.20, .70], .8), ours=method([.10, .25, .35], .4)))
        timeline = replay_timeline(comparison)
        rounds = [(t, e['method'], e['event']['round']) for t, e in timeline if e['kind']=='round']
        self.assertEqual(rounds, [(.1,'ours',1),(.2,'anchor',1),(.25,'ours',2),(.35,'ours',3),(.7,'anchor',2)])
        completions = [(t,e['method']) for t,e in timeline if e['kind']=='observed_done']
        self.assertEqual(completions, [(.4,'ours'),(.8,'anchor')])
        self.assertEqual(comparison['methods']['anchor']['rounds'][-1]['observed_decode_s'], .7)

    def test_nonmonotonic_timestamps_fail(self):
        bad = dict(rounds=[dict(observed_decode_s=.8),dict(observed_decode_s=.3)],
                   observed=dict(decode_s=1), measured={}, speedup=1)
        with self.assertRaises(ValueError):
            replay_timeline(dict(methods=dict(anchor=bad, ours=bad)))

    def test_finishing_one_side_keeps_the_other_running(self):
        view = initial_view()
        update_view(view, dict(kind='replay_start', playback_rate=.1))
        update_view(view, dict(kind='observed_done', method='ours', result=dict(decode_s=.4), replay_time=.4))
        self.assertTrue(view['ours']['exact'])
        self.assertFalse(view['anchor']['exact'])
        self.assertFalse(view['done'])
        self.assertEqual(view['replay_time'], .4)


if __name__ == '__main__':
    unittest.main()
