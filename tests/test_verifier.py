"""Exercise actual verifier branch/crop logic without loading target weights."""
import unittest
from types import SimpleNamespace
import torch

try:
    from progress_asr.target_cache import CachedTargetRunner
except ImportError:
    CachedTargetRunner = None


@unittest.skipIf(CachedTargetRunner is None, 'install ASR extras for target verifier tests')
class VerifierTests(unittest.TestCase):
    def test_all_rejection_depths_bonus_and_eos(self):
        for accepted in range(4):
            self.run_round(accepted, 7, [1,2,3], [1,2,3,4], expected_out=[1,2,3][:accepted]+[1,2,3,4][accepted:accepted+1])
        self.run_round(2,7,[1,7,3],[1,7,3,4],expected_out=[1,7],terminal=True)
        self.run_round(0,7,[3,2,1],[7,2,1,4],expected_out=[7],terminal=True)

    def run_round(self,accepted,eos,candidates,targets,expected_out,terminal=False):
        # Force the specified first rejection except when EOS is accepted.
        if accepted<len(candidates) and not terminal: candidates[accepted]=6
        cache=SimpleNamespace(length=10)
        def crop(n): cache.length=n
        cache.crop=crop
        runner=SimpleNamespace(forward_counts={'candidate_verification':0},
            target=SimpleNamespace(thinker=SimpleNamespace(get_input_embeddings=lambda:lambda x:x)),
            processor=SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _:eos)))
        def logits(ids): return torch.nn.functional.one_hot(torch.tensor(ids),8).float()*10
        def forward(**kwargs):
            cache.length+=len(candidates)
            return SimpleNamespace(last_hidden_state=torch.tensor(targets[1:])[None])
        runner._forward=forward; runner.target_logits=lambda x:logits(x.tolist())
        def append(ctx,token):
            self.assertEqual(cache.length,10+accepted)
            cache.length+=1
            return SimpleNamespace(current_token=token)
        runner.append_target_token=append
        ctx=SimpleNamespace(cache=cache,cache_length=10,next_logits=logits([targets[0]]))
        n,_,out,next_ctx=CachedTargetRunner.verify_block(runner,ctx,torch.tensor(candidates))
        self.assertEqual(out,expected_out)
        self.assertEqual(n,accepted if not terminal else (2 if expected_out==[1,7] else 0))
        self.assertEqual(next_ctx is None,terminal)


if __name__=='__main__': unittest.main()
