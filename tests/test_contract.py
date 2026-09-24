"""CPU tests for token/state indexing, loss reduction and acoustic gradients."""
import unittest
import torch
from progress_asr.drafter import Drafter, DraftConfig
from progress_asr.progress import AcousticProgressPredictor
from progress_asr.objective import random_k_rollout_loss, WEIGHTS
from progress_asr.alignment import positions_from_alignment
from progress_asr.data import epoch_plan
import numpy as np


def fixture():
    torch.manual_seed(4)
    n, h, audio, vocab = 6, 32, 24, 31
    draft = Drafter(DraftConfig(hidden_size=h, attention_heads=4, ffn_hidden=64, feature_noise=0))
    predictor = AcousticProgressPredictor(h)
    embedding = torch.nn.Embedding(vocab, h).requires_grad_(False)
    head = torch.nn.Linear(h, vocab).requires_grad_(False)
    gold = (torch.arange(8)[None]+torch.arange(n)[:, None]+2) % vocab
    batch = dict(audio_memory=torch.randn(n,audio,h), audio_mask=torch.ones(n,audio,dtype=torch.bool),
                 current_tokens=torch.arange(n), features=[torch.randn(n,1,h) for _ in range(4)],
                 target_feature=torch.randn(n,h), gold_extended=gold,
                 audio_duration_s=torch.full((n,),20.), audio_time_positions=torch.linspace(0,20,audio).expand(n,-1),
                 a1_verification_s=torch.full((n,),.2), fa_extended=torch.arange(8).float().expand(n,-1),
                 fa_valid_extended=torch.ones(n,8,dtype=torch.bool))
    return draft,predictor,embedding,head,batch


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_teacher_tokens_kv_and_recursive_positions(self):
        draft,predictor,embedding,head,batch=fixture()
        calls=[]
        handle=draft.register_forward_pre_hook(lambda m,args,kw:calls.append((args,kw)),with_kwargs=True)
        _,obs,graph=random_k_rollout_loss(draft,predictor,embedding,head,batch,torch.arange(3,9),6,[6]*8,probabilities=[1.]*8)
        handle.remove()
        self.assertEqual([r['count'] for r in obs],[6,6,6,5,4,3,2,1])
        self.assertEqual([r['kv_length'] for r in graph],list(range(1,9)))
        for depth in range(1,8):
            active=graph[depth]['active']
            self.assertTrue(torch.equal(calls[depth][0][0],embedding(batch['gold_extended'][active,depth-1])[:,None]))
            prev=graph[depth-1]
            positions=torch.searchsorted(prev['active'],active)
            self.assertTrue(torch.equal(graph[depth]['previous'],prev['position'][positions]))
            self.assertIn('recurrent_state',calls[depth][1])
            self.assertNotIn('target_features',calls[depth][1])

    def test_token_ce_alone_reaches_predictor_and_previous_positions(self):
        draft,predictor,embedding,head,batch=fixture()
        _,_,g=random_k_rollout_loss(draft,predictor,embedding,head,batch,torch.full((6,),8),6,[6]*8)
        grad,=torch.autograd.grad(g[-1]['CE'],g[1]['position'],retain_graph=True)
        self.assertGreater(float(grad.abs().sum()),0)
        sum(row['CE'] for row in g).backward()  # no feature or progress supervision
        norm=sum(float(p.grad.square().sum()) for p in predictor.parameters())**.5
        self.assertGreater(norm,0)
        self.assertTrue(all(p.grad is None for p in embedding.parameters()))
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_exact_loss_reduction(self):
        draft,predictor,embedding,head,batch=fixture()
        batch['fa_valid_extended'][::2,2:]=False
        den=batch['fa_valid_extended'].sum(0).tolist()
        prob=[1,1,1,5/6,4/6,3/6,2/6,1/6]
        loss,obs,_=random_k_rollout_loss(draft,predictor,embedding,head,batch,torch.arange(3,9),6,den,probabilities=prob)
        expected=.5*obs[0]['feature_sum']/6
        for d,row in enumerate(obs):
            expected+=WEIGHTS[d]/prob[d]*row['CE_sum']/6
            if d: expected+=.1*WEIGHTS[d]/prob[d]*row['progress_sum']/max(den[d],1)
        self.assertAlmostEqual(float(loss.detach()),expected,places=5)

    def test_fa_span_midpoint_and_punctuation(self):
        class Tokenizer:
            def decode(self,ids,**kwargs): return ''.join({1:'ab',2:' ',3:',',4:'c',5:''}[i] for i in ids)
        align=dict(normalized_target_text='abc',character_intervals=[dict(start_s=0,end_s=.1),dict(start_s=.1,end_s=.3),dict(start_s=.4,end_s=.6)])
        a,valid=positions_from_alignment(Tokenizer(),[1,2,3,4,5],align)
        self.assertEqual(valid.tolist(),[True,False,False,True,False])
        self.assertAlmostEqual(float(a[0]),.15,places=6)
        self.assertAlmostEqual(float(a[3]),.5,places=6)

    def test_horizon_clip_and_empirical_probabilities(self):
        records=[dict(token_ids=list(range(12)),current_tokens=list(range(10)))]
        sampled,actual,prob=epoch_plan(records,[0],np.random.default_rng(0))
        self.assertTrue(((sampled[0]>=3)&(sampled[0]<=8)).all())
        self.assertEqual(actual[0][-1],3)
        self.assertEqual(prob,[float((actual[0]>=k).mean()) for k in range(1,9)])


if __name__=='__main__': unittest.main()
