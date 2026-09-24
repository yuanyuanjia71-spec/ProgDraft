"""Portable cached-feature training of the released Random-K objective.

All token inputs use the frozen target's greedy trajectory. Only acoustic
positions and drafter hidden/KV states recur through the model's predictions.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .io import read_json, write_json, save_checkpoint, sha256
from .data import load_records, collate, epoch_plan, progress_denominators
from .models import load_target, make_models, load_weights
from .objective import random_k_rollout_loss


@torch.inference_mode()
def evaluate(draft, predictor, target, records, device):
    draft.eval(); predictor.eval()
    sums = [dict(count=0, correct=0, top5=0, CE_sum=0., position_abs_sum=0., progress_valid_count=0) for _ in range(8)]
    for record in records:
        items = [(record, a) for a in range(len(record['current_tokens']))]
        for left in range(0, len(items), 24):
            part = items[left:left+24]
            batch = collate(part, device)
            horizons = torch.tensor([min(8, len(c['token_ids'])-a) for c, a in part], device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, observations, graph = random_k_rollout_loss(draft, predictor, target.thinker.get_input_embeddings(),
                    target.thinker.lm_head, batch, horizons, len(part), progress_denominators(part), probabilities=[1.]*8)
                for entry, node in zip(observations, graph):
                    row = sums[entry['depth']-1]
                    for key in row:
                        if key != 'top5': row[key] += entry[key]
                    logits = target.thinker.lm_head(node['state'][:, 0]).float()
                    labels = batch['gold_extended'][node['active'], entry['depth']-1]
                    row['top5'] += int(logits.topk(5, -1).indices.eq(labels[:, None]).any(-1).sum())
    draft.train(); predictor.train()
    return [dict(depth=d+1, N=r['count'], correct=r['correct'], top1=r['correct']/max(r['count'],1),
                 top5=r['top5']/max(r['count'],1), CE=r['CE_sum']/max(r['count'],1),
                 position_MAE=r['position_abs_sum']/max(r['progress_valid_count'],1)) for d, r in enumerate(sums)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--initial-weights', required=True, help='original step-0 tensor-only export')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--target-path')
    parser.add_argument('--resume', help='trusted checkpoint created by this trainer')
    parser.add_argument('--stop-after', type=int, help='debug stopping point; never changes scheduler horizon')
    args = parser.parse_args()
    config = read_json(args.config); recipe = config['training']
    if (recipe['K_min'], recipe['K_max']) != (3, 8):
        raise ValueError('this release reproduces Random-K[3,8] only')
    target, processor = load_target(config, args.device, training=True, target_path=args.target_path)
    eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    records = load_records(args.manifest, 'train', config['drafter']['hidden_size'], eos)
    validation = load_records(args.manifest, 'validation', config['drafter']['hidden_size'], eos)
    assert len(records) == recipe['expected_train_utterances'] and len(validation) == 250
    # The exported initialization carries the exact CPU/CUDA RNG states after
    # predictor construction used by the original experiment.
    draft, predictor = make_models(config, args.device)
    initial = load_weights(draft, predictor, args.initial_weights)
    if 'rng_torch' not in initial or 'rng_cuda' not in initial:
        raise ValueError('exact step-0 export with rng_torch/rng_cuda is required')
    torch.set_rng_state(initial['rng_torch'])
    torch.cuda.set_rng_state(initial['rng_cuda'], args.device)
    shuffle = random.Random(recipe['seed'])
    krng = np.random.default_rng(recipe['seed'])
    parameters = list(draft.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe['cosine_T_max'])
    draft.train(); predictor.train()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    identity = dict(config=config, manifest_sha256=sha256(args.manifest), initial_sha256=sha256(args.initial_weights))
    step, start_epoch, offset, plan = 0, 1, 0, None
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert state['identity'] == identity
        draft.load_state_dict(state['drafter']); predictor.load_state_dict(state['progress_predictor'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng_torch']); torch.cuda.set_rng_state(state['rng_cuda'], args.device)
        shuffle.setstate(state['rng_shuffle']); krng.bit_generator.state = state['rng_K']
        step, start_epoch, offset, plan = state['step'], state['epoch'], state['offset'], state['plan']
        if (out/'loss_curve.csv').exists():
            with (out/'loss_curve.csv').open() as f: rows = list(csv.DictReader(f))
            with (out/'loss_curve.csv').open('w') as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
                    w.writerows([r for r in rows if int(r['step']) <= step])
    elif (out/'training_identity.json').exists():
        raise FileExistsError('use --resume or a new output directory')
    write_json(out/'training_identity.json', identity)
    def save(epoch, next_offset, current_plan):
        save_checkpoint(out/'checkpoints'/f'step_{step:05d}.pt', dict(
            step=step, epoch=epoch, offset=next_offset, plan=current_plan, identity=identity,
            drafter=draft.state_dict(), progress_predictor=predictor.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(args.device),
            rng_shuffle=shuffle.getstate(), rng_K=krng.bit_generator.state))
        write_json(out/f'validation_step_{step:05d}.json', evaluate(draft, predictor, target, validation, args.device))
    if not args.resume: save(1, 0, None)
    limit = args.stop_after or recipe['steps']
    if not 0 < limit <= recipe['steps']: raise ValueError('invalid stopping point')
    batch_size = recipe['utterance_batch_size']; micro = recipe['anchor_microbatch']
    started = time.monotonic()
    for epoch in range(start_epoch, recipe['epochs']+1):
        if step >= limit: break
        if plan is None:
            order = list(range(len(records))); shuffle.shuffle(order)
            if recipe['tail_policy'] == 'repeat_first_to_full_batch': order += order[:(-len(order)) % batch_size]
            sampled, actual, probabilities = epoch_plan(records, order, krng)
            plan = dict(order=order, sampled=sampled, actual=actual, probabilities=probabilities)
        order, actual, probabilities = plan['order'], plan['actual'], plan['probabilities']
        for left in range(offset, len(order), batch_size):
            if step >= limit: break
            selected = order[left:left+batch_size]
            items = [(records[i], a) for i in selected for a in range(len(records[i]['current_tokens']))]
            horizons = np.concatenate(actual[left:left+batch_size])
            denominators = progress_denominators(items)
            optimizer.zero_grad(set_to_none=True)
            total = 0.; observations = []
            for start in range(0, len(items), micro):
                batch = collate(items[start:start+micro], args.device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, obs, graph = random_k_rollout_loss(draft, predictor, target.thinker.get_input_embeddings(),
                        target.thinker.lm_head, batch, torch.tensor(horizons[start:start+micro], device=args.device),
                        len(items), denominators, probabilities=probabilities)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                loss.backward(); total += float(loss.detach()); observations.extend(obs)
                del graph, loss, batch
            norm = torch.nn.utils.clip_grad_norm_(parameters, recipe['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1
            row = dict(step=step, epoch=epoch, loss=total, gradient_norm=float(norm),
                       actual_K_mean=float(horizons.mean()), wall_seconds=time.monotonic()-started)
            for depth in range(1,9):
                values = [r for r in observations if r['depth'] == depth]
                row[f'd{depth}_CE'] = sum(r['CE_sum'] for r in values)/max(sum(r['count'] for r in values),1)
            exists = (out/'loss_curve.csv').exists()
            with (out/'loss_curve.csv').open('a') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not exists: writer.writeheader()
                writer.writerow(row)
            if step % 25 == 0: print(json.dumps(row), flush=True)
            if step % 1000 == 0 or step == limit or (config['scale'] == '1.7b' and step % 328 == 0):
                save(epoch, left+len(selected), plan)
        offset, plan = 0, None
    assert all(not p.requires_grad and p.grad is None for p in target.parameters())
    write_json(out/'status.json', dict(step=step, complete=step == recipe['steps'], target_frozen=True))


if __name__ == '__main__':
    main()
