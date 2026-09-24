"""Build compatible caches on licensed audio supplied by the researcher.

Historical paper reproduction should use the exported immutable assets. This
batch-one builder is also useful for new data; it does not claim bitwise
equivalence to the historical 0.6B batched feature extraction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .models import load_target
from .runtime import WaveformRunner, read_audio, decode
from .alignment import align_target_text, positions_from_alignment
from .data import validate_record
from .io import read_json, write_json, save_checkpoint, sha256


@torch.inference_mode()
def extract_features(target, processor, waveform, tokens, device):
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import apply_rotary_pos_emb, repeat_kv
    runner = WaveformRunner(target, processor, waveform, device, capture=False)
    ids = runner.prompt_ids; length = ids.shape[1]
    gold = torch.tensor(tokens, device=device)
    n = len(tokens)-2
    all_ids = torch.cat([ids, gold[None]], 1)
    embeds = torch.cat([runner.prepared['inputs_embeds'], target.thinker.get_input_embeddings()(gold)[None]], 1)
    mask = torch.ones_like(all_ids)
    pos, _ = target.thinker.get_rope_index(mask)
    anchors = torch.arange(length-1, length-1+n, device=device)
    captured, coord, handles = {}, {}, []
    for layer in (0,9,18,27):
        def hook(module, args, output, layer=layer):
            captured[layer] = (output[0] if isinstance(output,tuple) else output)[0,anchors].cpu()
        handles.append(target.thinker.model.layers[layer].register_forward_hook(hook))
    def attention(module, args, kwargs):
        x = kwargs['hidden_states']; shape = (*x.shape[:-1], -1, module.head_dim)
        q = module.q_norm(module.q_proj(x).view(shape)).transpose(1,2)
        k = module.k_norm(module.k_proj(x).view(shape)).transpose(1,2)
        q,k = apply_rotary_pos_emb(q,k,*kwargs['position_embeddings'])
        k = repeat_kv(k,module.num_key_value_groups)
        scores = (q[:,:,anchors].float() @ k.float().transpose(-2,-1)) * module.scaling
        scores.masked_fill_(torch.arange(k.shape[-2],device=device)[None,:] > anchors[:,None],float('-inf'))
        weights = scores.softmax(-1)[0,:,:,runner.audio_positions].mean(0)
        coord['a1'] = runner.audio_times[weights.argmax(-1)].cpu()
    handles.append(target.thinker.model.layers[21].self_attn.register_forward_pre_hook(attention,with_kwargs=True))
    try:
        output = target.thinker.model(inputs_embeds=embeds,attention_mask=mask,position_ids=pos,use_cache=False)
    finally:
        for h in handles: h.remove()
    return dict(audio_memory=runner.audio_memory[0].cpu(),
                features=torch.stack([captured[layer] for layer in (0,9,18,27)],1),
                target_feature=output.last_hidden_state[0,anchors].cpu(),
                current_tokens=all_ids[0,anchors].cpu(),a1_verification_s=coord['a1'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',required=True,choices=['trajectories','features','alignment'])
    p.add_argument('--config',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--target-path');p.add_argument('--device',default='cuda:0')
    a=p.parse_args();config=read_json(a.config);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest=Path(a.manifest);samples=[json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    assert len({s['id'] for s in samples})==len(samples)
    if a.phase=='alignment':
        from huggingface_hub import snapshot_download
        from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
        target_spec=config['target']
        path=a.target_path or snapshot_download(target_spec['model_id'],revision=target_spec['revision'])
        processor=Qwen3ASRProcessor.from_pretrained(path)
    else:
        target,processor=load_target(config,a.device,training=a.phase=='features',target_path=a.target_path)
    eos=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    rows=[]
    for sample in samples:
        stem=hashlib.sha256(sample['id'].encode()).hexdigest()
        wave=read_audio(manifest.parent/sample['audio'])
        trajectory=out/'trajectories'/f'{stem}.json'
        record_path=out/'records'/f'{stem}.pt'
        if a.phase=='trajectories':
            with torch.inference_mode():
                runner=WaveformRunner(target,processor,wave,a.device,capture=False)
                tokens,_=decode(runner)
            assert len(tokens)>=3 and tokens[-1]==eos and eos not in tokens[:-1]
            write_json(trajectory,dict(id=sample['id'],token_ids=tokens,waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                                      target=config['target'],builder='batch_one_fp32_eager_target_greedy_v1'))
        else:
            t=read_json(trajectory)
            assert t['target']==config['target'] and t['waveform_sha256']==hashlib.sha256(wave.tobytes()).hexdigest()
            if a.phase=='features':
                c=extract_features(target,processor,wave,t['token_ids'],a.device)
                c.update(id=sample['id'],token_ids=t['token_ids'],duration_s=len(wave)/16000,
                         contract='same_position_target_greedy_l21_span_midpoint_v1')
                save_checkpoint(record_path,c)
            else:
                c=torch.load(record_path,map_location='cpu',weights_only=True)
                alignment=align_target_text(wave,t['token_ids'],processor.tokenizer,a.device)
                c['fa_positions'],c['fa_valid']=positions_from_alignment(processor.tokenizer,t['token_ids'],alignment)
                validate_record(c,config['drafter']['hidden_size'],eos)
                save_checkpoint(record_path,c)
                write_json(out/'alignments'/f'{stem}.json',alignment)
                rows.append(dict(id=sample['id'],split=sample['split'],cache=f'records/{stem}.pt',sha256=sha256(record_path)))
    if a.phase=='alignment':
        (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(out/f'{a.phase}_complete.json',dict(count=len(samples),input_manifest_sha256=sha256(manifest),config=config,
               historical_bitwise_reproduction=False,builder='standalone_batch_one_v1'))


if __name__=='__main__':main()
