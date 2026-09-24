"""Pinned upstream target and tensor-only Ours checkpoint loading."""
from pathlib import Path
import torch
from safetensors.torch import load_file
from .drafter import Drafter, DraftConfig
from .progress import AcousticProgressPredictor


def load_target(config, device, *, training=False, target_path=None):
    from huggingface_hub import snapshot_download
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
    torch.backends.cuda.enable_cudnn_sdp(False)
    spec = config['target']
    path = target_path or snapshot_download(spec['model_id'], revision=spec['revision'])
    target = Qwen3ASRForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16 if training else torch.float32,
        attn_implementation='sdpa' if training else 'eager').to(device).requires_grad_(False).eval()
    assert len(target.thinker.model.layers) == 28
    assert target.thinker.config.text_config.hidden_size == config['drafter']['hidden_size']
    return target, Qwen3ASRProcessor.from_pretrained(path)


def make_models(config, device):
    draft = Drafter(DraftConfig(**config['drafter'])).to(device)
    predictor = AcousticProgressPredictor(config['drafter']['hidden_size']).to(device)
    return draft, predictor


def load_weights(draft, predictor, path, device='cpu'):
    tensors = load_file(str(Path(path)), device=str(device))
    draft.load_state_dict({k.removeprefix('drafter.'): v for k, v in tensors.items() if k.startswith('drafter.')}, strict=True)
    predictor.load_state_dict({k.removeprefix('progress_predictor.'): v for k, v in tensors.items() if k.startswith('progress_predictor.')}, strict=True)
    return tensors
