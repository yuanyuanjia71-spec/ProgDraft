import argparse
import json
from .io import read_json
from .models import load_target, make_models, load_weights
from .runtime import execute


def main():
    p = argparse.ArgumentParser(description='Greedy Ours decode; no FA at inference.')
    p.add_argument('--config', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--audio', required=True)
    p.add_argument('--target-path')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--k', type=int, default=8)
    a = p.parse_args()
    config = read_json(a.config)
    target, processor = load_target(config, a.device, target_path=a.target_path)
    draft, predictor = make_models(config, a.device)
    load_weights(draft, predictor, a.weights, a.device)
    draft.eval().requires_grad_(False)
    predictor.eval().requires_grad_(False)
    print(json.dumps(execute(target, processor, a.audio, a.device, draft, predictor, a.k), ensure_ascii=False))


if __name__ == '__main__':
    main()
