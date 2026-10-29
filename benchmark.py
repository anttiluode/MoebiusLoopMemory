"""Fixed-seed, independent held-out tests of fading memory and temporal binding.

Run: python benchmark.py --seeds 5 --train 1200 --test 600 --json results.json
All rows use identical train/test sequences and identical input-noise tensors.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from loopmemory import LoopConfig, LoopMemory, fit_linear_readout, accuracy


def build_data(seed: int, train: int, test: int, steps: int, dim: int, noise_sd: float):
    rng = np.random.default_rng(seed)
    total = train + test
    # One strict antipodal stimulus channel (same physical input magnitude).
    vector = rng.normal(size=dim) * 0.6
    bits = rng.choice(np.array([-1, 1]), size=(total, steps))
    inputs = bits[:, :, None] * vector[None, None, :]
    noise = rng.normal(scale=noise_sd, size=inputs.shape)
    return bits, inputs, noise


def conditions():
    return {
        'signed_gated': LoopConfig(geometry='signed_shift', gate=1.2),
        'cyclic_gated': LoopConfig(geometry='cyclic', gate=1.2),
        'orthogonal_gated': LoopConfig(geometry='orthogonal', gate=1.2),
        'identity_gated': LoopConfig(geometry='identity', gate=1.2),
        'signed_no_gate': LoopConfig(geometry='signed_shift', gate=0.0),
        'identity_no_gate': LoopConfig(geometry='identity', gate=0.0),
        'signed_linear': LoopConfig(geometry='signed_shift', gate=0.0, nonlinearity='linear'),
        'identity_linear': LoopConfig(geometry='identity', gate=0.0, nonlinearity='linear'),
    }


def evaluate(seed: int, train: int = 1200, test: int = 600,
             steps: int = 12, dim: int = 16, noise_sd: float = 0.01):
    bits, inputs, noise = build_data(seed, train, test, steps, dim, noise_sd)
    targets = {f'lag_{lag}': bits[:, steps - 1 - lag] for lag in (0, 2, 4, 6, 8) if lag < steps}
    # The labels are balanced in expectation and require a bilinear representation.
    targets.update({'xor_1_2': bits[:, -2] * bits[:, -3],
                    'xor_2_3': bits[:, -3] * bits[:, -4]})
    scores = {}
    for name, cfg in conditions().items():
        cfg = LoopConfig(**(asdict(cfg) | {'dim': dim}))
        network = LoopMemory(cfg, seed=seed + 1000)
        states = network.encode_batch(inputs, noise)
        row = {}
        for task, y in targets.items():
            readout = fit_linear_readout(states[:train], y[:train])
            row[task] = accuracy(states[train:], y[train:], readout)
        # Causal knockout: erase all history before the final two inputs, which
        # destroys earlier-lag targets; do NOT interpret as physical lesion.
        erased = network.encode_batch(inputs, noise, erase_before=steps-2)
        y = targets['xor_2_3']
        w = fit_linear_readout(erased[:train], y[:train])
        row['xor_2_3_erased'] = accuracy(erased[train:], y[train:], w)
        scores[name] = row
    return scores


def run(seeds: int, train: int, test: int, steps: int, dim: int, noise_sd: float):
    by_seed = [evaluate(20261029 + i * 101, train, test, steps, dim, noise_sd) for i in range(seeds)]
    summary = {}
    for model in conditions():
        summary[model] = {}
        for task in by_seed[0][model]:
            v = np.array([s[model][task] for s in by_seed])
            summary[model][task] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if seeds > 1 else 0.0}
    return {'config': dict(seeds=seeds, train=train, test=test, steps=steps, dim=dim, noise_sd=noise_sd),
            'by_seed': by_seed, 'summary': summary}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--train', type=int, default=1200)
    p.add_argument('--test', type=int, default=600)
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--dim', type=int, default=16)
    p.add_argument('--noise', type=float, default=0.01)
    p.add_argument('--json', default=None)
    args = p.parse_args()
    report = run(args.seeds, args.train, args.test, args.steps, args.dim, args.noise)
    print('Recurrent temporal-memory benchmark (linear readout only)')
    print('Model               lag4    lag6    XOR(1,2)  XOR(2,3)   erased XOR(2,3)')
    for model, scores in report['summary'].items():
        def val(k):
            return f"{100 * scores[k]['mean']:5.1f}%"
        print(f'{model:<19} {val("lag_4")}  {val("lag_6")}  {val("xor_1_2")}     {val("xor_2_3")}        {val("xor_2_3_erased")}')
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Wrote {args.json}')


if __name__ == '__main__':
    main()
