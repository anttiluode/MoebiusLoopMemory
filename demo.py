"""Do real work: recognize whether two earlier events match using a linear output.

Example: python demo.py --history 101100101011
The readout has learned the XOR of the two bits preceding the latest bit.
"""
import argparse
import numpy as np
from loopmemory import LoopMemory, LoopConfig, fit_linear_readout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--history', default='101100101011', help='12-bit signal history')
    p.add_argument('--train', type=int, default=2500)
    args = p.parse_args()
    if len(args.history) != 12 or any(ch not in '01' for ch in args.history):
        p.error('history must be exactly 12 binary digits')
    rng = np.random.default_rng(20261029)
    code = rng.normal(size=16) * 0.6
    train_bits = rng.choice([-1, 1], size=(args.train, 12))
    # Fixed feature generator; learn only one linear output vector.
    net = LoopMemory(LoopConfig(geometry='signed_shift', gate=1.2), seed=99)
    data = train_bits[:, :, None] * code[None, None, :]
    states = net.encode_batch(data)
    labels = train_bits[:, -2] * train_bits[:, -3]
    weights = fit_linear_readout(states, labels)
    held_bits = rng.choice([-1, 1], size=(500, 12))
    held_states = net.encode_batch(held_bits[:, :, None] * code[None, None, :])
    held_labels = held_bits[:, -2] * held_bits[:, -3]
    acc = float(np.mean((np.column_stack([held_states, np.ones(500)]) @ weights >= 0) == (held_labels > 0)))
    bits = np.array([1 if ch == '1' else -1 for ch in args.history])
    s = net.encode((bits[:, None] * code[None, :]))
    margin = float(s @ weights[:-1] + weights[-1])
    # The question is whether event T-2 equals T-3, not whether the most recent bit is 1.
    print('History:', args.history)
    print('Question: do the TWO symbols just before the latest symbol match?')
    print('Readout:', 'MATCH' if margin >= 0 else 'DIFFERENT', '(margin %.4f)' % margin)
    print('Ground truth:', 'MATCH' if bits[-2] == bits[-3] else 'DIFFERENT')
    print('Held-out accuracy: %.1f%% (500 new independent sequences)' % (100 * acc))
    print('No symbol-by-symbol logic is hard-coded into the readout; it sees 16 final state values.')


if __name__ == '__main__':
    main()
