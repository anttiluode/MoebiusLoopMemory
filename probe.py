"""Causal probe: do different written histories change response to an identical later input?

Computes the four-way contrast:
D = [S(+history,+probe) - S(+history,-probe)]
  - [S(-history,+probe) - S(-history,-probe)].
For an odd, ungated reservoir with antipodal inputs D must be zero (sanity check).
"""
import argparse
import csv
import numpy as np
from loopmemory import LoopMemory, LoopConfig


def contrast(net: LoopMemory, delay: int, code: np.ndarray):
    result = {}
    for h in (-1., 1.):
        for p in (-1., 1.):
            seq = np.zeros((delay + 2, net.config.dim))
            seq[0] = h * code
            seq[-1] = p * code
            result[(h, p)] = net.encode(seq)
    return ((result[(1., 1.)] - result[(1., -1.)])
            - (result[(-1., 1.)] - result[(-1., -1.)]))


def run(delays, dim=16, gate=1.2, seed=99):
    rng = np.random.default_rng(20261029)
    code = rng.normal(size=dim) * 0.6
    rows = []
    for delay in delays:
        row = {'delay': delay}
        for geometry in ('signed_shift', 'cyclic', 'identity'):
            for enabled in (False, True):
                cfg = LoopConfig(dim=dim, geometry=geometry,
                                 gate=gate if enabled else 0, output_normalize=False)
                net = LoopMemory(cfg, seed=seed)
                delta = contrast(net, delay, code)
                row[f'{geometry}_{"gate" if enabled else "control"}'] = float(np.linalg.norm(delta))
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-delay', type=int, default=64)
    p.add_argument('--csv', default=None)
    args = p.parse_args()
    delays = sorted(set(range(0, args.max_delay + 1, 2)) | {16, 32, 48, 64})
    delays = [x for x in delays if x <= args.max_delay]
    rows = run(delays)
    print('Four-condition interaction contrast; raw final states; no learned decoder')
    print('Delay  signed+gate cyclic+gate identity+gate signed/no-gate')
    for row in rows:
        if row['delay'] in (0,2,4,8,12,16,20,24,28,32,40,48,64):
            print(f"{row['delay']:5}  {row['signed_shift_gate']:11.5f} {row['cyclic_gate']:11.5f} {row['identity_gate']:13.5f} {row['signed_shift_control']:14.6f}")
    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print('Wrote', args.csv)


if __name__ == '__main__':
    main()
