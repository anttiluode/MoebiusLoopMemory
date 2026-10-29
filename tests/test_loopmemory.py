import unittest
import numpy as np
from loopmemory import LoopConfig, LoopMemory, recurrence_matrix, fit_linear_readout, accuracy


class LoopMemoryTests(unittest.TestCase):
    def test_signed_return(self):
        R = recurrence_matrix(12, 'signed_shift', np.random.default_rng(1))
        self.assertTrue(np.allclose(np.linalg.matrix_power(R, 12), -np.eye(12)))
        self.assertTrue(np.allclose(np.linalg.matrix_power(R, 24), np.eye(12)))

    def test_cyclic_return(self):
        R = recurrence_matrix(12, 'cyclic', np.random.default_rng(1))
        self.assertTrue(np.allclose(np.linalg.matrix_power(R, 12), np.eye(12)))

    def test_batch_matches_single(self):
        net = LoopMemory(LoopConfig(dim=8))
        rng = np.random.default_rng(20)
        seq = rng.normal(size=(5, 7, 8))
        noise = rng.normal(scale=0.01, size=seq.shape)
        batch = net.encode_batch(seq, noise)
        single = np.stack([net.encode(a, b) for a, b in zip(seq, noise)])
        self.assertTrue(np.allclose(single, batch, atol=1e-12))

    def test_erase_removes_early_history(self):
        net = LoopMemory(LoopConfig(dim=8))
        a = np.zeros((8, 8)); b = a.copy()
        a[0, 0] = +1.; b[0, 0] = -1.
        self.assertGreater(np.linalg.norm(net.encode(a) - net.encode(b)), 0.0001)
        self.assertTrue(np.allclose(net.encode(a, erase_before=1), net.encode(b, erase_before=1)))

    def test_gating_is_not_accidentally_enabled(self):
        rng = np.random.default_rng(5)
        a = rng.normal(size=(8, 8))
        net0 = LoopMemory(LoopConfig(dim=8, gate=0))
        net1 = LoopMemory(LoopConfig(dim=8, gate=1.2))
        self.assertGreater(np.linalg.norm(net0.encode(a) - net1.encode(a)), 0.01)

    def test_linear_readout_is_not_quadratic(self):
        X = np.array([[-1, -1], [-1, +1], [+1, -1], [+1, +1]], dtype=float)
        y = X[:, 0] * X[:, 1]
        weights = fit_linear_readout(X, y)
        self.assertLessEqual(accuracy(X, y, weights), .75)

    def test_additive_conjugate_order_invariance_with_no_leak(self):
        cfg = LoopConfig(dim=8, geometry='identity', gate=0, leak=1., nonlinearity='linear', output_normalize=False)
        net = LoopMemory(cfg)
        x = np.zeros((4, 8)); x[0, 0] = 1.; x[1, 1] = 1.
        y = x.copy(); y[0], y[1] = x[1].copy(), x[0].copy()
        self.assertTrue(np.allclose(net.encode(x), net.encode(y)))


if __name__ == '__main__':
    unittest.main()

class ProbeTests(unittest.TestCase):
    def test_odd_reservoir_has_zero_four_way_interaction(self):
        from probe import contrast
        rng = np.random.default_rng(10)
        code = rng.normal(size=8) * 0.4
        net = LoopMemory(LoopConfig(dim=8, gate=0, output_normalize=False))
        self.assertLess(np.linalg.norm(contrast(net, 3, code)), 1e-10)

    def test_gate_creates_context_by_probe_interaction(self):
        from probe import contrast
        rng = np.random.default_rng(10)
        code = rng.normal(size=8) * 0.4
        net = LoopMemory(LoopConfig(dim=8, gate=1.2, output_normalize=False))
        self.assertGreater(np.linalg.norm(contrast(net, 3, code)), 0.001)
